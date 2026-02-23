from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp


class FatalAPIError(Exception):
    pass


class TransientAPIError(Exception):
    pass


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model_name: str
    api_type: str = "openai_compatible"
    request_timeout_seconds: float = 30.0


class BaseProtocolAdapter(ABC):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        request_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.request_timeout_seconds = request_timeout_seconds

    @staticmethod
    def _normalize_openai_v1_base(base_url: str) -> str:
        base = (base_url or "").strip().rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    async def _post_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status >= 400:
                    body_preview = (await resp.text())[:300]
                    message = f"HTTP {resp.status}, body={body_preview}"
                    if resp.status in {408, 425, 429, 500, 502, 503, 504}:
                        raise TransientAPIError(message)
                    raise FatalAPIError(message)
                raw_text = await resp.text()
                if not raw_text.strip():
                    raise FatalAPIError("API 返回空响应体")
                return json.loads(raw_text)
        except asyncio.TimeoutError as exc:
            raise TransientAPIError(f"请求超时（>{self.request_timeout_seconds}s）") from exc
        except aiohttp.ClientError as exc:
            raise TransientAPIError(f"网络请求失败: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise FatalAPIError("响应体不是合法 JSON（API 返回异常）") from exc

    async def _post_sse_extract_text(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> str:
        """
        针对 /responses 的流式事件（SSE）解析文本。
        兼容 response.output_text.delta / response.output_text.done 两类事件。
        """
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status >= 400:
                    body_preview = (await resp.text())[:300]
                    message = f"HTTP {resp.status}, body={body_preview}"
                    if resp.status in {408, 425, 429, 500, 502, 503, 504}:
                        raise TransientAPIError(message)
                    raise FatalAPIError(message)

                content = await resp.text()
                if not content.strip():
                    raise FatalAPIError("API 返回空响应体")

                blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
                delta_parts: List[str] = []
                done_text = ""
                for block in blocks:
                    data_lines: List[str] = []
                    for line in block.split("\n"):
                        if line.startswith("data: "):
                            data_lines.append(line[6:].strip())

                    if not data_lines:
                        continue
                    data_text = "\n".join(data_lines)
                    if data_text == "[DONE]":
                        continue

                    try:
                        obj = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue

                    event_type = str(obj.get("type", "")).strip()
                    if event_type == "response.output_text.delta":
                        delta = obj.get("delta")
                        if isinstance(delta, str) and delta:
                            delta_parts.append(delta)
                    elif event_type == "response.output_text.done":
                        text = obj.get("text")
                        if isinstance(text, str) and text.strip():
                            done_text = text

                if done_text.strip():
                    return done_text
                merged = "".join(delta_parts).strip()
                if merged:
                    return merged

                raise FatalAPIError("responses 流式事件中未提取到文本内容")

        except asyncio.TimeoutError as exc:
            raise TransientAPIError(f"请求超时（>{self.request_timeout_seconds}s）") from exc
        except aiohttp.ClientError as exc:
            raise TransientAPIError(f"网络请求失败: {exc}") from exc

    @abstractmethod
    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> str:
        pass


class OpenAIChatAdapter(BaseProtocolAdapter):
    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> str:
        url = f"{self._normalize_openai_v1_base(self.base_url)}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = await self._post_json(session, url, headers, payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FatalAPIError("API 返回了意外结构，未找到 choices[0].message.content") from exc


class OpenAIResponsesAdapter(BaseProtocolAdapter):
    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> str:
        url = f"{self._normalize_openai_v1_base(self.base_url)}/responses"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "input": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        return await self._post_sse_extract_text(session, url, headers, payload)


class GeminiAdapter(BaseProtocolAdapter):
    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> str:
        base = self.base_url.strip().rstrip("/")
        if base:
            prefix = base if base.endswith("/v1beta") else f"{base}/v1beta"
        else:
            prefix = "https://generativelanguage.googleapis.com/v1beta"
        url = f"{prefix}/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        data = await self._post_json(session, url, headers, payload)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FatalAPIError("API 返回了意外结构，未找到 candidates[0].content.parts[0].text") from exc


class LLMClient:
    COLLOCATION_EMPTY_MARKERS = {"none", "null", "n/a", "no collocation", "na", "nil", "无", "没有"}

    SYSTEM_PROMPT_BOTH = (
        "你是一个严谨的英语词汇助手。"
        "你必须只输出 JSON 对象，不要输出任何额外说明。"
        "JSON 必须包含且仅包含以下字段："
        "word, part_of_speech, context_meaning, collocation, collocation_meaning。"
        "其中 word 是英文单词原样返回；"
        "part_of_speech 为该词在当前语境下的简写词性（如 n., v., adj., adv.）；"
        "context_meaning 为该词在给定句子中的中文释义；"
        "collocation 为一个高频英文搭配；"
        "collocation_meaning 为该搭配的中文释义。"
        "If no fixed collocation is found, return empty strings for 'collocation' and 'collocation_meaning'."
        "DO NOT skip the word."
    )
    SYSTEM_PROMPT_WORD_ONLY = (
        "你是一个严谨的英语词汇助手。"
        "你必须只输出 JSON 对象，不要输出任何额外说明。"
        "JSON 必须包含且仅包含字段：word, part_of_speech, context_meaning。"
        "word 为原样返回的单词；part_of_speech 为简写词性（如 n., v., adj., adv.）；"
        "context_meaning 为该词在给定句子中的中文释义。"
    )
    SYSTEM_PROMPT_COLLOCATION_ONLY = (
        "你是一个严谨的英语词汇助手。"
        "你必须只输出 JSON 对象，不要输出任何额外说明。"
        "JSON 必须包含且仅包含字段：word, part_of_speech, collocation, collocation_meaning。"
        "word 为原样返回的单词；part_of_speech 为简写词性（如 n., v., adj., adv.）；"
        "collocation 为固定搭配；collocation_meaning 为搭配中文释义。"
        "If no fixed collocation is found, return empty strings for 'collocation' and 'collocation_meaning'."
        "DO NOT skip the word."
    )

    def __init__(
        self,
        api_type: str | LLMConfig,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if isinstance(api_type, LLMConfig):
            self.api_type = api_type.api_type
            self.base_url = api_type.base_url
            self.api_key = api_type.api_key
            self.model_name = api_type.model_name
            self.request_timeout_seconds = api_type.request_timeout_seconds
            self.adapter = self._build_adapter()
            return

        if base_url is None or api_key is None or model_name is None:
            raise ValueError("LLMClient 初始化需要 api_type, base_url, api_key, model_name")

        self.api_type = api_type
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.request_timeout_seconds = request_timeout_seconds
        self.adapter = self._build_adapter()

    def _build_adapter(self) -> BaseProtocolAdapter:
        adapter_map = {
            "openai_official": OpenAIChatAdapter,
            "openai_compatible": OpenAIChatAdapter,
            "ollama_local": OpenAIChatAdapter,
            "openai_responses": OpenAIResponsesAdapter,
            "gemini_official": GeminiAdapter,
        }
        adapter_cls = adapter_map.get(self.api_type)
        if adapter_cls is None:
            raise FatalAPIError(f"不支持的 api_type: {self.api_type}")
        return adapter_cls(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            request_timeout_seconds=self.request_timeout_seconds,
        )

    async def preflight_check(self, session: aiohttp.ClientSession) -> bool:
        await self.generate(session, word="hello", sentence="hi", extract_mode="word_only")
        return True

    @staticmethod
    def _build_user_prompt(word: str, sentence: str) -> str:
        return (
            "请基于下面的单词与上下文句子，返回 JSON：\n"
            f"word: {word}\n"
            f"sentence: {sentence}\n"
        )

    @staticmethod
    def strip_markdown_code_fence(text: str) -> str:
        # 详细说明（重点注释：正则清洗）
        # 模型常返回：```json\n{...}\n``` 或 ```\n{...}\n```
        # 我们使用一个“首尾严格匹配”的正则，只在整段文本被代码块包裹时才剥离：
        # ^\s*```(?:json)?\s*   -> 开头允许空白，匹配 ``` 或 ```json
        # (.*?)                  -> 非贪婪捕获中间主体内容（DOTALL 允许跨行）
        # \s*```\s*$             -> 结尾必须是闭合 ```
        # 这样可以尽量避免误删正文里的普通反引号。
        candidate = text.strip()
        fenced_match = re.match(
            r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
            candidate,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            return fenced_match.group(1).strip()
        return candidate

    @staticmethod
    def parse_llm_json(raw_text: str, extract_mode: str = "both") -> Dict[str, Any]:
        cleaned = LLMClient.strip_markdown_code_fence(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM 返回不是合法 JSON，原始片段: {cleaned[:200]}") from exc
        if not isinstance(data, dict):
            raise ValueError("LLM 返回 JSON 不是对象类型（dict）")

        # 解析兜底：字段缺失或 null 时，统一补空字符串，避免单词被阻断。
        normalized: Dict[str, Any] = {
            "word": "",
            "part_of_speech": "",
            "context_meaning": "",
            "collocation": "",
            "collocation_meaning": "",
        }
        for key in normalized.keys():
            value = data.get(key, "")
            if value is None:
                normalized[key] = ""
            else:
                normalized[key] = str(value).strip()

        # 强清洗：搭配字段出现占位词时统一转空字符串，确保不会因“无搭配”丢词。
        collocation_text = normalized.get("collocation", "").strip()
        if collocation_text.lower() in LLMClient.COLLOCATION_EMPTY_MARKERS:
            normalized["collocation"] = ""

        collocation_meaning_text = normalized.get("collocation_meaning", "").strip()
        if collocation_meaning_text.lower() in LLMClient.COLLOCATION_EMPTY_MARKERS:
            normalized["collocation_meaning"] = ""

        # 若搭配为空，则搭配释义也清空，避免脏数据组合。
        if not normalized["collocation"]:
            normalized["collocation_meaning"] = ""

        if not normalized["word"]:
            raise ValueError("LLM 返回 JSON 缺少有效 word 字段")

        if extract_mode == "word_only":
            # 单词模式下至少要有词性或语境释义之一
            if not normalized["part_of_speech"] and not normalized["context_meaning"]:
                raise ValueError("LLM 返回内容过空：word_only 至少需要词性或语境释义")
        elif extract_mode == "collocation_only":
            # 允许搭配为空（非阻断），但词性应尽量给出
            pass

        return normalized

    @staticmethod
    def _resolve_system_prompt(extract_mode: str) -> str:
        if extract_mode == "word_only":
            return LLMClient.SYSTEM_PROMPT_WORD_ONLY
        if extract_mode == "collocation_only":
            return LLMClient.SYSTEM_PROMPT_COLLOCATION_ONLY
        return LLMClient.SYSTEM_PROMPT_BOTH

    async def generate(
        self,
        session: aiohttp.ClientSession,
        word: str,
        sentence: str,
        extract_mode: str = "both",
    ) -> Dict[str, Any]:
        prompt = f"{self._resolve_system_prompt(extract_mode)}\n\n{self._build_user_prompt(word, sentence)}"
        content = await self.adapter.generate(session, prompt)

        parsed = self.parse_llm_json(content, extract_mode=extract_mode)
        if str(parsed.get("word", "")).strip().lower() != word.strip().lower():
            parsed["word"] = word
        return parsed

    async def request_word_info(
        self,
        session: aiohttp.ClientSession,
        word: str,
        sentence: str,
        extract_mode: str = "both",
    ) -> Dict[str, Any]:
        return await self.generate(session, word, sentence, extract_mode=extract_mode)


class CheckpointManager:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.processed_keys: Set[str] = set()
        self.dirty_keys: Set[str] = set()

    @staticmethod
    def _book_scope(book_key: str) -> str:
        text = (book_key or "").strip().lower()
        match = re.match(r"^([0-9a-f]{8})_", text)
        if match:
            return match.group(1)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _build_unique_key(cls, book_key: str, extract_mode: str, word: str) -> str:
        return f"{cls._book_scope(book_key)}_{extract_mode.strip().lower()}_{word.strip().lower()}"

    def load_processed_words(self) -> Set[str]:
        self.processed_keys.clear()
        self.dirty_keys.clear()
        if not self.checkpoint_path.exists():
            return self.processed_keys

        # 详细说明（重点注释：严格断点续传）
        # JSONL 每行一个 JSON。读取时只提取 word 并放入 set：
        # - set 查询是 O(1)，过滤待处理任务效率高
        # - 即使文件很大，也只需按行流式读取，不会一次性吃满内存
        with self.checkpoint_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    print(f"[Checkpoint] 警告：第 {line_no} 行 JSON 无法解析，已跳过")
                    continue
                book_key = str(row.get("book_key") or row.get("book_name") or "").strip()
                extract_mode = str(row.get("extract_mode", "")).strip()
                word = str(row.get("word", "")).strip().lower()
                if book_key and extract_mode and word:
                    key = self._build_unique_key(book_key, extract_mode, word)
                    part_of_speech = str(row.get("part_of_speech", "") or "").strip()
                    if part_of_speech:
                        self.processed_keys.add(key)
                        self.dirty_keys.discard(key)
                    else:
                        # 脏缓存：关键字段缺失，后续需要强制重跑补全
                        if key not in self.processed_keys:
                            self.dirty_keys.add(key)
        return self.processed_keys

    def filter_pending(
        self,
        word_sentence_map: Dict[str, str],
        book_key: str,
        extract_mode: str,
    ) -> List[Tuple[str, str]]:
        pending: List[Tuple[str, str]] = []
        for word, sentence in word_sentence_map.items():
            key = self._build_unique_key(book_key, extract_mode, word)
            if key in self.processed_keys:
                continue
            # dirty_keys 显式保留在待处理列表中，触发 reprocess
            if key in self.dirty_keys or key not in self.processed_keys:
                pending.append((word, sentence))
        return pending

    def append_result(self, data: Dict[str, Any], book_key: str, extract_mode: str) -> None:
        parent_dir = self.checkpoint_path.parent
        if parent_dir and not parent_dir.exists():
            os.makedirs(parent_dir, exist_ok=True)

        record = dict(data)
        record["book_key"] = book_key
        record["extract_mode"] = extract_mode

        line = json.dumps(record, ensure_ascii=False)
        with self.checkpoint_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        word = str(record.get("word", "")).strip().lower()
        if word:
            key = self._build_unique_key(book_key, extract_mode, word)
            part_of_speech = str(record.get("part_of_speech", "") or "").strip()
            if part_of_speech:
                self.processed_keys.add(key)
                self.dirty_keys.discard(key)
            else:
                self.dirty_keys.add(key)


class AsyncRunner:
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(
        self,
        llm_client: LLMClient,
        checkpoint_manager: CheckpointManager,
        max_concurrency: int = 10,
    ) -> None:
        self.llm_client = llm_client
        self.checkpoint_manager = checkpoint_manager
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_one(
        self,
        session: aiohttp.ClientSession,
        word: str,
        sentence: str,
        book_key: str,
        extract_mode: str = "both",
    ) -> Dict[str, Any]:
        async with self.semaphore:
            key = self.checkpoint_manager._build_unique_key(book_key, extract_mode, word)
            # 脏缓存（例如缺 part_of_speech）必须重跑补全，不做跳过。
            if key in self.checkpoint_manager.dirty_keys:
                pass
            result = await self.llm_client.generate(session, word, sentence, extract_mode=extract_mode)
            result["sentence"] = sentence
            self.checkpoint_manager.append_result(result, book_key=book_key, extract_mode=extract_mode)
            return result

    async def _process_one_with_word(
        self,
        session: aiohttp.ClientSession,
        word: str,
        sentence: str,
        book_key: str,
        extract_mode: str = "both",
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            result = await self._process_one(session, word, sentence, book_key, extract_mode)
            return word, result
        except FatalAPIError as exc:
            raise FatalAPIError(f"word={word}, error={exc}") from exc
        except TransientAPIError as exc:
            raise TransientAPIError(f"word={word}, error={exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"word={word}, error={exc}") from exc

    async def run(
        self,
        pending_items: List[Tuple[str, str]],
        progress_queue: Optional[asyncio.Queue[str]] = None,
        book_key: str = "",
        extract_mode: str = "both",
    ) -> List[Dict[str, Any]]:
        if not pending_items:
            if progress_queue is not None:
                await progress_queue.put("没有待处理任务，已全部命中 checkpoint。")
            else:
                print("没有待处理任务，已全部命中 checkpoint。")
            return []

        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(limit=100)
        consecutive_errors = 0
        results: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                asyncio.create_task(self._process_one_with_word(session, word, sentence, book_key, extract_mode))
                for word, sentence in pending_items
            ]

            try:
                for done_task in asyncio.as_completed(tasks):
                    try:
                        word, result = await done_task
                        results.append(result)
                        consecutive_errors = 0
                        if progress_queue is not None:
                            await progress_queue.put(f"[OK] {word}")
                    except FatalAPIError as exc:
                        word = "<unknown>"
                        if progress_queue is not None:
                            await progress_queue.put(f"[ERROR] word={word}, error={exc}")
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                    except Exception as exc:
                        word = "<unknown>"
                        consecutive_errors += 1
                        if progress_queue is not None:
                            await progress_queue.put(f"[ERROR] word={word}, error={exc}")
                        if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                            raise RuntimeError(
                                f"连续失败超过阈值({self.MAX_CONSECUTIVE_ERRORS})，已触发熔断并终止任务"
                            )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        return results


if __name__ == "__main__":
    base_url = ""
    api_key = ""
    model_name = "gpt-4o-mini"

    sample_word_sentence_map = {
        "accommodate": "The hotel can accommodate up to 500 guests during peak season.",
        "mitigate": "Vaccination can mitigate the spread of infectious diseases.",
        "comprehensive": "She conducted a comprehensive review of the literature.",
    }

    checkpoint_file = "./cache_results.jsonl"

    async def _demo() -> None:
        if not base_url:
            print("[提示] 你还没有填写 base_url/api_key。请先填写后再运行真实请求。")

        config = LLMConfig(
            base_url=base_url or "http://localhost:11434",
            api_key=api_key,
            model_name=model_name,
            request_timeout_seconds=30,
        )
        client = LLMClient(config)
        checkpoint = CheckpointManager(checkpoint_file)

        processed = checkpoint.load_processed_words()
        print(f"已从 checkpoint 恢复 {len(processed)} 个已处理单词")

        pending = checkpoint.filter_pending(sample_word_sentence_map, book_key="demo.txt", extract_mode="both")
        print(f"本次待处理任务数: {len(pending)}")

        runner = AsyncRunner(client, checkpoint, max_concurrency=10)
        results = await runner.run(pending, book_key="demo.txt", extract_mode="both")

        print(f"本次成功写入结果数: {len(results)}")
        print(f"checkpoint 文件位置: {Path(checkpoint_file).resolve()}")
        for idx, item in enumerate(results[:3], start=1):
            print(f"{idx}. {json.dumps(item, ensure_ascii=False)}")

    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        print("用户中断执行。")
    except Exception as e:
        print(f"执行失败: {e}")
