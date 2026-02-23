from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import ebooklib
import fitz
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from ebooklib import epub
from spacy.language import Language


@dataclass
class ExtractionConfig:
    header_footer_ratio: float = 0.10
    spacy_model: str = "en_core_web_sm"


class BaseTextIterator(ABC):
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        cleaned = re.sub(r"[ \t\f\v]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 2000) -> Iterator[str]:
        """
        将长文本按“自然段优先、长度兜底”切成块。
        - 优先按空行段落分割
        - 若段落过长，再按固定长度切分
        """
        cleaned = BaseTextIterator.normalize_whitespace(text)
        if not cleaned:
            return

        paragraphs = [p.strip() for p in re.split(r"\n\n+", cleaned) if p.strip()]
        buffer = ""

        for para in paragraphs:
            if len(para) > chunk_size:
                if buffer:
                    yield buffer
                    buffer = ""
                start = 0
                while start < len(para):
                    piece = para[start : start + chunk_size].strip()
                    if piece:
                        yield piece
                    start += chunk_size
                continue

            candidate = para if not buffer else f"{buffer}\n\n{para}"
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    yield buffer
                buffer = para

        if buffer:
            yield buffer

    @abstractmethod
    def yield_chunks(self) -> Iterator[str]:
        pass


class PDFIterator(BaseTextIterator):
    def __init__(self, file_path: str | Path, header_footer_ratio: float) -> None:
        super().__init__(file_path=file_path)
        self.header_footer_ratio = header_footer_ratio

    def _extract_page_body_text(self, page: fitz.Page) -> str:
        rect = page.rect
        page_height = rect.height
        margin = page_height * self.header_footer_ratio
        body_clip = fitz.Rect(rect.x0, rect.y0 + margin, rect.x1, rect.y1 - margin)
        return page.get_text("text", clip=body_clip)

    @staticmethod
    def _clean_pdf_text(raw_text: str) -> str:
        # 正则说明：将类似 accom-\nmodate 的断词拼接回来。
        # (\w+)-\s*\n\s*(\w+) 匹配“单词片段-换行-单词片段”，替换为 \1\2。
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", raw_text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return BaseTextIterator.normalize_whitespace(text)

    def yield_chunks(self) -> Iterator[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {self.file_path}")

        page_texts = []
        try:
            with fitz.open(self.file_path) as doc:
                if doc.needs_pass:
                    raise ValueError(f"PDF 文件已加密，无法直接读取: {self.file_path}")
                for page in doc:
                    page_texts.append(self._extract_page_body_text(page))
        except fitz.FileDataError as exc:
            raise ValueError(f"PDF 文件损坏或格式异常，无法解析: {self.file_path}") from exc
        except RuntimeError as exc:
            raise RuntimeError(f"读取 PDF 失败: {self.file_path}, 详情: {exc}") from exc

        cleaned_text = self._clean_pdf_text("\n".join(page_texts))
        for chunk in self.chunk_text(cleaned_text):
            yield chunk


class EpubIterator(BaseTextIterator):
    def yield_chunks(self) -> Iterator[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"EPUB 文件不存在: {self.file_path}")

        try:
            book = epub.read_epub(str(self.file_path))
        except Exception as exc:
            raise RuntimeError(f"读取 EPUB 失败: {self.file_path}, 详情: {exc}") from exc

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            html_bytes = item.get_content()
            soup = BeautifulSoup(html_bytes, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            text = self.normalize_whitespace(text)
            if not text:
                continue

            for chunk in self.chunk_text(text):
                yield chunk


class TxtIterator(BaseTextIterator):
    def yield_chunks(self) -> Iterator[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"TXT 文件不存在: {self.file_path}")

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                text = f.read()
                for chunk in self.chunk_text(text):
                    yield chunk
        except UnicodeDecodeError as exc:
            raise ValueError(f"TXT 文件编码读取失败，请使用 UTF-8 编码: {self.file_path}") from exc


class VocabularySentenceExtractor:
    def __init__(self, config: ExtractionConfig | None = None, max_extracted_words: int = 1000) -> None:
        self.config = config or ExtractionConfig()
        self.vocabulary: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.nlp: Language = self._load_spacy_model(self.config.spacy_model)
        self.max_extracted_words = max_extracted_words
        self.valid_word_whitelist: Set[str] = self._load_valid_words_file(Path("dicts/valid_words.txt"))

    @staticmethod
    def _load_valid_words_file(file_path: Path) -> Set[str]:
        """加载轻量本地合法词白名单（可选）。"""
        if not file_path.exists():
            return set()
        words: Set[str] = set()
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        words.add(w)
        except Exception:
            return set()
        return words

    @staticmethod
    def _load_spacy_model(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError:
            # 兜底策略：在容器无法联网下载模型时，使用轻量英文管线保证流程可运行。
            # 注意：该模式下词形还原精度会下降（通常退化为用词面小写），但不会阻断整体任务。
            print(
                f"[Warning] 无法加载 spaCy 模型 '{model_name}'，已回退到 spacy.blank('en')。"
                f"如需更高词形还原精度，请安装模型：python -m spacy download {model_name}"
            )
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    @staticmethod
    def _token_lemma(token) -> str:
        """
        兼容无完整模型时的 lemma 退化：
        - 优先用 token.lemma_
        - 若为空或占位值，则回退到 token.text
        """
        lemma = (getattr(token, "lemma_", "") or "").strip().lower()
        if not lemma or lemma == "-pron-":
            lemma = (getattr(token, "text", "") or "").strip().lower()
        return lemma

    def _is_valid_word(self, lemma: str) -> bool:
        """
        强校验：过滤 OCR 碎片与无效词。
        1) 长度 > 2
        2) 纯字母
        3) 若是 OOV，则必须在本地 valid_words 白名单中
        """
        if len(lemma) <= 2:
            return False
        if not lemma.isalpha():
            return False

        # 当模型词表可用时优先走 OOV 判定；否则退化到白名单存在性判定
        lexeme = self.nlp.vocab[lemma]
        if getattr(lexeme, "is_oov", False):
            return lemma in self.valid_word_whitelist or lemma in self.vocabulary
        return True

    @staticmethod
    def _append_filtered_sample(samples: List[str], token_text: str, max_samples: int = 10) -> None:
        text = (token_text or "").strip().lower()
        if not text:
            return
        if len(samples) >= max_samples:
            return
        if text not in samples:
            samples.append(text)

    def load_vocabulary_from_csv(self, csv_path: str | Path, min_level: float = 0) -> Set[str]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"词汇 CSV 文件不存在: {path}")

        vocabulary: Set[str] = set()
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"CSV 文件编码读取失败，请使用 UTF-8 编码: {path}") from exc
        except Exception as exc:
            raise ValueError(f"CSV 文件读取失败: {path}, 详情: {exc}") from exc

        if "word" not in df.columns:
            raise ValueError("CSV 格式不正确：必须包含名为 'word' 的列")

        working_df = df.copy()
        if "level" in working_df.columns:
            level_series = pd.to_numeric(working_df["level"], errors="coerce")
            working_df = working_df[level_series >= float(min_level)]
        else:
            if float(min_level) > 0:
                print("[Warning] 词库缺少 level 列，已忽略最低等级筛选并加载全部单词")

        for value in working_df["word"].tolist():
            raw_word = str(value).strip().lower()
            if raw_word and raw_word != "nan":
                vocabulary.add(raw_word)

        self.vocabulary = vocabulary
        return self.vocabulary

    def load_blacklist_from_csv(self, csv_path: str | Path) -> Set[str]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"黑名单 CSV 文件不存在: {path}")

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"黑名单 CSV 编码读取失败，请使用 UTF-8 编码: {path}") from exc
        except Exception as exc:
            raise ValueError(f"黑名单 CSV 读取失败: {path}, 详情: {exc}") from exc

        if "word" in df.columns:
            series = df["word"]
        elif len(df.columns) >= 1:
            series = df[df.columns[0]]
        else:
            raise ValueError("黑名单 CSV 格式不正确：至少需要一列单词数据")

        blacklist: Set[str] = set()
        for value in series.tolist():
            raw_word = str(value).strip().lower()
            if raw_word and raw_word != "nan":
                blacklist.add(raw_word)

        self.blacklist = blacklist
        return self.blacklist

    def _build_iterator(self, file_path: str | Path) -> BaseTextIterator:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PDFIterator(path, self.config.header_footer_ratio)
        if suffix == ".epub":
            return EpubIterator(path)
        if suffix == ".txt":
            return TxtIterator(path)
        raise ValueError("不支持的文件格式")

    def extract_with_progress(
        self,
        file_path: str | Path,
        filter_strategy: str = "whitelist",
        max_extracted_words: int | None = None,
        progress_every_sentences: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """
        以生成器方式执行提取：
        - 每处理 progress_every_sentences 句，返回一次解析进度
        - 达到最大提取词数后触发安全熔断
        - 最后返回 done + data
        """
        strategy = (filter_strategy or "whitelist").strip().lower()
        if strategy not in {"whitelist", "blacklist"}:
            raise ValueError("filter_strategy 仅支持 'whitelist' 或 'blacklist'")

        if strategy == "whitelist" and not self.vocabulary:
            raise ValueError("词库为空：请先调用 load_vocabulary_from_csv() 加载词汇表")
        if strategy == "blacklist" and not self.blacklist:
            raise ValueError("黑名单为空：请先调用 load_blacklist_from_csv() 加载黑名单")

        limit = self.max_extracted_words if max_extracted_words is None else int(max_extracted_words)
        iterator = self._build_iterator(file_path)
        matched: Dict[str, str] = {}
        processed_sentences = 0
        processed_docs = 0
        scanned_words = 0
        filtered_noise_words = 0
        filtered_samples: List[str] = []
        text_batches: List[str] = list(iterator.yield_chunks())

        for doc in self.nlp.pipe(text_batches, batch_size=50):
            processed_docs += 1
            for sent in doc.sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue

                processed_sentences += 1
                for token in sent:
                    if token.is_space or token.is_punct:
                        continue

                    scanned_words += 1

                    if not token.is_alpha:
                        filtered_noise_words += 1
                        self._append_filtered_sample(filtered_samples, token.text)
                        continue
                    lemma = self._token_lemma(token)
                    if not lemma or lemma in matched:
                        continue
                    if not self._is_valid_word(lemma):
                        filtered_noise_words += 1
                        self._append_filtered_sample(filtered_samples, lemma)
                        continue
                    if strategy == "whitelist":
                        if lemma not in self.vocabulary:
                            continue
                    else:
                        if lemma in self.blacklist:
                            continue

                    matched[lemma] = sentence
                    if limit > 0 and len(matched) >= limit:
                        yield {
                            "status": "parsing",
                            "message": (
                                f"正在解析... 已处理 {processed_sentences} 句（{processed_docs} 块），"
                                f"已提取 {len(matched)} 个候选词（已达到上限 {limit}，提前终止）"
                            ),
                        }
                        yield {"status": "done", "data": matched, "truncated": True, "limit": limit}
                        print(
                            f"[清洗报告] 共扫描 {scanned_words} 个词，保留 {len(matched)} 个。"
                            f"已过滤 {filtered_noise_words} 个噪点/非法词。"
                        )
                        print(f"[样例] 被杀掉的词 (前10个): {filtered_samples}")
                        return

                if progress_every_sentences > 0 and processed_sentences % progress_every_sentences == 0:
                    yield {
                        "status": "parsing",
                        "message": (
                            f"正在解析... 已处理 {processed_sentences} 句（{processed_docs} 块），"
                            f"已提取 {len(matched)} 个候选词"
                        ),
                    }

        yield {"status": "done", "data": matched, "truncated": False, "limit": limit}
        print(
            f"[清洗报告] 共扫描 {scanned_words} 个词，保留 {len(matched)} 个。"
            f"已过滤 {filtered_noise_words} 个噪点/非法词。"
        )
        print(f"[样例] 被杀掉的词 (前10个): {filtered_samples}")

    @staticmethod
    def _collect_done_data(progress_stream: Iterator[Dict[str, Any]]) -> Dict[str, str]:
        for event in progress_stream:
            if event.get("status") == "done":
                data = event.get("data", {})
                if isinstance(data, dict):
                    return data
        return {}

    def extract_from_file(self, file_path: str | Path) -> Dict[str, str]:
        return self._collect_done_data(
            self.extract_with_progress(
                file_path,
                filter_strategy="whitelist",
                max_extracted_words=self.max_extracted_words,
            )
        )

    def extract_all_words_from_file(self, file_path: str | Path) -> Dict[str, str]:
        """
        不依赖本地词库，直接提取电子书中的候选词（按 lemma 去重，保留首次出现原句）。
        用于“未上传 CSV 词库”时的全量识别模式。
        """
        return self._collect_done_data(
            self.extract_with_progress(
                file_path,
                filter_strategy="blacklist",
                max_extracted_words=self.max_extracted_words,
            )
        )

    def extract_matched_words(self, file_path: str | Path) -> Dict[str, str]:
        return self.extract_from_file(file_path)


if __name__ == "__main__":
    sample_text_path = "./sample_book.txt"
    sample_vocab_csv = "./ielts_words.csv"

    try:
        extractor = VocabularySentenceExtractor()
        extractor.load_vocabulary_from_csv(sample_vocab_csv)
        result = extractor.extract_from_file(sample_text_path)

        print(f"共匹配到 {len(result)} 个去重词汇，展示前 5 条：\n")
        for idx, (word, sentence) in enumerate(result.items()):
            if idx >= 5:
                break
            print(f"{idx + 1}. {word} -> {sentence}")
    except Exception as e:
        print(f"执行失败: {e}")
