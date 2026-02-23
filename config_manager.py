from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


class LLMConfigManager:
    def __init__(self, settings_path: str | Path = "llm_settings.json") -> None:
        self.settings_path = Path(settings_path)

    def _default_settings(self) -> Dict[str, Any]:
        return {
            "providers": {},
            "default_provider": "",
            "default_model": "",
        }

    @staticmethod
    def _normalize_api_type(api_type: str) -> str:
        raw = (api_type or "").strip()
        mapping = {
            "openai-responses": "openai_responses",
            "openai_responses": "openai_responses",
            "openai-compatible": "openai_compatible",
            "openai_compatible": "openai_compatible",
            "openai-official": "openai_official",
            "openai_official": "openai_official",
            "gemini-official": "gemini_official",
            "gemini_official": "gemini_official",
            "ollama-local": "ollama_local",
            "ollama_local": "ollama_local",
        }
        return mapping.get(raw, raw or "openai_compatible")

    @staticmethod
    def _extract_models(models_value: Any) -> List[str]:
        models: List[str] = []
        if isinstance(models_value, list):
            for item in models_value:
                if isinstance(item, dict):
                    model_id = str(item.get("id", "")).strip()
                    model_name = str(item.get("name", "")).strip()
                    picked = model_id or model_name
                    if picked:
                        models.append(picked)
                else:
                    text = str(item).strip()
                    if text:
                        models.append(text)
        return models

    def _normalize_settings(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._default_settings()

        providers_source: Dict[str, Any] = {}
        if isinstance(raw_data.get("providers"), dict):
            providers_source = raw_data.get("providers", {})
        elif isinstance(raw_data.get("models"), dict) and isinstance(raw_data["models"].get("providers"), dict):
            providers_source = raw_data["models"].get("providers", {})

        for provider_name, provider_value in providers_source.items():
            if not isinstance(provider_value, dict):
                continue
            name = str(provider_name).strip()
            if not name:
                continue

            api_type = self._normalize_api_type(
                str(provider_value.get("api_type") or provider_value.get("api") or "openai_compatible")
            )
            base_url = str(provider_value.get("base_url") or provider_value.get("baseUrl") or "").strip()
            api_key = str(provider_value.get("api_key") or provider_value.get("apiKey") or "").strip()
            models = self._extract_models(provider_value.get("models", []))

            normalized["providers"][name] = {
                "api_type": api_type,
                "base_url": base_url,
                "api_key": api_key,
                "models": models,
            }

        default_provider = str(raw_data.get("default_provider", "")).strip()
        default_model = str(raw_data.get("default_model", "")).strip()

        if (not default_provider or not default_model) and isinstance(raw_data.get("agents"), dict):
            primary = raw_data["agents"].get("defaults", {}).get("model", {}).get("primary")
            if isinstance(primary, str) and "/" in primary:
                p_name, m_name = primary.split("/", 1)
                if not default_provider:
                    default_provider = p_name.strip()
                if not default_model:
                    default_model = m_name.strip()

        if not default_provider and normalized["providers"]:
            default_provider = next(iter(normalized["providers"].keys()))

        if default_provider and not default_model:
            provider_models = normalized["providers"].get(default_provider, {}).get("models", [])
            default_model = provider_models[0] if provider_models else ""

        normalized["default_provider"] = default_provider
        normalized["default_model"] = default_model
        return normalized

    def import_external_config(self, config_path: str | Path) -> Dict[str, Any]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        normalized = self._normalize_settings(raw_data)
        self.save_settings(normalized)
        return normalized

    def load_settings(self) -> Dict[str, Any]:
        if not self.settings_path.exists():
            data = self._default_settings()
            self.save_settings(data)
            return data

        try:
            with self.settings_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = self._default_settings()
            self.save_settings(data)
            return data

        normalized = self._normalize_settings(data if isinstance(data, dict) else {})
        if normalized != data:
            self.save_settings(normalized)
        return normalized

    def save_settings(self, data: Dict[str, Any]) -> None:
        with self.settings_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_provider_names(self) -> List[str]:
        data = self.load_settings()
        return list(data.get("providers", {}).keys())

    def get_provider(self, provider_name: str) -> Optional[Dict[str, Any]]:
        data = self.load_settings()
        provider = data.get("providers", {}).get(provider_name)
        if not isinstance(provider, dict):
            return None
        return provider

    def add_provider(
        self,
        provider_name: str,
        api_type: str,
        base_url: str,
        api_key: str,
        models: List[str],
    ) -> None:
        name = provider_name.strip()
        if not name:
            raise ValueError("提供商名称不能为空")

        data = self.load_settings()
        providers = data["providers"]
        providers[name] = {
            "api_type": api_type,
            "base_url": base_url.strip(),
            "api_key": api_key.strip(),
            "models": [m for m in (x.strip() for x in models) if m],
        }

        if not data.get("default_provider"):
            data["default_provider"] = name
        if not data.get("default_model") and providers[name]["models"]:
            data["default_model"] = providers[name]["models"][0]

        self.save_settings(data)

    def delete_provider(self, provider_name: str) -> None:
        data = self.load_settings()
        providers = data.get("providers", {})
        if provider_name in providers:
            del providers[provider_name]

        if data.get("default_provider") == provider_name:
            names = list(providers.keys())
            data["default_provider"] = names[0] if names else ""
            if data["default_provider"]:
                first_models = providers[data["default_provider"]].get("models", [])
                data["default_model"] = first_models[0] if first_models else ""
            else:
                data["default_model"] = ""

        self.save_settings(data)

    def update_provider_models(self, provider_name: str, models: List[str]) -> None:
        data = self.load_settings()
        provider = data.get("providers", {}).get(provider_name)
        if not isinstance(provider, dict):
            raise ValueError(f"未找到提供商: {provider_name}")
        provider["models"] = [m for m in (x.strip() for x in models) if m]
        self.save_settings(data)

    def upsert_provider(
        self,
        provider_name: str,
        api_type: str,
        base_url: str,
        api_key: str,
        models: List[str],
        default_model: str,
    ) -> None:
        self.add_provider(provider_name, api_type, base_url, api_key, models)
        data = self.load_settings()
        data["default_provider"] = provider_name
        if default_model.strip():
            data["default_model"] = default_model.strip()
        self.save_settings(data)

    def get_provider_model_pairs(self) -> List[Tuple[str, str]]:
        data = self.load_settings()
        pairs: List[Tuple[str, str]] = []
        providers = data.get("providers", {})
        for provider_name, provider_data in providers.items():
            models = provider_data.get("models", []) if isinstance(provider_data, dict) else []
            for model in models:
                model_name = str(model).strip()
                if model_name:
                    pairs.append((provider_name, model_name))
        return pairs

    async def fetch_models(
        self,
        api_type: str,
        base_url: str,
        api_key: str,
    ) -> Tuple[List[str], str]:
        api_type_text = api_type.strip()
        base = (base_url or "").strip().rstrip("/")
        key = (api_key or "").strip()
        timeout = aiohttp.ClientTimeout(total=12)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if api_type_text in {"openai_official", "openai_compatible", "openai_responses"}:
                    if not base:
                        return [], "Base URL 不能为空"
                    headers = {}
                    if key:
                        headers["Authorization"] = f"Bearer {key}"
                    async with session.get(f"{base}/models", headers=headers) as resp:
                        if resp.status >= 400:
                            txt = (await resp.text())[:200]
                            return [], f"拉取模型失败（HTTP {resp.status}）: {txt}"
                        payload = await resp.json(content_type=None)
                    data = payload.get("data", []) if isinstance(payload, dict) else []
                    models = [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]
                    return [m for m in models if m], ""

                if api_type_text == "ollama_local":
                    if not base:
                        return [], "Base URL 不能为空"
                    ollama_base = re_sub_v1(base)
                    candidates = [f"{ollama_base}/tags", f"{ollama_base}/api/tags"]
                    last_error = ""
                    for url in candidates:
                        try:
                            async with session.get(url) as resp:
                                if resp.status >= 400:
                                    last_error = f"HTTP {resp.status}"
                                    continue
                                payload = await resp.json(content_type=None)
                            data = payload.get("models", []) if isinstance(payload, dict) else []
                            models = [str(item.get("name", "")).strip() for item in data if isinstance(item, dict)]
                            valid = [m for m in models if m]
                            if valid:
                                return valid, ""
                        except Exception as exc:
                            last_error = str(exc)
                    return [], f"拉取 Ollama 模型失败: {last_error or '未返回可用模型'}"

                if api_type_text == "gemini_official":
                    if not key:
                        return [], "Gemini 官方接口需要 API Key"
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                    async with session.get(url) as resp:
                        if resp.status >= 400:
                            txt = (await resp.text())[:200]
                            return [], f"拉取 Gemini 模型失败（HTTP {resp.status}）: {txt}"
                        payload = await resp.json(content_type=None)
                    data = payload.get("models", []) if isinstance(payload, dict) else []
                    models = [str(item.get("name", "")).strip() for item in data if isinstance(item, dict)]
                    return [m for m in models if m], ""

                return [], f"不支持的 api_type: {api_type_text}"

        except Exception as exc:
            return [], f"拉取模型失败: {exc}"


def re_sub_v1(base_url: str) -> str:
    text = base_url.rstrip("/")
    if text.endswith("/v1"):
        return text[:-3]
    return text
