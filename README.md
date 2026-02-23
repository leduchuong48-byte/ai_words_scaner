# AI 生词检索器（words_scaner）

[English](README_en.md)

一个基于 `Gradio + spaCy + LLM` 的英文阅读词汇提取工具。

- 支持从 `PDF / EPUB / TXT` 中提取候选词
- 支持两种过滤策略：专业阅读模式（黑名单）与雅思备考模式（白名单）
- 支持调用兼容 OpenAI / Ollama / Gemini 的模型做语境释义与搭配补全
- 支持导出 `Excel / PDF / TXT`（按你在界面中选择）

## 1. 运行环境

- Python `3.11+`
- 推荐系统：Linux / macOS / WSL
- 可选：Docker 与 Docker Compose

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 配置模型（首次必做）

编辑 `llm_settings.json`：

- 填写你自己的 `base_url`
- 填写你自己的 `api_key`
- 选择默认 `provider` 与 `model`

说明：仓库中的配置文件已做脱敏处理，不包含任何真实密钥。

## 3. 启动方式

### 方式 A：本地 Python

```bash
python3 app.py
```

默认监听：`0.0.0.0:7860`

### 方式 B：Docker Compose

```bash
docker compose up --build
```

默认映射端口：`1016 -> 7860`

## 4. 使用流程

1. 在“模型配置中心”测试并保存可用模型
2. 在“任务看板”上传 `PDF / EPUB / TXT`
3. 选择过滤策略与导出格式
4. 启动任务并下载结果文件

## 5. 输入与输出

### 输入

- 电子书文件：`PDF / EPUB / TXT`
- 可选自定义词库：
  - `word,level` 两列时按白名单处理
  - 文件名含 `black` 时按黑名单处理

### 产出（运行后生成）

- `cache_results.jsonl`（断点与缓存）
- `run_trace.log`（运行日志）
- `typing_world.csv`
- `maimemo_vocabulary.txt`
- `vocabulary_reading.xlsx`
- `vocabulary_reading.pdf`

以上文件属于运行产物，默认已在 `.gitignore` 中排除。

## 6. 隐私与安全

- 不要提交真实 `API Key`
- 不要提交包含个人或受限内容的输入文档与导出结果
- 建议把本地敏感配置写入 `llm_settings.local.json` 或 `.env`（并保持忽略）

## 7. 目录结构

```text
.
├── app.py
├── extractor_core.py
├── llm_processor.py
├── config_manager.py
├── dicts/
├── llm_settings.json
├── llm_settings.example.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## 8. 仓库维护建议

- 提交前执行一次敏感信息扫描（密钥、令牌、私有 URL、个人数据）
- 单文件超过 GitHub 限制时，改用 Git LFS 或外部对象存储
- 持续保持 `README.md` 与实际功能一致

## 9. 免责声明

使用本项目即表示你已阅读并同意 [免责声明](DISCLAIMER.md)。
