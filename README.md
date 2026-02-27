# AI 生词检索器（ai_words_scaner）

[![Docker Pulls](https://img.shields.io/docker/pulls/leduchuong/ai_words_scaner?logo=docker&style=flat-square)](https://hub.docker.com/r/leduchuong/ai_words_scaner)
[![Docker Stars](https://img.shields.io/docker/stars/leduchuong/ai_words_scaner?logo=docker&style=flat-square)](https://hub.docker.com/r/leduchuong/ai_words_scaner)
[![GitHub Stars](https://img.shields.io/github/stars/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/issues)
[![License](https://img.shields.io/github/license/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/blob/main/LICENSE)

[English](README_en.md)

面向英文阅读场景的 AI 生词流水线：上传 `PDF/EPUB/TXT`，自动提取候选词，结合 LLM 生成语境释义与固定搭配，并导出可直接用于背词与复盘的结果文件。

## 为什么做它

传统“查词 + 记词”流程最大的痛点是碎片化：提词、去噪、释义、搭配、导出都在不同工具里来回切换。`ai_words_scaner` 把这条链路收敛到一个 WebUI 里，目标是让你从“读到不认识”到“得到可复习词表”只走一条可重复的流程。

## 核心卖点

- 一次上传，完整处理：支持 `PDF / EPUB / TXT`。
- 双策略过滤：
  - 专业阅读模式（黑名单）用于剔除常见词；
  - 雅思备考模式（白名单）用于对齐考试词表与等级。
- 可配置多模型：OpenAI 官方、OpenAI 兼容、OpenAI Responses、Gemini、Ollama 本地。
- 稳定性优先：断点续跑、失败重试、并发池控制与熔断保护。
- 结果可直接用：导出 `Excel / PDF / TXT 单词本`，并生成 Typing-World 词表。

## For Portainer/Synology Users

Copy this into Portainer stacks and hit Deploy. Done.

## Docker Compose

```yaml
services:
  ai_words_scaner:
    image: leduchuong/ai_words_scaner:latest
    container_name: ai_words_scaner
    restart: unless-stopped
    ports:
      - "1016:7860"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command: ["python", "app.py"]
```

## 快速开始

### Docker 一条命令

```bash
docker run --rm -it \
  -p 1016:7860 \
  --add-host host.docker.internal:host-gateway \
  leduchuong/ai_words_scaner:latest
```

启动后访问：`http://<你的主机IP>:1016`

### 源码运行

```bash
pip install -r requirements.txt
python3 app.py
```

## 典型使用流程

1. 进入“模型配置中心”，添加 Provider、拉取模型并设置默认模型。
2. 回到“任务看板”，上传电子书文件（PDF/EPUB/TXT）。
3. 选择过滤策略（专业阅读 / 雅思白名单）与导出格式。
4. 点击“启动任务”，等待处理完成后下载结果。

## 模型与配置能力

- 支持 Provider 的新增、编辑、删除与默认模型切换。
- 支持在线拉取模型列表（OpenAI 兼容接口 / Gemini / Ollama）。
- `llm_settings.json` 已按可公开模板组织，默认不包含真实密钥。

## 稳定性设计（实用向）

- 断点续跑：使用 `cache_results.jsonl` 记录进度，重复任务可命中已处理结果。
- 并发执行：异步并发池（Semaphore）提升吞吐。
- 故障恢复：对临时错误重试，对连续严重错误熔断终止，避免无效刷接口。

## 输出结果

- `vocabulary_reading.xlsx`：结构化词表（含词性、语境释义、搭配等）。
- `vocabulary_reading.pdf`：便于打印或离线阅读。
- `maimemo_vocabulary.txt`：可用于单词本导入。
- `typing_world.csv`：Typing-World 练习词表。

## Profile Style Metrics

<p align="left"><img src="https://komarev.com/ghpvc/?username=leduchuong48-byte&label=Repo%20views&color=0e75b6&style=flat" alt="views" /></p>

<p>
  <img align="left" src="https://github-readme-stats-sigma-five.vercel.app/api/top-langs?username=leduchuong48-byte&show_icons=true&locale=en&layout=compact" alt="top-langs" />
  <img align="center" src="https://github-readme-stats-sigma-five.vercel.app/api?username=leduchuong48-byte&show_icons=true&locale=en" alt="stats" />
</p>

<p><img align="center" src="https://github-readme-streak-stats.herokuapp.com/?user=leduchuong48-byte" alt="streak" /></p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=leduchuong48-byte/ai_words_scaner&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=leduchuong48-byte/ai_words_scaner&type=Date" />
  <img alt="Star History" src="https://api.star-history.com/svg?repos=leduchuong48-byte/ai_words_scaner&type=Date" />
</picture>

## License

MIT，详见 [LICENSE](LICENSE)。

## Disclaimer

使用本项目即表示你已阅读并同意 [免责声明](DISCLAIMER.md)。
