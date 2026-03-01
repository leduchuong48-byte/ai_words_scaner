FROM python:3.11-slim

ARG APP_NAME="Ai Words Scaner"
ARG APP_VERSION="1.1"

LABEL org.opencontainers.image.title="${APP_NAME}" \
      org.opencontainers.image.version="${APP_VERSION}"

ENV APP_NAME="${APP_NAME}" \
    APP_VERSION="${APP_VERSION}"

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-droid-fallback \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r /app/requirements.txt

# 离线安装 spaCy 模型（如果本地存在模型文件）
COPY en_core_web_sm-*.whl /tmp/models/
RUN if ls /tmp/models/en_core_web_sm-*.whl 1>/dev/null 2>&1; then \
        pip install --no-cache-dir /tmp/models/en_core_web_sm-*.whl && \
        rm -rf /tmp/models; \
    else \
        echo "[Info] 未找到本地 spaCy 模型文件，将使用轻量回退模式"; \
    fi

COPY . /app

RUN mkdir -p /app/dicts && chmod -R a+r /app/dicts

EXPOSE 7860

CMD ["python", "app.py"]
