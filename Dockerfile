FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_core.py bot_telegram.py update_index.py ./

RUN mkdir -p /app/knowledge-base /app/index_out /app/logs

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3'); print('Embedding model cached OK')"

ENV RAG_TOP_K=5 \
    RAG_MAX_HISTORY_TURNS=5

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD python -c "import faiss, sentence_transformers; print('ok')" || exit 1

CMD ["python", "bot_telegram.py", "--index_dir", "/app/index_out"]
