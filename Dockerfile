FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/hf_cache

COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
    && grep -viE '^(torch|torchaudio|torchcodec|triton)==|^nvidia-' requirements.txt > requirements-cpu.txt \
    && uv venv \
    && uv pip install -r requirements-cpu.txt \
    && uv pip install --torch-backend cpu torch==2.14.0 torchaudio==2.11.0 torchcodec==0.16.0

RUN mkdir -p /root/panns_data \
    && .venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1', '/root/panns_data/Cnn14_mAP=0.431.pth')"

RUN .venv/bin/python -c "from transformers import ASTFeatureExtractor, ASTModel; ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593'); ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY src src
RUN uv pip install --no-deps .

COPY models models

CMD .venv/bin/uvicorn kitty3000_ml.api:app --host 0.0.0.0 --port ${PORT:-8080}
