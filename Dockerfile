FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

COPY --from=ghcr.io/astral-sh/uv:0.12@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/hf_cache

RUN mkdir -p /root/panns_data \
    && python3 -c "import urllib.request; urllib.request.urlretrieve('https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1', '/root/panns_data/Cnn14_mAP=0.431.pth')" \
    && python3 -c "import urllib.request; urllib.request.urlretrieve('http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv', '/root/panns_data/class_labels_indices.csv')"

RUN uv venv /tmp/hf \
    && uv pip install --python /tmp/hf/bin/python huggingface_hub==1.30.0 \
    && /tmp/hf/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('MIT/ast-finetuned-audioset-10-10-0.4593', revision='f826b80d28226b62986cc218e5cec390b1096902', allow_patterns=['*.json', '*.safetensors'])" \
    && mkdir -p /app/hf_cache/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/refs \
    && echo -n f826b80d28226b62986cc218e5cec390b1096902 > /app/hf_cache/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/refs/main \
    && rm -rf /tmp/hf

COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
    && grep -viE '^(torch|torchaudio|torchcodec|triton)==|^nvidia-' requirements.txt > requirements-cpu.txt \
    && uv venv \
    && uv pip install -r requirements-cpu.txt \
    && uv pip install --torch-backend cpu torch==2.14.0 torchaudio==2.11.0 torchcodec==0.16.0

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN .venv/bin/python -c "import panns_inference; from transformers import ASTFeatureExtractor, ASTModel; ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593'); ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')"

COPY src src
RUN uv pip install --no-deps -e .

COPY models models

CMD .venv/bin/uvicorn kitty3000_ml.api:app --host 0.0.0.0 --port ${PORT:-8080}
