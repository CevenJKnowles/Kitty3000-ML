# Deploy: API on Cloud Run

Live: https://kitty3000-api-1097344477099.europe-west1.run.app (`/docs`, `/health`, `/models`, `POST /predict?model=`)

GCP project `wagon-bootcamp-503009` (Moritz), region `europe-west1`, service `kitty3000-api`, Artifact Registry repo `kitty3000`. Image: CPU-only torch, PANNs CNN14 and AST checkpoints baked in, ~1.6 GB. Idle cost is zero (scales to 0).

## What you need on your machine

1. Docker Desktop, running.
2. `gcloud` logged in with an account that has access to the project (see "Giving someone deploy access" below):

        gcloud auth login
        gcloud config set project wagon-bootcamp-503009

3. The trained classifiers in `models/` (gitignored, they ride into the image, not into git):

        models/panns_cnn14_32000hz_11s_classifier.joblib    # Sebastian's notebook, bas-PANN-model-1
        models/ast_logreg.joblib                            # Chris's notebook, ce-AST-model-1 (once ast.py is merged)

    Get them from the team Drive or from whoever trained them. The filenames must match `MODEL_PATH` in `src/kitty3000_ml/models/panns.py` / `ast.py`. A model whose file is missing simply shows up as not available in `/models` - the API still starts.

    Not needed in `models/`: the PANNs checkpoint `Cnn14_mAP=0.431.pth` and the AST weights - the Dockerfile downloads both at build time.

## Deploy

From the repo root, on the branch you want to deploy:

    bash deploy.sh

First build on a machine: ~5 min (torch + two checkpoints), push ~1.5 GB. Every later deploy: ~1 min build, only the `src` and `models` layers are pushed. The script prints the URL at the end.

Check:

    curl https://kitty3000-api-1097344477099.europe-west1.run.app/models
    f=$(ls data/raw/CatSound_originals/Resting/*.mp3 | head -1)
    curl -F "file=@$f" "https://kitty3000-api-1097344477099.europe-west1.run.app/predict?model=panns"

Expected: `panns` in the model list, `"label": "Resting"` with ~0.96. The first request after a deploy is a cold start (CNN14 loads), up to a minute.

## If the deploy fails

`gcloud run deploy` says "container failed to start" - the previous revision keeps serving, nothing is broken for users. Read the traceback:

    gcloud run services logs read kitty3000-api --region europe-west1 --limit 40

Things that bit us already:

- A model file loads at import without checking it exists -> the whole API fails to start, dummy included. Every model module must guard its loading with `if os.path.exists(MODEL_PATH)`.
- `panns_inference` fetches `class_labels_indices.csv` at import via `wget`, which the slim image doesn't have -> the Dockerfile downloads it at build time. Don't remove that step.
- The package is installed editable (`uv pip install -e .`), so `Path(__file__).parents[3]` is the repo root both locally and in the container (`/app`). Don't switch to a non-editable install.

Local test of the image before pushing (optional, slow on Apple Silicon):

    docker build --platform linux/amd64 -t kitty3000-api .
    docker run -p 8080:8080 kitty3000-api
    curl localhost:8080/health

## Giving someone deploy access

Run once per person, from an account that owns the project:

    PROJECT=wagon-bootcamp-503009
    for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
      gcloud projects add-iam-policy-binding $PROJECT --member="user:<their google account>" --role=$ROLE
    done

They then do step 2 above and `bash deploy.sh`.

## Demo day

Avoid the cold start in front of the audience:

    gcloud run services update kitty3000-api --region europe-west1 --min-instances 1

Costs a few euros per day at 4Gi / 2 CPU. Set it back to `--min-instances 0` afterwards.

## Why the Dockerfile doesn't just `uv sync`

`uv.lock` resolves `torch` to the CUDA build on Linux (plus ~5 GB of `nvidia-*` packages). The API doesn't need a GPU, so the Dockerfile exports the lock, drops torch/torchaudio/torchcodec/nvidia-*/triton, installs the rest, and installs the same torch version from the PyTorch CPU index. `uv sync` or `uv run` inside the image would put CUDA torch back - hence `.venv/bin/uvicorn` in `CMD`.
