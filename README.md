# Loupe: GenAI Assignment 1

Four image models, served through one React + FastAPI web app in Docker:

1. **Universal Restoration**: one convolutional autoencoder for salt-and-pepper noise, Gaussian blur and occlusion (Oxford-IIIT Pet).
2. **Hard-Routed Restoration**: a CNN classifier names the corruption, and one specialist autoencoder restores it.
3. **Soft Mixture-of-Experts Restoration**: a gate blends an identity branch and the three specialists.
4. **Face-to-Sketch Generator**: a style-conditioned pix2pix cGAN trained on FS2K (three sketch styles).

## Run the app

You need Docker with Compose. From the repo root:

```bash
./get_models.sh && docker compose up --build
```

`get_models.sh` downloads the seven trained ONNX models (214 MB) from the [`models-v1` release](https://github.com/Er0sama/GenAI-Assignment1/releases/tag/models-v1) into `models/` and checks their SHA-256 sums. On Windows, run it in Git Bash or WSL, or download the `.onnx` files from the release page into `models/` by hand.

- App: http://localhost:8080
- MLflow (training records, read-only): http://localhost:5001

The first build needs internet to install packages. After that the app runs offline. Stop it with `Ctrl+C`.

## Repository layout

| Folder | Contents |
|---|---|
| `common/` | Pet dataset loading, 80/20 split, runtime corruptions, fixed val/test manifests, losses and metrics |
| `task1_universal_autoencoder/` | Model, training, Optuna search, test evaluation, ONNX export |
| `task2_hard_routing/` | Classifier, specialists, router, their Optuna searches, evaluation, ONNX export |
| `task3_soft_moe/` | Soft MoE (initialised from Task 2), training, Optuna search, evaluation, ONNX export |
| `task4_cgan_sketch/` | FS2K data pipeline, generator and discriminator, training, Optuna search, evaluation, ONNX export |
| `app/backend/` | FastAPI service running the ONNX models (`selftest.py` checks it against the training code) |
| `app/frontend/` | React + Tailwind frontend |
| `manifests/` | Fixed splits and the val/test corruption parameters, so every result can be reproduced exactly |
| `optuna/` | The five Optuna studies (SQLite) |
| `mlflow.db` | MLflow tracking database for every run (artifacts not included) |
| `models/` | ONNX models (downloaded by `get_models.sh`) |

## Reproduce the training

Tested with Python 3.12, PyTorch 2.14 (CUDA 13.0) on an RTX 5060 Laptop GPU (8 GB).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # pinned versions; PyTorch comes from its CUDA 13.0 index
```

**Data** goes in `data/` (not in the repo):
- Oxford-IIIT Pet downloads itself on first use.
- FS2K: download it from the [official repository](https://github.com/DengPingFan/FS2K) and unpack it to `data/FS2K/` (with `photo/`, `sketch/`, `anno_train.json`, `anno_test.json`).

```bash
python -m common.make_manifests        # only needed if manifests/ is missing; the committed ones are what the results use
python -m common.test_common           # checks the corruption pipeline
python -m task4_cgan_sketch.data       # FS2K split and caches
```

Each task follows the same order: Optuna search, final training with the best settings, a single test evaluation, then ONNX export. The final settings below are the best trials stored in `optuna/`.

```bash
# Task 1
python -m task1_universal_autoencoder.tune --trials 40
python -m task1_universal_autoencoder.train --epochs 60 --name final --cfg '{"lr": 0.0008783164913704277, "batch_size": 16, "latent_ch": 32, "base_ch": 64, "dropout": 0.04408982445617521, "alpha": 0.5415273983442785}'
python -m task1_universal_autoencoder.evaluate
python -m task1_universal_autoencoder.export_onnx

# Task 2
python -m task2_hard_routing.tune_classifier --trials 40
python -m task2_hard_routing.train_classifier --epochs 30 --name final --cfg '{"lr": 0.0022111083220645446, "batch_size": 16, "channels": "32-64-128-128-256", "dropout": 0.4374703991026736, "weight_decay": 3.875226456412055e-05}'
python -m task2_hard_routing.tune_specialists --trials 45
C='"lr": 0.0008853007303819069, "batch_size": 16, "latent_ch": 32, "base_ch": 64, "alpha": 0.49679882325945124'
python -m task2_hard_routing.train_specialist --type salt      --epochs 60 --name final_salt      --cfg "{$C, \"seed\": 1}"
python -m task2_hard_routing.train_specialist --type blur      --epochs 60 --name final_blur      --cfg "{$C, \"seed\": 2}"
python -m task2_hard_routing.train_specialist --type occlusion --epochs 60 --name final_occlusion --cfg "{$C, \"seed\": 3}"
python -m task2_hard_routing.evaluate
python -m task2_hard_routing.export_onnx

# Task 3 (needs Task 2's final checkpoints)
python -m task3_soft_moe.tune --trials 40
python -m task3_soft_moe.train --epochs 16 --name final --cfg '{"lr_joint": 4.347788604191704e-05, "tau": 1.1905534085753122, "lam_c": 0.0019692299730100006, "lam_b": 0.0006961085663064285, "alpha": 0.7718958710980289, "lr_gate": 0.0009204331007007203}'
python -m task3_soft_moe.evaluate
python -m task3_soft_moe.export_onnx

# Task 4
python -m task4_cgan_sketch.tune --trials 43
python -m task4_cgan_sketch.train --epochs 200 --name final --cfg '{"lr_G": 0.000252975786282591, "lr_D": 6.56628826226686e-05, "batch_size": 1, "base_ch": 48, "dropout": 0.3148348189782642, "emb_dim": 8, "lam_l1": 92.7541414559482}'
python -m task4_cgan_sketch.evaluate
python -m task4_cgan_sketch.export_onnx
```

Outputs (checkpoints, figures, ONNX) go to each task's `outputs/` folder. Copy the ONNX files into `models/` to serve them in the app.

**Browse the records:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
optuna-dashboard sqlite:///optuna/task1.db      # also task2_classifier, task2_specialists, task3, task4
```

## Backend self-test

```bash
MODELS_DIR=models python app/backend/selftest.py
```
Checks the backend's numpy corruptions and face preprocessing against the training code, and calls every endpoint.
