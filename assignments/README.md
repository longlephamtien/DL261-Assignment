# Assignments

Python code for the three assignments.

## Setup

We use Python 3.12.

```sh
# macOS / Linux
cd assignments
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip check
```

```powershell
# Windows (PowerShell)
cd assignments
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip check
```

## Credentials

Copy `.env.example` at the repository root to `.env` (`cp` on macOS/Linux, `copy` on Windows) and fill in the Hugging Face repository id and access token.

## Layout

```
pyproject.toml   pinned dependencies and the shared package
shared/          code used by every assignment
assignment-1/    Fashion-MNIST classifiers
```

Each assignment README covers its own workflow; [shared](shared/README.md) covers publishing checkpoints to the Hugging Face Hub.
