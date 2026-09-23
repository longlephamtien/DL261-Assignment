# shared

Code used by every assignment, installed with the environment described in [assignments](../README.md).

## Publish

Fill `.env` first, then upload a finished run from its assignment folder:

```sh
cd assignment-1
python -m src.publish results/runs/<run_id>
python -m src.publish results/runs/<run_id> --message "Selected MLP baseline for the M1 draft"
```


## Reconstruct

```python
from shared import hub
path = hub.download_checkpoint("assignment-1", "<run_id>")
```

## API

| Function | Purpose |
| --- | --- |
| `run_path(assignment, run_id)` | Path of a run inside the repository |
| `sha256(path)`, `checksums(run_dir)` | Checksums published with every upload |
| `upload_run(run_dir, assignment, repo_id=None, private=False, message=None)` | Upload a run, returns the commit revision |
| `download_checkpoint(assignment, run_id, repo_id=None, revision='main')` | Fetch one checkpoint |
