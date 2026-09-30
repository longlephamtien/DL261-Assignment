# shared

Code used by every assignment, installed with the environment described in [assignments](../README.md).

## Publish

Fill `.env` first, then upload a finished run from its assignment folder. Evaluate the run first; publishing refuses a run with no report, because the reports are not in git and the Hub commit is what a reported number is traced to:

```sh
cd assignment-1
python -m src.publish <run_id>
python -m src.publish <run_id> --message "Selected MLP baseline for the M1 draft"
```


## Reconstruct

```python
from shared import hub
path = hub.download_checkpoint("assignment-1", "<run_id>")
report = hub.download_file("assignment-1", "<run_id>", "evaluation/val/report.json")
```

## API

| Function | Purpose |
| --- | --- |
| `run_path(assignment, run_id)` | Path of a run inside the repository |
| `artifacts(run_dir, extras=None)` | Files an upload covers, keyed by their path on the Hub |
| `sha256(path)`, `checksums(files)` | Checksums published with every upload |
| `upload_run(run_dir, assignment, repo_id=None, private=False, message=None, extras=None)` | Upload a run in one commit, returns the revision |
| `download_file(assignment, run_id, name='checkpoint.pt', repo_id=None, revision='main')` | Fetch one file of a run |
| `download_checkpoint(assignment, run_id, repo_id=None, revision='main')` | Fetch one checkpoint |
