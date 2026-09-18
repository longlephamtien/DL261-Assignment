# shared

Code used by every assignment, installed with the environment described in [assignments](../README.md).

## Checkpoints on the Hugging Face Hub

Weights never go into git. One Hub repository holds all three assignments:

```
<repo_id>/
  assignment-1/runs/<run_id>/{checkpoint.pt, config.yaml, environment.json, history.json, summary.json}
  assignment-2/runs/<run_id>/...
  assignment-3/runs/<run_id>/...
```

A run is uploaded whole rather than as a bare `.pt`, so every reported number traces back to the configuration, split, and environment that produced it, as handbook section 4.2 requires.

## Publish

```sh
python -m src.publish results/runs/<run_id>
```

The assignment folder comes from each assignment's `publish` module; `--repo-id` overrides `HF_REPO_ID` from `.env`.

Cite the returned commit revision in the report and on the assignment page.

## API

| Function | Purpose |
| --- | --- |
| `run_path(assignment, run_id)` | Path of a run inside the repository |
| `sha256(path)`, `checksums(run_dir)` | Checksums published with every upload |
| `upload_run(run_dir, assignment, repo_id=None)` | Upload a run, returns the commit revision |
| `download_checkpoint(assignment, run_id, repo_id=None)` | Fetch one checkpoint |
