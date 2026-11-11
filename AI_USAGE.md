# AI Usage Disclosure

CO3133 course project, group DeepDive (Group 18).

## Adding an entry

Copy the template into the section for the assignment you are working on and fill in every field.

```
### <what the tool was asked to do>

- **Tool:** 
- **Used by:** 
- **Stage:** 
- **Prompt summary:** 
- **AI contribution:** 
- **Student verification:** 
- **Affected files/sections:** 
- **Responsible member:** 
- **Sources used for verification:** 
```

## Shared

### Choosing a framework for the GitHub Pages landing page

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Course-wide, shared site and tooling
- **Prompt summary:** Asked which static site framework suits a GitHub Pages course site that has to host one landing page and three assignment pages
- **AI contribution:** Compared the candidates and recommended Astro with Tailwind, with content collections for the assignment pages
- **Student verification:** Built the site locally, deployed it to GitHub Pages, and confirmed the whole site is static with no runtime dependency
- **Affected files/sections:** `web/`, `.github/workflows/`
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** Astro documentation on static output and the `base` option; a real GitHub Pages deployment

### Rendering LaTeX formulas in the page UI

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Course-wide, shared site and tooling
- **Prompt summary:** Asked whether a tool exists to convert LaTeX formulas into the page UI, and how the mapping works
- **AI contribution:** Recommended remark-math with rehype-katex so the formulas are rendered at build time, and explained how `$...$` and `$$...$$` map to inline and display math
- **Student verification:** Wrote formulas into the assignment page and inspected the built HTML to confirm KaTeX markup is produced and no JavaScript is shipped for it
- **Affected files/sections:** `web/astro.config.mjs`, `web/src/content/assignments/`
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** KaTeX and remark-math documentation; the generated HTML

### Debugging cross-references from MDX to the rendered page

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Course-wide, shared site and tooling
- **Prompt summary:** Asked why the figure and table labels written in MDX did not resolve to the right numbers and links in the UI
- **AI contribution:** Traced the problem to the directive parsing stage and drafted the remark and rehype plugins that number the floats and resolve each reference to its target
- **Student verification:** Built the site and checked in the generated HTML that every label renders as a numbered float and every reference points at an existing one
- **Affected files/sections:** `web/src/lib/remark-scholarly.ts`, `web/src/lib/rehype-scholarly.ts`
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** remark-directive documentation; the generated HTML for the assignment page

## Assignment 1

### Deciding which aspects of the dataset the EDA should cover

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 1, Milestone 1 draft
- **Prompt summary:** Asked which aspects of Fashion-MNIST the exploratory analysis should cover and whether the current coverage was sufficient
- **AI contribution:** Proposed adding class overlap, pixel correlation against distance, and the row, column, and patch sequence profiles, and pointed out that the handbook also requires input size and imbalance
- **Student verification:** Executed the notebook end to end, checked every figure against its printed output, and corrected two conclusions the measurements did not support
- **Affected files/sections:** `assignments/assignment-1/notebooks/eda.ipynb`; report Dataset and EDA
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** Notebook outputs recomputed from the committed split; the assignment handbook, section 13

### Reviewing the training code for correctness

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 1, Milestone 1 draft
- **Prompt summary:** Asked whether the training code was correct
- **AI contribution:** Found that the run directory was written before the normalization statistics were computed, that metrics were rounded before checkpoint selection, and that model factories silently accepted unknown arguments; drafted the corrections
- **Student verification:** Ran the pipeline from two different working directories, inspected the written run configuration to confirm the statistics are present, and retrained both models to confirm the reported metrics are unchanged
- **Affected files/sections:** `assignments/assignment-1/src/train.py`, `src/data.py`, `src/utils.py`, `src/models/`
- **Responsible member:** Hồ Minh Nhật
- **Sources used for verification:** Real runs of `python -m src.data` and `python -m src.train`; the written run artifacts

### Writing pytest cases for the model contract

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 1, Milestone 1 draft
- **Prompt summary:** Asked for pytest cases covering output shape, parameter count, and rejection of invalid model arguments
- **AI contribution:** Drafted `tests/test_models.py`, 18 cases across both models
- **Student verification:** Ran the suite, then checked a claim the assistant made about flattening being unconditional; it was wrong for unbatched input, so the guard was kept and a regression test added
- **Affected files/sections:** `assignments/assignment-1/tests/test_models.py`, `src/models/`
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** `torch.flatten` behaviour measured directly for rank 1, 2, and 4 tensors

### Proposing an MLP architecture suited to the dataset

- **Tool:** Gemini 3.8 Flash
- **Used by:** Hồ Minh Nhật
- **Stage:** Assignment 1, Milestone 1 draft
- **Prompt summary:** Asked which hidden layer sizes, activation, and regularization suit 28x28 grayscale images with ten balanced classes
- **AI contribution:** Suggested two hidden layers of 512 and 256 units with ReLU and dropout, and explained the trade-off against a single wider layer
- **Student verification:** Trained the proposed architecture and compared its parameter count and validation macro-F1 against the linear baseline before adopting it
- **Affected files/sections:** `assignments/assignment-1/configs/mlp.yaml`, `src/models/mlp.py`; report Methodology
- **Responsible member:** Hồ Minh Nhật
- **Sources used for verification:** The trained run artifacts; the assignment handbook, section 11.1

### Generating a code skeleton for the model module

- **Tool:** Gemini 3.8 Flash
- **Used by:** Hồ Minh Nhật
- **Stage:** Assignment 1, Milestone 1 draft
- **Prompt summary:** Asked for a skeleton of the model module that fits the registry interface, with depth, width, activation, and dropout taken from configuration
- **AI contribution:** Drafted the classifier class and its factory function
- **Student verification:** Completed and adjusted the skeleton by hand, then ran the model tests to confirm the output shape and the parameter count
- **Affected files/sections:** `assignments/assignment-1/src/models/mlp.py`
- **Responsible member:** Hồ Minh Nhật
- **Sources used for verification:** PyTorch documentation for `nn.Sequential` and `nn.Dropout`; the model test suite

### Reviewing whether the source code matches the handbook and is internally consistent

- **Tool:** Gemini 3.8 Flash
- **Used by:** Hồ Minh Nhật
- **Stage:** Assignment 1, Milestone 1 draft
- **Prompt summary:** Asked whether the modules satisfy the mandatory model requirements and whether they agree with each other on the batch and logits contract
- **AI contribution:** Compared the modules against the handbook requirements and pointed out where the interfaces disagreed
- **Student verification:** Re-read the relevant handbook sections and checked each point against the code, then ran the test suite
- **Affected files/sections:** `assignments/assignment-1/src/`
- **Responsible member:** Hồ Minh Nhật
- **Sources used for verification:** The assignment handbook, sections 11.1 and 11.3; the model test suite

### Drafting the experiment plan and fairness protocol

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 1, Milestone 2 final
- **Prompt summary:** Asked for an experiment plan fixing the shared protocol, the seed count, the tuning budget, and a quantitative rule for deciding when one model is better
- **AI contribution:** Drafted `EXPERIMENTS.md`, proposed the non-overlap rule on mean plus or minus one standard deviation across seeds, and pointed out that the handbook requires this content in the report rather than as a repository file
- **Student verification:** Checked every protocol value against `configs/base.yaml` and the committed split, confirmed the per-epoch timings against the recorded runs, and confirmed each experiment maps to an existing backlog issue
- **Affected files/sections:** `assignments/assignment-1/EXPERIMENTS.md`, `README.md`; report Methodology
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The assignment handbook, sections 1.5, 3.2, 12.1 and 12.2; run artifacts under `results/runs/`

## Assignment 2

### Checking the candidate datasets against the handbook thresholds

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 1 proposal
- **Prompt summary:** Asked whether the six proposed datasets met the section 19 thresholds for their tracks, and which to choose
- **AI contribution:** Extracted the thresholds from the handbook, checked each candidate against them, found that NYU Depth V2 falls short of the 5,000-pair floor with 1,449 labelled pairs, and that the Cityscapes link pointed at a resized subset rather than the official release
- **Student verification:** Re-read section 19 for every track, registered a Cityscapes account and confirmed the official packages, and reconciled the counts against each dataset's own documentation
- **Affected files/sections:** `assignments/assignment-2/PROPOSAL.md`; report Problem statement
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The assignment handbook, sections 17, 18 and 19; the Cityscapes and NYU Depth dataset pages

### Designing the controlled experiment and ablation

- **Tool:** Claude Opus 5.5
- **Used by:** Hồ Minh Nhật
- **Stage:** Assignment 2, Milestone 3 final
- **Prompt summary:** Asked for an experiment plan defining the controlled experiment and the ablation, each with hypothesis, changed and fixed factors, metrics and decision criterion, and fixing the seeds and checkpoint rule
- **AI contribution:** Drafted `EXPERIMENTS.md`, confirmed that every varied factor is already a configuration key the trainer reads, derived the per-class hypotheses from the distribution analysis in the proposal, and recomputed the compute budget for three seeds per arm
- **Student verification:** Checked every protocol value against `configs/base.yaml`, the data loader and the committed split, the budget against the proposal's compute table, and that each experiment maps to an existing backlog issue
- **Affected files/sections:** `assignments/assignment-2/EXPERIMENTS.md`, `README.md`; report Methodology
- **Responsible member:** Hồ Minh Nhật
- **Sources used for verification:** The assignment handbook, sections 20 and 21; `assignments/assignment-2/PROPOSAL.md`

### Specifying the fine-tuning procedure

- **Tool:** Claude Opus 5.5
- **Used by:** Hồ Minh Nhật
- **Stage:** Assignment 2, Milestone 3 final
- **Prompt summary:** Asked what issue #70 still required against the existing code, then asked to complete it
- **AI contribution:** Compared the acceptance criteria with the configuration and training code, recommended keeping the single-stage procedure from the proposal over staged unfreezing, moved the hardcoded flip probability into `loader.flip_probability` with a fallback for runs recorded before the key existed, documented the procedure with its configuration keys, and added tests that hold the committed values
- **Student verification:** Ran the Assignment 2 test suite on the branch; the responsible member reviews the diff and the procedure table against `configs/base.yaml` and `PROPOSAL.md` before merging
- **Affected files/sections:** `assignments/assignment-2/configs/base.yaml`, `src/data.py`, `EXPERIMENTS.md`, `README.md`, `tests/test_configs.py`, `tests/test_data.py`
- **Responsible member:** Hồ Minh Nhật
- **Sources used for verification:** The assignment handbook, section 20; `assignments/assignment-2/PROPOSAL.md`, `configs/base.yaml`

### Checking the compute plan for the training runs

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 2 draft
- **Prompt summary:** Asked how many epochs each model could be trained for within the time left before the milestone, and whether the planned budget was realistic
- **AI contribution:** Compared the per-epoch estimates in `results/compute.json` against the rate the running job was actually achieving, found the U-Net running 1.22 times slower than estimated, rescaled the DeepLabV3 estimate by the same factor, and tabulated the finishing time and remaining margin for 10, 20, 30 and 40 epochs
- **Student verification:** Checked the projected finishing time against the epoch timings written into `history.json` as the run progressed, and confirmed the chosen budget left enough margin for evaluation and the remaining deliverables
- **Affected files/sections:** `assignments/assignment-2/results/compute.json`; report Training setup
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The per-epoch timings recorded in each run's `history.json`; `assignments/assignment-2/PROPOSAL.md`

### Reviewing the data loader for leakage

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 2 draft
- **Prompt summary:** Asked whether the data loading path could leak information between partitions, and whether the split and preprocessing hold the guarantees the proposal claims
- **AI contribution:** Traced the three routes by which leakage could enter and reported on each: the split indices are generated once and committed rather than regenerated, so every run reads identical data; normalization statistics are computed on the training split alone and recorded with the run; and the test partition is read once per configuration after model selection concludes. Noted that masks must be resized by nearest neighbour, since interpolating label ids produces classes present in neither neighbour
- **Student verification:** Ran the loaders against the real archives and confirmed 2,406, 569 and 500 images with no unusable annotations, that no city appears in more than one partition, and that validation and test batches arrive uncropped; the labelId to trainId table was checked against `cityscapesscripts.helpers.labels` rather than accepted, since a slip there would corrupt every later metric without failing
- **Affected files/sections:** `assignments/assignment-2/src/data.py`, `tests/test_data.py`; report Data splitting and Preprocessing
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The official Cityscapes toolkit `labels.py`; the committed split file; the dataset on disk

### Rewriting the assignment page in academic English

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 2 draft
- **Prompt summary:** Asked to express the methodology and experimental setup in concise academic English, following the structure the handbook requires
- **AI contribution:** Drafted the Methodology and Experimental setup sections from `EXPERIMENTS.md`, the committed configurations and the implemented code, and flagged where a claim needed a number rather than an adjective
- **Student verification:** Checked every figure in the drafted text against its source, the parameter counts against the models themselves, and the protocol values against `configs/base.yaml`; confirmed the page builds with no unresolved cross-references and that every table and figure is referenced in the text
- **Affected files/sections:** `web/src/content/assignments/assignment-2.mdx`; report Methodology and Training setup
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** `assignments/assignment-2/EXPERIMENTS.md`, `configs/base.yaml`, the model definitions; the assignment handbook, sections 3 and 20

### Implementing the qualitative example selection and the comparison outputs

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 3 final
- **Prompt summary:** Asked for per-image scoring over a whole split, selection of correct, difficult and failure cases spanning the range, and the comparison tables and figures issue #73 requires
- **AI contribution:** Drafted `predict`, `select_examples` and `export_overlays`; lifted the task-agnostic plotting and aggregation out of Assignment 1 into `shared/figures.py` so all three assignments reuse it; added peak-memory measurement covering CUDA, MPS and CPU
- **Student verification:** Found and corrected two faults in the drafted code before adopting it. The selected examples were discarded by a stray `del` before any overlay was written, so the figures could never have been produced; and scoring kept only the first six images of the split, so the reported failure case was the worst of an arbitrary six rather than of the split. Re-ran the evaluation after each correction and confirmed the per-image mIoU range widened from 0.58–0.42 to 0.77–0.29
- **Affected files/sections:** `assignments/assignment-2/src/evaluate.py`, `src/figures.py`, `assignments/shared/figures.py`, `assignments/shared/device.py`; report Quantitative and Qualitative results
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The regenerated evaluation reports and overlays; `tests/test_evaluate.py`

### Building the experiment runner and checking that the arms are comparable

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 3 final
- **Prompt summary:** Asked for a resumable runner for the thirteen queued runs, and whether every arm resolves to the same configuration apart from the factor it varies
- **AI contribution:** Drafted `scripts/experiments.sh`; verified against the resolved configurations that E2, E3 and E4 each differ from the baseline in exactly one key and that the E1 arms differ only in keys under `model`; reported that two earlier draft runs used ten epochs, so their cosine schedule differs from a forty-epoch run at every step and they cannot be pooled
- **Student verification:** Deleted the two draft runs rather than reporting them, smoke-tested every arm before committing the machine for 24 hours, and confirmed from each run's written `config.yaml` that the protocol values match the shared protocol. A claim that the frozen arm would train only the classifier was checked against the model and found wrong: 8,053,011 of 11,024,963 parameters remain trainable, which the report now states
- **Affected files/sections:** `assignments/assignment-2/scripts/experiments.sh`; report Baseline and main methods, Comparison and discussion
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The resolved `config.yaml` written into each run directory; `torch` parameter counts read from the constructed models

### Drafting the report, the slides and the assignment page from the finished runs

- **Tool:** Claude Opus 5
- **Used by:** Lê Phạm Tiến Long
- **Stage:** Assignment 2, Milestone 3 final
- **Prompt summary:** Asked to write the three report parts, the slide deck and the assignment page from the committed runs and evaluation reports, following the structure the handbook requires
- **AI contribution:** Drafted the three report parts, the deck and the page; computed the aggregates across seeds from the run directories; and identified the mechanism behind the Dice ablation result by correlating each class's batch-absence rate with the IoU the loss cost it, giving r = −0.809
- **Student verification:** Checked every figure in the drafted text against the evaluation report it comes from; recomputed the per-class standard deviations rather than accepting two that had been written from memory and were wrong; confirmed both LaTeX documents compile with no unresolved cross-reference, that the page builds with no pending section, and that every table and figure is referenced in the text
- **Affected files/sections:** `assignments/assignment-2/report/`, `presentation/`, `scripts/analysis.py`, `scripts/class_presence.py`, `scripts/pipeline_figure.py`, `web/src/content/assignments/assignment-2.mdx`
- **Responsible member:** Lê Phạm Tiến Long
- **Sources used for verification:** The thirteen run directories and twenty-six evaluation reports under `assignments/assignment-2/results/`; the assignment handbook, sections 3, 20, 21 and 22

## Assignment 3

No entries yet.
