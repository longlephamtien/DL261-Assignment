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

## Assignment 2

No entries yet.

## Assignment 3

No entries yet.
