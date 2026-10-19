# DL261-Assignment

Course project website for **Deep Learning and Its Applications (CO3133)**, Semester 261 - Ho Chi Minh City University of Technology (VNU-HCM), Faculty of Computer Science and Engineering.

Live site: https://longlephamtien.github.io/DL261-Assignment

## Layout

```
web/                         Astro + Tailwind site
  src/content/assignments/   one MDX file per assignment
  src/data/site.ts           institution, course, group, members
  src/components/            UI; ui/ holds the shadcn primitives
  src/lib/                   markdown plugins and helpers
  src/styles/global.css      colour tokens, type scale, prose styles
  public/assignments/<slug>/ figures
assignments/                 one Python environment for all assignments
  shared/                    code shared by every assignment
  assignment-1/              Fashion-MNIST pipeline, models, and experiments
docs/                        course specification
AI_USAGE.md                  AI disclosure log
.github/workflows/           GitHub Pages deployment
```

## Run it

```sh
cd web
npm install
npm run dev        # http://localhost:4321/DL261-Assignment
npm run build      # static output in web/dist
npm run check      # types, links, content schema
```

The assignment code is separate: [assignments/README.md](assignments/README.md) covers the environment and credentials, each assignment README covers its own workflow.

## Edit the landing page

Everything on it comes from `web/src/data/site.ts` - institution, course, group, members. Use `null` for anything not decided yet; it renders as "Pending" instead of a made-up value.

## Edit an assignment

One file per assignment: `web/src/content/assignments/assignment-N.mdx`.

Frontmatter holds the metadata, the body holds the write-up. Keep the eight `##` headings - the handbook requires them, and the page navigation is generated from them. An empty section renders a "Not written yet" badge.

### Body syntax

| Element | Syntax |
| --- | --- |
| Inline math | `$\sigma(z)_i = e^{z_i} / \sum_j e^{z_j}$` |
| Display math | `$$ ... $$` on its own lines |
| Code | fenced block with a language tag |
| Plain table | Markdown pipe table |
| Numbered table | `:::table{label="tab:x"}` ... `:::` |
| Numbered figure | `:::figure{label="fig:x"}` ... `:::` |
| Cross-reference | `:ref[tab:x]`, renders as "Table 1", linked |
| Citation | `:cite[key]`, renders as "[1]", linked |

Inside a `:::table` or `:::figure` block, the **last paragraph is the caption**. Numbering is automatic and independent per kind, so inserting a figure renumbers the ones after it. Captions sit above tables and below figures.

### Bibliography

List sources in frontmatter; the citation number is the position in this list. The References section and its navigation entry appear only when the list is non-empty.

```yaml
references:
  - id: he2016resnet
    text: "K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016."
    href: https://arxiv.org/abs/1512.03385
```

An unknown `:cite[key]` renders `[?]` and warns during the build rather than failing silently.

### Figures

Put files in `web/public/assignments/<slug>/` and reference them with the site prefix:

```
![](/DL261-Assignment/assignments/assignment-1/<example>.svg)
```

Export plots from matplotlib as PNG or SVG. Math is rendered at build time by KaTeX, so no JavaScript is shipped for it.

### AI disclosure

```yaml
aiDisclosure:
  noAiUsed: false
  summary: One or two sentences covering AI use in this assignment.
  entries:
    - tool: Name and version or model
      usedBy: Member name
      task: What it was used for
      promptSummary: Representative prompt
      promptLog: null
      aiContribution: What the tool produced or suggested
      studentVerification: How the output was edited and checked
      affectedSections: [Affected file, report section]
      responsibleMember: Member responsible for final verification
```

If no AI tool was used, set `noAiUsed: true` and leave `entries` empty. The build rejects `noAiUsed: true` together with listed entries.

## Report and presentation

Each assignment carries two LaTeX documents, `assignments/assignment-N/report/` and `.../presentation/`. Write content only in `contents/*.tex`; the main file holds the preamble and the title page, and its section order follows handbook section 3 for the report and section 2.3 for the slides. Every `\subsection` opens with a `% Required:` comment listing what the handbook expects there.

### Floats

Never leave a float unreferenced. A `\label` with no `\ref` pointing at it means the reader meets the float with no idea why it is there, and the handbook grades on evidence tied to claims. Pair them every time:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{images/class_distribution.png}
\caption{Share of evaluated pixels per class.}
\label{fig:class-distribution}
\end{figure}

Three classes hold 75.8\% of all pixels, as Figure~\ref{fig:class-distribution} shows.
```

Use `[H]` so the float stays where it is written, and `~\ref` so the number never wraps to a new line. Captions go **below** figures and **above** tables.

### Figures

| Source | How to import |
| --- | --- |
| PNG, JPEG | `\includegraphics[width=0.8\textwidth]{images/<name>.png}` |
| PDF | `\includegraphics[width=0.8\textwidth]{images/<name>.pdf}` |
| TikZ or PGF | `\resizebox{0.8\textwidth}{!}{\input{images/<name>.tex}}` |

Scale with `width=<fraction>\textwidth`, never with absolute lengths, so a figure still fits when the layout changes. Vector sources beat raster: use PDF or PGF for plots and diagrams, PNG only for photographs and screenshots. `matplotlib` writes all three through `shared.figures.save`.

`\resizebox` is for `\input`-ed TikZ and PGF only, because those carry their own dimensions. Applying it to `\includegraphics` distorts the aspect ratio.

### Tables

```latex
\begin{table}[H]
\centering
\caption{Validation results across three seeds.}
\label{tab:results}
\begin{tabular}{lrr}
\hline
\textbf{Model} & \textbf{mIoU} & \textbf{Parameters} \\
\hline
U-Net        & 0.6123 & 7.8M  \\
DeepLabV3    & 0.7045 & 11.0M \\
\hline
\end{tabular}
\end{table}
```

Align text left and numbers right, `l` against `r`. For a column that must wrap, use `tabularx` with an `X` column and give the environment a width: `\begin{tabularx}{\textwidth}{|l|l|X|c|}`. Report every number at the precision it was measured at; do not round before averaging across seeds.

### Listings

The preamble loads `listings` with named styles, and `minted` where a build has Pygments. Prefer `listings` since it needs no external tool and no `--shell-escape`:

```latex
\begin{lstlisting}[language=Python, caption={Model registry.}, label={lst:registry}]
@register("unet")
def build_unet(**kwargs) -> nn.Module: ...
\end{lstlisting}

Models are built from configuration, as Listing~\ref{lst:registry} shows.
```

Keep listings to the few lines that carry the argument. A long listing belongs in the repository, which the report links from Code Availability.

### Bullet points

```latex
\begin{itemize}
    \item \textbf{Class imbalance:} road, building and vegetation hold 75.8\% of pixels.
    \item \textbf{Two kinds of scarcity:} thin structures lost to boundary error, and
          large objects that are rarely seen.
\end{itemize}
```

Bold the lead phrase so a reader can scan. Nest at most two levels. Lists suit slides; in the report prefer prose, because the handbook rejects claims without evidence and a bullet rarely carries its own.

## Conventions

- One branch per issue, named `<type>/<short-topic>`.
- Commit subjects use the same types as issue titles: `feat`, `fix`, `chore`, `refactor`.
- Reference the issue from the pull request, and close it from the commit that finishes the work with `Closes #N`.
- A second member reviews changes to shared pipeline code before merging to `main`.
- Freeze each milestone submission with an annotated tag: `a1-m1-draft`, `a1-final`, and so on.
- Run `npm run check` in `web/` before opening a pull request.