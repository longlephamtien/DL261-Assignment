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

## Conventions

- One branch per issue, named `<type>/<short-topic>`.
- Commit subjects use the same types as issue titles: `feat`, `fix`, `chore`, `refactor`.
- Reference the issue from the pull request, and close it from the commit that finishes the work with `Closes #N`.
- A second member reviews changes to shared pipeline code before merging to `main`.
- Freeze each milestone submission with an annotated tag: `a1-m1-draft`, `a1-final`, and so on.
- Run `npm run check` in `web/` before opening a pull request.