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

**`web/src/content/EXAMPLE.mdx` demonstrates every element below.** It sits outside the `assignments/` folder, so it is never built as a page.

### Body syntax

| Element | Syntax |
| --- | --- |
| Inline math | `$\sigma(z)_i = e^{z_i} / \sum_j e^{z_j}$` |
| Display math | `$$ ... $$` on its own lines |
| Code | fenced block with a language tag |
| Plain table | Markdown pipe table |
| Numbered table | `:::table{label="tab:x"}` … `:::` |
| Numbered figure | `:::figure{label="fig:x"}` … `:::` |
| Cross-reference | `:ref[tab:x]` → "Table 1", linked |
| Citation | `:cite[key]` → "[1]", linked |

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