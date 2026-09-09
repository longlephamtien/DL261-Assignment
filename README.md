# DL261-Assignment

Course project website for **Deep Learning and Its Applications (CO3133)**, Semester 261 —
Ho Chi Minh City University of Technology (VNU-HCM), Faculty of Computer Science and Engineering.

Live site: https://longlephamtien.github.io/DL261-Assignment

## Layout

```
web/                    Astro + Tailwind site
  src/data/             all editable content (group, course, assignments)
  src/components/ui/    shadcn/ui primitives
  src/components/       content, layout, and page-specific components
  src/styles/global.css design tokens and the shared type scale
docs/                   course specification
.github/workflows/      GitHub Pages deployment
```

## Editing content

Everything shown on the site lives in `web/src/data/`:

- `site.ts` — institution, course, group, members, course-wide AI disclosure
- `assignments.ts` — per-assignment write-up, resource links, AI disclosure
- `sections.ts` — the fixed section order required by the assignment handbook

Placeholders are written as `TODO — ...` and render as a muted "Pending" state.
A `null` link renders as "Pending" rather than a fabricated URL.

## Local development

```sh
cd web
npm install
npm run dev
```

`npm run build` writes the static site to `web/dist`.

## Deployment

Pushing to `main` runs `.github/workflows/deploy.yml`, which builds `web/` and publishes to
GitHub Pages. Pull requests build without deploying. Repository **Settings → Pages → Source**
must be set to **GitHub Actions**.
