import type { AiDisclosure, Course, Group, Institution } from "./types";

export const institution: Institution = {
  university: "Ho Chi Minh City University of Technology (VNU-HCM)",
  faculty: "Faculty of Computer Science and Engineering",
};

export const course: Course = {
  name: "Deep Learning and Its Applications",
  code: "CO3133",
  semester: "Semester 261",
  instructor: "Lê Thành Sách",
};

export const group: Group = {
  id: "18",
  name: "DeepDive",
  repository: "https://github.com/longlephamtien/DL261-Assignment",
  members: [
    {
      name: "Lê Phạm Tiến Long",
      studentId: "2352688",
      role: "Data analysis and delivery: EDA, dataset proposals, repository, website, reports, and submissions",
      github: "https://github.com/longlephamtien",
    },
    {
      name: "Ngô Tiểu Nghi",
      studentId: "2352799",
      role: "Data and training pipeline: loaders, preprocessing, splits, training, fine-tuning, and experiment runs",
      github: "http://github.com/nghingo169",
    },
    {
      name: "Hồ Minh Nhật",
      studentId: "2352858",
      role: "Models and evaluation: architectures, fusion, metrics, ablations, and error analysis",
      github: "https://github.com/henries05",
    },
  ],
};

export const courseAiDisclosure: AiDisclosure = {
  noAiUsed: false,
  summary:
    "This only disclouses shared site. Assignment-specific use is disclosed on each assignment page.",
  entries: [
    {
      tool: "Claude Opus 5",
      usedBy: "Lê Phạm Tiến Long",
      task: "Choosing a framework for the GitHub Pages landing page",
      promptSummary:
        "Asked which static site framework suits a GitHub Pages course site hosting one landing page and three assignment pages.",
      promptLog: null,
      aiContribution:
        "Compared the candidates and recommended Astro with Tailwind, with content collections for the assignment pages.",
      studentVerification:
        "Built the site locally, deployed it to GitHub Pages, and confirmed the output is fully static.",
      affectedSections: ["Site framework", "Deployment workflow"],
      responsibleMember: "Lê Phạm Tiến Long",
    },
    {
      tool: "Claude Opus 5",
      usedBy: "Lê Phạm Tiến Long",
      task: "Rendering LaTeX formulas in the page UI",
      promptSummary:
        "Asked whether a tool exists to convert LaTeX formulas into the page UI, and how the mapping works.",
      promptLog: null,
      aiContribution:
        "Recommended remark-math with rehype-katex so formulas render at build time, and explained how inline and display delimiters map to the output.",
      studentVerification:
        "Wrote formulas into an assignment page and inspected the built HTML to confirm KaTeX markup is produced and no JavaScript is shipped for it.",
      affectedSections: ["Markdown pipeline", "Assignment pages"],
      responsibleMember: "Lê Phạm Tiến Long",
    },
    {
      tool: "Claude Opus 5",
      usedBy: "Lê Phạm Tiến Long",
      task: "Debugging cross-references from MDX to the rendered page",
      promptSummary:
        "Asked why figure and table labels written in MDX did not resolve to the right numbers and links in the UI.",
      promptLog: null,
      aiContribution:
        "Traced the problem to the directive parsing stage and drafted the plugins that number the floats and resolve each reference to its target.",
      studentVerification:
        "Built the site and checked in the generated HTML that every label renders as a numbered float and every reference points at an existing one.",
      affectedSections: ["Markdown pipeline", "Assignment pages"],
      responsibleMember: "Lê Phạm Tiến Long",
    },
  ],
};

export const site = {
  name: "DL261-Assignment",
  title: `${course.code}: ${course.name}`,
  description: `Course project website for ${course.name} (${course.code}), ${course.semester}, ${institution.university}.`,
};
