import type { Assignment, Section, SectionMeta } from "./types";

export const SECTION_ORDER: readonly SectionMeta[] = [
  { id: "problem-statement", title: "Problem statement" },
  { id: "dataset-and-eda", title: "Dataset and EDA" },
  { id: "methodology", title: "Methodology" },
  { id: "experimental-setup", title: "Experimental setup" },
  { id: "results", title: "Results" },
  { id: "comparison-and-discussion", title: "Comparison and discussion" },
  { id: "error-analysis", title: "Error analysis" },
  { id: "limitations-and-conclusion", title: "Limitations and conclusion" },
];

export function resolveSections(assignment: Assignment): Section[] {
  return SECTION_ORDER.map((meta) => ({
    ...meta,
    blocks: assignment.content[meta.id] ?? [],
  }));
}
