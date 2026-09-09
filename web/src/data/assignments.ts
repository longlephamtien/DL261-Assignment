import type { AiDisclosure, Assignment, AssignmentContent, ResourceLink } from "./types";

function emptyContent(): AssignmentContent {
  return {
    "problem-statement": [],
    "dataset-and-eda": [],
    "methodology": [],
    "experimental-setup": [],
    "results": [],
    "comparison-and-discussion": [],
    "error-analysis": [],
    "limitations-and-conclusion": [],
  };
}

function emptyResources(): ResourceLink[] {
  return [
    { label: "Source code", href: null, description: "Training and evaluation code." },
    { label: "Checkpoints", href: null, description: "Weights, or steps to reproduce them." },
    { label: "Report and slides", href: null, description: "Written report and presentation deck." },
    { label: "YouTube video", href: null, description: "Public or unlisted presentation recording." },
  ];
}

function emptyDisclosure(): AiDisclosure {
  return { summary: null, tools: [], usedFor: [], verification: null };
}

export const assignments: Assignment[] = [
  {
    slug: "assignment-1",
    number: 1,
    title: "Foundations of Deep Learning Pipelines",
    topic: "From linear models to modern sequence models for image classification",
    status: "planned",
    abstract:
      "Linear baselines, convolutional networks, and sequence models compared on one benchmark with data and evaluation held fixed.",
    content: emptyContent(),
    resources: emptyResources(),
    aiDisclosure: emptyDisclosure(),
  },
  {
    slug: "assignment-2",
    number: 2,
    title: "Large-Scale Data and Specialised Tasks",
    topic: "Deep learning on large-scale data for a specialised vision or language task",
    status: "planned",
    abstract:
      "Scaling a pipeline to a large dataset, with the throughput and memory trade-offs that only appear at scale.",
    content: emptyContent(),
    resources: emptyResources(),
    aiDisclosure: emptyDisclosure(),
  },
  {
    slug: "assignment-3",
    number: 3,
    title: "Multimodal Deep Learning",
    topic: "Representation, fusion, and evaluation across modalities",
    status: "planned",
    abstract:
      "Joint modelling across modalities, evaluated so that agreement and conflict between modalities are separated.",
    content: emptyContent(),
    resources: emptyResources(),
    aiDisclosure: emptyDisclosure(),
  },
];

export function getAssignment(slug: string): Assignment | undefined {
  return assignments.find((assignment) => assignment.slug === slug);
}
