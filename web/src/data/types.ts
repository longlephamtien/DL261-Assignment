export type Block =
  | { kind: "text"; value: string }
  | { kind: "list"; items: string[] }
  | { kind: "table"; head: string[]; rows: string[][] };

export type SectionId =
  | "problem-statement"
  | "dataset-and-eda"
  | "methodology"
  | "experimental-setup"
  | "results"
  | "comparison-and-discussion"
  | "error-analysis"
  | "limitations-and-conclusion";

export interface SectionMeta {
  id: SectionId;
  title: string;
}

export interface Section extends SectionMeta {
  blocks: Block[];
}

export type AssignmentContent = Record<SectionId, Block[]>;

export interface ResourceLink {
  label: string;
  href: string | null;
  description: string;
}

export interface AiDisclosure {
  summary: string | null;
  tools: string[];
  usedFor: string[];
  verification: string | null;
}

export interface Member {
  name: string;
  studentId: string;
  role: string | null;
  github: string | null;
}

export interface Institution {
  university: string;
  faculty: string;
}

export interface Course {
  name: string;
  code: string;
  semester: string;
  instructor: string;
}

export interface Group {
  id: string | null;
  name: string | null;
  repository: string | null;
  members: Member[];
}

export type AssignmentStatus = "planned" | "in-progress" | "submitted";

export interface Assignment {
  slug: string;
  number: number;
  title: string;
  topic: string;
  status: AssignmentStatus;
  abstract: string;
  content: AssignmentContent;
  resources: ResourceLink[];
  aiDisclosure: AiDisclosure;
}
