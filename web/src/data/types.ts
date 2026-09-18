export interface ResourceLink {
  label: string;
  href: string | null;
}

export interface AiToolUse {
  tool: string;
  usedBy: string;
  task: string;
  promptSummary: string;
  promptLog: string | null;
  aiContribution: string;
  studentVerification: string;
  affectedSections: string[];
  responsibleMember: string;
}

export interface AiDisclosure {
  noAiUsed: boolean;
  summary: string | null;
  entries: AiToolUse[];
}

export const NO_AI_DECLARATION =
  "The group declares that no generative AI tool was used in this assignment.";

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
