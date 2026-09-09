export interface ResourceLink {
  label: string;
  href: string | null;
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
