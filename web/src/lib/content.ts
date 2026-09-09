import { course, group } from "@/data/site";
import type { AssignmentStatus } from "@/data/types";

export function isPending(value: string | null | undefined): boolean {
  return !value || value.trim() === "";
}

export function displayValue(value: string | null | undefined): string | null {
  return isPending(value) ? null : (value as string);
}

export function githubAvatar(profile: string | null | undefined): string | null {
  if (!profile) return null;
  try {
    const { hostname, pathname } = new URL(profile);
    if (hostname !== "github.com" && hostname !== "www.github.com") return null;
    const login = pathname.split("/").filter(Boolean)[0];
    return login ? `https://github.com/${login}.png?size=160` : null;
  } catch {
    return null;
  }
}

export function videoTitle(assignmentNumber: number): string {
  const id = displayValue(group.id) ?? "[ID]";
  const semester = course.semester.trim().replace(/\s+/g, "-");
  return `${course.code}-${semester} – Group ${id} – Assignment ${assignmentNumber}`;
}

export function groupLabel(): string {
  const name = displayValue(group.name);
  const id = displayValue(group.id);
  if (name && id) return `${name} · Group ${id}`;
  return name ?? (id ? `Group ${id}` : "Group details pending");
}

export const STATUS_LABEL: Record<AssignmentStatus, string> = {
  planned: "Planned",
  "in-progress": "In progress",
  submitted: "Submitted",
};
