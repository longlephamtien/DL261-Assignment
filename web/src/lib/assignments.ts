import { getCollection } from "astro:content";
import { assignmentUrl } from "@/lib/paths";

export async function listAssignments() {
  const entries = await getCollection("assignments");
  return entries
    .sort((a, b) => a.data.number - b.data.number)
    .map((entry) => ({ ...entry.data, slug: entry.id, href: assignmentUrl(entry.id) }));
}

export type AssignmentSummary = Awaited<ReturnType<typeof listAssignments>>[number];
