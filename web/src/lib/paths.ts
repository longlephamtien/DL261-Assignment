const base = import.meta.env.BASE_URL.replace(/\/$/, "");

export function url(path: string): string {
  return `${base}/${path.replace(/^\//, "")}`.replace(/\/$/, "") || "/";
}

export function assignmentUrl(slug: string): string {
  return url(`assignments/${slug}`);
}
