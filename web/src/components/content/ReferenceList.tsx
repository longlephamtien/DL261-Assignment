export interface Reference {
  id: string;
  text: string;
  href: string | null;
}

export function ReferenceList({ references }: { references: Reference[] }) {
  return (
    <ol className="space-y-3">
      {references.map((reference, index) => (
        <li
          key={reference.id}
          id={`ref-${reference.id}`}
          className="grid scroll-mt-24 grid-cols-[2.5rem_1fr] text-sm"
        >
          <span className="text-muted-foreground tabular-nums">[{index + 1}]</span>
          <span>
            {reference.text}
            {reference.href && (
              <>
                {" "}
                <a
                  href={reference.href}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline"
                >
                  Link
                </a>
              </>
            )}
          </span>
        </li>
      ))}
    </ol>
  );
}
