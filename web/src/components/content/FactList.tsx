import { displayValue } from "@/lib/content";

export interface Fact {
  label: string;
  value: string | null;
  href?: string | null;
}

export function FactList({ facts }: { facts: Fact[] }) {
  return (
    <dl className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
      {facts.map((fact) => {
        const value = displayValue(fact.value);
        return (
          <div key={fact.label}>
            <dt className="eyebrow">{fact.label}</dt>
            <dd className="mt-1 text-sm">
              {value ? (
                fact.href ? (
                  <a href={fact.href} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                    {value}
                  </a>
                ) : (
                  value
                )
              ) : (
                <span className="text-muted-foreground">Pending</span>
              )}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
