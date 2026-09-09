import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { PendingBadge } from "@/components/content/PendingBadge";
import { displayValue } from "@/lib/content";
import type { AiDisclosure } from "@/data/types";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-[7rem_1fr] sm:gap-4">
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="text-sm">{children}</div>
    </div>
  );
}

function Tags({ items }: { items: string[] }) {
  if (items.length === 0) return <PendingBadge label="Not written yet" />;

  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item, index) => (
        <Badge key={index} variant="secondary">
          {item}
        </Badge>
      ))}
    </div>
  );
}

function Text({ value }: { value: string | null }) {
  const text = displayValue(value);
  return text ? <span>{text}</span> : <PendingBadge label="Not written yet" />;
}

export function DisclosureCard({ title, disclosure }: { title: string; disclosure: AiDisclosure }) {
  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <h3 className="h-card">{title}</h3>
          <div className="text-sm">
            <Text value={disclosure.summary} />
          </div>
        </div>
        <Row label="Tools">
          <Tags items={disclosure.tools} />
        </Row>
        <Row label="Used for">
          <Tags items={disclosure.usedFor} />
        </Row>
        <Row label="Verification">
          <Text value={disclosure.verification} />
        </Row>
      </CardContent>
    </Card>
  );
}
