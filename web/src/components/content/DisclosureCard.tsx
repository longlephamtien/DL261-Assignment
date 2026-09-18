import { SquareArrowOutUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { PendingBadge } from "@/components/content/PendingBadge";
import { displayValue } from "@/lib/content";
import { NO_AI_DECLARATION, type AiDisclosure, type AiToolUse } from "@/data/types";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1 sm:grid-cols-[9rem_1fr] sm:gap-4">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="text-sm">{children}</dd>
    </div>
  );
}

function Tags({ items }: { items: string[] }) {
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

function Entry({ entry }: { entry: AiToolUse }) {
  return (
    <div className="space-y-3">
      <h4 className="h-card">{entry.tool}</h4>
      <dl className="space-y-2">
        <Row label="Used by">{entry.usedBy}</Row>
        <Row label="Task">{entry.task}</Row>
        <Row label="Prompt summary">
          <span>{entry.promptSummary}</span>
          {entry.promptLog && (
            <a
              href={entry.promptLog}
              target="_blank"
              rel="noreferrer"
              className="ml-2 inline-flex items-center gap-1 font-medium text-primary hover:underline"
            >
              Prompt log
              <SquareArrowOutUpRight className="size-3.5" />
            </a>
          )}
        </Row>
        <Row label="AI contribution">{entry.aiContribution}</Row>
        <Row label="Student verification">{entry.studentVerification}</Row>
        <Row label="Affected files/sections">
          <Tags items={entry.affectedSections} />
        </Row>
        <Row label="Responsible member">{entry.responsibleMember}</Row>
      </dl>
    </div>
  );
}

export function DisclosureCard({ title, disclosure }: { title: string; disclosure: AiDisclosure }) {
  const summary = displayValue(disclosure.summary);
  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <h3 className="h-card">{title}</h3>
          {disclosure.noAiUsed ? (
            <p className="text-sm">{NO_AI_DECLARATION}</p>
          ) : summary ? (
            <p className="text-sm">{summary}</p>
          ) : (
            <PendingBadge label="Not written yet" />
          )}
        </div>
        {!disclosure.noAiUsed &&
          (disclosure.entries.length === 0 ? (
            <Row label="Tools used">
              <PendingBadge label="Not written yet" />
            </Row>
          ) : (
            disclosure.entries.map((entry, index) => (
              <div key={index} className="space-y-4">
                <Separator />
                <Entry entry={entry} />
              </div>
            ))
          ))}
      </CardContent>
    </Card>
  );
}
