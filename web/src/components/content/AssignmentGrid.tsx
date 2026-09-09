import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { STATUS_LABEL } from "@/lib/content";
import type { Assignment } from "@/data/types";

export interface AssignmentLink {
  assignment: Assignment;
  href: string;
}

export function AssignmentGrid({ items }: { items: AssignmentLink[] }) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {items.map(({ assignment, href }) => (
        <a
          key={assignment.slug}
          href={href}
          className="group rounded-xl outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          <Card className="h-full transition-colors group-hover:bg-canvas">
            <CardContent className="flex h-full flex-col gap-3">
              <div className="flex items-center justify-between gap-2">
                <span className="eyebrow">Assignment {assignment.number}</span>
                <Badge variant="secondary">{STATUS_LABEL[assignment.status]}</Badge>
              </div>
              <h3 className="h-card">{assignment.title}</h3>
              <p className="grow text-sm text-muted-foreground">{assignment.topic}</p>
              <span className="inline-flex items-center gap-1.5 text-sm font-medium text-primary">
                Open
                <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
              </span>
            </CardContent>
          </Card>
        </a>
      ))}
    </div>
  );
}
