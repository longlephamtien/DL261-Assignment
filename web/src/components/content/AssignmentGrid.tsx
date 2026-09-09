import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { STATUS_LABEL } from "@/lib/content";
import type { AssignmentStatus } from "@/data/types";

export interface AssignmentCardData {
  slug: string;
  href: string;
  number: number;
  title: string;
  topic: string;
  status: AssignmentStatus;
}

export function AssignmentGrid({ items }: { items: AssignmentCardData[] }) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {items.map((item) => (
        <a
          key={item.slug}
          href={item.href}
          className="group rounded-xl outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          <Card className="h-full transition-colors group-hover:bg-canvas">
            <CardContent className="flex h-full flex-col gap-3">
              <div className="flex items-center justify-between gap-2">
                <span className="eyebrow">Assignment {item.number}</span>
                <Badge variant="secondary">{STATUS_LABEL[item.status]}</Badge>
              </div>
              <h3 className="h-card">{item.title}</h3>
              <p className="grow text-sm text-muted-foreground">{item.topic}</p>
              <span className="inline-flex items-center gap-1.5 self-end text-sm font-medium text-primary">
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
