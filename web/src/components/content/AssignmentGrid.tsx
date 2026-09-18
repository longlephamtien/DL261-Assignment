import { ArrowRight, SquareArrowOutUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { STATUS_LABEL } from "@/lib/content";
import type { AssignmentStatus, ResourceLink } from "@/data/types";

export interface AssignmentCardData {
  slug: string;
  href: string;
  number: number;
  title: string;
  topic: string;
  status: AssignmentStatus;
  resources: ResourceLink[];
}

function Deliverables({ item }: { item: AssignmentCardData }) {
  const links = [
    ...item.resources,
    { label: "AI usage disclosure", href: `${item.href}#ai-usage` },
  ];
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1.5 text-sm">
      {links.map((link) => (
        <li key={link.label}>
          {link.href ? (
            <a
              href={link.href}
              target={link.href.startsWith("http") ? "_blank" : undefined}
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-medium text-primary hover:underline"
            >
              {link.label}
              {link.href.startsWith("http") && <SquareArrowOutUpRight className="size-3" />}
            </a>
          ) : (
            <span className="text-muted-foreground">{link.label}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

export function AssignmentGrid({ items }: { items: AssignmentCardData[] }) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {items.map((item) => (
        <Card key={item.slug} className="h-full">
          <CardContent className="flex h-full flex-col gap-3">
            <div className="flex items-center justify-between gap-2">
              <span className="eyebrow">Assignment {item.number}</span>
              <Badge variant="secondary">{STATUS_LABEL[item.status]}</Badge>
            </div>
            <h3 className="h-card">
              <a
                href={item.href}
                className="group inline-flex items-start gap-1.5 hover:text-primary focus-visible:ring-3 focus-visible:ring-ring/50"
              >
                {item.title}
                <ArrowRight className="mt-1 size-3.5 shrink-0 transition-transform group-hover:translate-x-0.5" />
              </a>
            </h3>
            <p className="grow text-sm text-muted-foreground">{item.topic}</p>
            <Separator />
            <Deliverables item={item} />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
