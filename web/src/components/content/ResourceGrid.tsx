import { SquareArrowOutUpRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { PendingBadge } from "@/components/content/PendingBadge";
import type { ResourceLink } from "@/data/types";

function ResourceBody({ resource }: { resource: ResourceLink }) {
  return (
    <Card className="h-full">
      <CardContent className="flex h-full items-center justify-between gap-3">
        <span className="h-card">{resource.label}</span>
        {resource.href ? (
          <SquareArrowOutUpRight className="size-4 shrink-0 text-muted-foreground" />
        ) : (
          <PendingBadge label="Pending" />
        )}
      </CardContent>
    </Card>
  );
}

export function ResourceGrid({ resources }: { resources: ResourceLink[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {resources.map((resource) =>
        resource.href ? (
          <a
            key={resource.label}
            href={resource.href}
            target="_blank"
            rel="noreferrer"
            className="rounded-xl outline-none transition-colors hover:**:data-[slot=card]:bg-canvas focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            <ResourceBody resource={resource} />
          </a>
        ) : (
          <div key={resource.label}>
            <ResourceBody resource={resource} />
          </div>
        )
      )}
    </div>
  );
}
