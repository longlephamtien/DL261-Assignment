import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export function PendingBadge({ label = "To be filled in", className }: { label?: string; className?: string }) {
  return (
    <Badge variant="outline" className={cn("border-dashed text-muted-foreground", className)}>
      {label}
    </Badge>
  );
}
