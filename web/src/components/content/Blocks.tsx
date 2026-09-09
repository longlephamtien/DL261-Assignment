import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PendingBadge } from "@/components/content/PendingBadge";
import { isPending } from "@/lib/content";
import { cn } from "@/lib/utils";
import type { Block } from "@/data/types";

const pendingText = "text-muted-foreground";

function TextBlock({ value }: { value: string }) {
  return (
    <p className={cn("max-w-prose text-sm leading-relaxed", isPending(value) && pendingText)}>{value}</p>
  );
}

function ListBlock({ items }: { items: string[] }) {
  return (
    <ul className="max-w-prose space-y-1.5 text-sm">
      {items.map((item, index) => (
        <li key={index} className="flex gap-2.5 leading-relaxed">
          <span className="mt-[0.45rem] size-1.5 shrink-0 rounded-full bg-primary" />
          <span className={cn(isPending(item) && pendingText)}>{item}</span>
        </li>
      ))}
    </ul>
  );
}

function TableBlock({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto rounded-lg ring-1 ring-foreground/10">
      <Table>
        <TableHeader>
          <TableRow>
            {head.map((cell, index) => (
              <TableHead key={index}>{cell}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, rowIndex) => (
            <TableRow key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <TableCell key={cellIndex} className={cn(isPending(cell) && pendingText)}>
                  {cell}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function Blocks({ blocks }: { blocks: Block[] }) {
  if (blocks.length === 0) {
    return <PendingBadge label="Not written yet" />;
  }

  return (
    <div className="space-y-4">
      {blocks.map((block, index) => {
        if (block.kind === "text") return <TextBlock key={index} value={block.value} />;
        if (block.kind === "list") return <ListBlock key={index} items={block.items} />;
        return <TableBlock key={index} head={block.head} rows={block.rows} />;
      })}
    </div>
  );
}
