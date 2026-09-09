import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

export interface NavItem {
  label: string;
  href: string;
  current: boolean;
}

export function MobileNav({ items, title }: { items: NavItem[]; title: string }) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open navigation">
          <Menu />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-72">
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
          <SheetDescription>Course project navigation</SheetDescription>
        </SheetHeader>
        <nav className="flex flex-col px-2">
          {items.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-lg px-2 py-2 text-sm transition-colors hover:bg-muted",
                item.current ? "bg-muted font-medium text-foreground" : "text-muted-foreground"
              )}
            >
              {item.label}
            </a>
          ))}
        </nav>
      </SheetContent>
    </Sheet>
  );
}
