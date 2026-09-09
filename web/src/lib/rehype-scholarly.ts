import type { Element, Root } from "hast";
import type { VFile } from "vfile";
import { visit } from "unist-util-visit";

type FloatKind = "figure" | "table";

const LABEL_PREFIX: Record<FloatKind, string> = {
  figure: "Figure",
  table: "Table",
};

interface FloatTarget {
  kind: FloatKind;
  number: number;
}

function referenceIds(file: VFile): string[] {
  const frontmatter = (file.data as { astro?: { frontmatter?: Record<string, unknown> } }).astro
    ?.frontmatter;
  const references = frontmatter?.references;
  if (!Array.isArray(references)) return [];
  return references.map((entry) => String((entry as { id?: unknown }).id ?? ""));
}

function label(kind: FloatKind, number: number): string {
  return `${LABEL_PREFIX[kind]} ${number}`;
}

export function rehypeScholarly() {
  return (tree: Root, file: VFile) => {
    const targets = new Map<string, FloatTarget>();
    const counters: Record<FloatKind, number> = { figure: 0, table: 0 };

    visit(tree, "element", (node: Element) => {
      const kind = node.properties?.dataFloat as FloatKind | undefined;
      if (node.tagName !== "figure" || (kind !== "figure" && kind !== "table")) return;

      const number = ++counters[kind];
      const id = typeof node.properties?.id === "string" ? node.properties.id : null;
      if (id) targets.set(id, { kind, number });

      const caption = node.children.find(
        (child): child is Element => child.type === "element" && child.tagName === "figcaption"
      );

      if (caption) {
        caption.children.unshift({
          type: "element",
          tagName: "strong",
          properties: {},
          children: [{ type: "text", value: `${label(kind, number)}.` }],
        });
        caption.children.splice(1, 0, { type: "text", value: " " });
      }
    });

    const ids = referenceIds(file);

    visit(tree, "element", (node: Element) => {
      const refKey = node.properties?.dataRef;
      if (typeof refKey === "string") {
        const target = targets.get(refKey);
        delete node.properties.dataRef;
        node.tagName = "a";
        node.properties = { ...node.properties, href: `#${refKey}`, className: ["xref"] };
        node.children = [
          { type: "text", value: target ? label(target.kind, target.number) : `?${refKey}` },
        ];
        if (!target) file.message(`Unresolved reference: ${refKey}`);
        return;
      }

      const citeKey = node.properties?.dataCite;
      if (typeof citeKey === "string") {
        const index = ids.indexOf(citeKey);
        delete node.properties.dataCite;
        node.tagName = "a";
        node.properties = { ...node.properties, href: `#ref-${citeKey}`, className: ["citation"] };
        node.children = [{ type: "text", value: index === -1 ? "[?]" : `[${index + 1}]` }];
        if (index === -1) file.message(`Unknown citation key: ${citeKey}`);
      }
    });
  };
}
