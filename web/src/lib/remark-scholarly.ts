import type { Parent, Root, RootContent } from "mdast";
import { visit } from "unist-util-visit";

type DirectiveNode = Parent & {
  type: "containerDirective" | "textDirective" | "leafDirective";
  name: string;
  attributes?: Record<string, string | null | undefined>;
  data?: { hName?: string; hProperties?: Record<string, unknown> };
};

const FLOAT_KINDS = new Set(["figure", "table"]);

/**
 * `:ref[fig:curves]` parses as text `fig` followed by a nested `:curves`
 * directive, because a colon opens a new text directive. Walk the children to
 * put the original label back together.
 */
function labelOf(node: Parent): string {
  return node.children
    .map((child: RootContent) => {
      if (child.type === "text") return child.value;
      const nested = child as unknown as DirectiveNode;
      if (nested.type === "textDirective") {
        return `:${nested.name}${labelOf(nested)}`;
      }
      return "children" in child ? labelOf(child as Parent) : "";
    })
    .join("")
    .trim();
}

function setHast(node: DirectiveNode, hName: string, hProperties: Record<string, unknown>) {
  node.data = { ...node.data, hName, hProperties };
}

export function remarkScholarly() {
  return (tree: Root) => {
    visit(tree, (node) => {
      const directive = node as unknown as DirectiveNode;

      if (directive.type === "containerDirective" && FLOAT_KINDS.has(directive.name)) {
        const label = directive.attributes?.label ?? null;
        const children = directive.children as Parent[];
        const captionIndex = children.findLastIndex((child) => child.type === "paragraph");

        if (captionIndex !== -1) {
          const caption = children[captionIndex]!;
          caption.data = { ...caption.data, hName: "figcaption" };
          if (directive.name === "table") {
            children.splice(captionIndex, 1);
            children.unshift(caption);
          }
        }

        setHast(directive, "figure", { id: label, dataFloat: directive.name });
        return;
      }

      if (directive.type === "textDirective" && directive.name === "ref") {
        setHast(directive, "span", { dataRef: labelOf(directive) });
        directive.children = [];
        return;
      }

      if (directive.type === "textDirective" && directive.name === "cite") {
        setHast(directive, "span", { dataCite: labelOf(directive) });
        directive.children = [];
      }
    });
  };
}
