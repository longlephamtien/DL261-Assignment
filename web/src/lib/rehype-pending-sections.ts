import type { Element, Root, RootContent } from "hast";

const MAX_DEPTH = 3;

function headingDepth(node: RootContent): number | null {
  if (node.type !== "element") return null;
  const match = /^h([1-6])$/.exec(node.tagName);
  return match ? Number(match[1]) : null;
}

function isBlank(node: RootContent): boolean {
  return node.type === "text" && node.value.trim() === "";
}

function pendingBlock(): Element {
  return {
    type: "element",
    tagName: "p",
    properties: { className: ["section-pending"] },
    children: [{ type: "text", value: "Not written yet" }],
  };
}

export function rehypePendingSections() {
  return (tree: Root) => {
    const children: RootContent[] = [];

    tree.children.forEach((node, index) => {
      children.push(node);

      const depth = headingDepth(node);
      if (depth === null || depth > MAX_DEPTH) return;

      let next = index + 1;
      while (next < tree.children.length && isBlank(tree.children[next]!)) next++;

      const following = tree.children[next];
      const followingDepth = following ? headingDepth(following) : null;

      if (!following || (followingDepth !== null && followingDepth <= depth)) {
        children.push(pendingBlock());
      }
    });

    tree.children = children;
  };
}
