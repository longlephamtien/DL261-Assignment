import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "zod";

const assignments = defineCollection({
  loader: glob({ base: "./src/content/assignments", pattern: "**/*.mdx" }),
  schema: z.object({
    number: z.number().int().positive(),
    title: z.string(),
    topic: z.string(),
    status: z.enum(["planned", "in-progress", "submitted"]),
    abstract: z.string(),
    resources: z.array(
      z.object({
        label: z.string(),
        href: z.url().nullable(),
      })
    ),
    references: z
      .array(
        z.object({
          id: z.string(),
          text: z.string(),
          href: z.url().nullable().default(null),
        })
      )
      .default([]),
    aiDisclosure: z
      .object({
        noAiUsed: z.boolean().default(false),
        summary: z.string().nullable().default(null),
        entries: z
          .array(
            z.object({
              tool: z.string(),
              usedBy: z.string(),
              task: z.string(),
              promptSummary: z.string(),
              promptLog: z.url().nullable().default(null),
              aiContribution: z.string(),
              studentVerification: z.string(),
              affectedSections: z.array(z.string()).min(1),
              responsibleMember: z.string(),
            })
          )
          .default([]),
      })
      .refine((value) => !value.noAiUsed || value.entries.length === 0, {
        message: "noAiUsed cannot be true while entries are listed",
      }),
  }),
});

export const collections = { assignments };
