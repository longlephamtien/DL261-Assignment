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
    aiDisclosure: z.object({
      summary: z.string().nullable(),
      tools: z.array(z.string()),
      usedFor: z.array(z.string()),
      verification: z.string().nullable(),
    }),
  }),
});

export const collections = { assignments };
