// @ts-check
import { defineConfig } from 'astro/config';

import { unified } from '@astrojs/markdown-remark';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import rehypeKatex from 'rehype-katex';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';

import { rehypePendingSections } from './src/lib/rehype-pending-sections';
import { rehypeScholarly } from './src/lib/rehype-scholarly';
import { remarkScholarly } from './src/lib/remark-scholarly';

export default defineConfig({
  site: 'https://longlephamtien.github.io',
  base: '/DL261-Assignment',
  trailingSlash: 'ignore',
  integrations: [react(), mdx()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkDirective, remarkScholarly],
      rehypePlugins: [rehypeKatex, rehypeScholarly, rehypePendingSections],
    }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      wrap: true,
    },
  },
  vite: {
    plugins: [tailwindcss()],
  },
});
