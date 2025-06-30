import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

import partytown from '@astrojs/partytown';

export default defineConfig({
    site: 'https://amandeepsp.github.io.',
    integrations: [
        mdx(),
        sitemap(),
        tailwind(),
        partytown({
            config: {
                forward: ['dataLayer.push']
            }
        })
    ],
    markdown: {
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
        syntaxHighlight: 'shiki',
        shikiConfig: {
            themes: {
                light: 'kanagawa-lotus',
                dark: 'kanagawa-dragon'
            }
        }
    },
    redirects: {
        '/making-models-smaller-1': '/blog/making-ml-models-smaller'
    }
});
