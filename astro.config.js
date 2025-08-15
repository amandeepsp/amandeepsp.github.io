import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import partytown from "@astrojs/partytown";
import { remarkAlert } from "remark-github-blockquote-alert";
import rehypeSlug from "rehype-slug";

/**
 * TODO: Nice to have features:
 * 1. Heading Links
 * 2. Better styling for remarkAlert
 */

export default defineConfig({
    site: "https://amandeepsp.github.io",
    integrations: [
        mdx(),
        sitemap(),
        tailwind(),
        partytown({
            config: {
                forward: ["dataLayer.push"]
            }
        })
    ],
    markdown: {
        remarkPlugins: [remarkMath, remarkAlert],
        rehypePlugins: [rehypeSlug, rehypeKatex],
        syntaxHighlight: "shiki",
        shikiConfig: {
            themes: {
                light: "rose-pine-dawn",
                dark: "kanagawa-dragon"
            }
        }
    },
    redirects: {
        "/making-models-smaller-1": "/blog/making-ml-models-smaller"
    }
});
