import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import partytown from "@astrojs/partytown";
import { remarkAlert } from "remark-github-blockquote-alert";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";

/**
 * TODO: Nice to have features:
 * 1. Heading Links
 * 2. Better styling for remarkAlert
 */

export default defineConfig({
    site: "https://amandeepsp.github.io",
    integrations: [
        sitemap(),
        tailwind(),
        partytown({
            config: {
                forward: ["dataLayer.push"]
            }
        })
    ],
    markdown: {
        remarkPlugins: [remarkMath, remarkAlert, remarkGfm],
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
        "/making-models-smaller-1": "/blog/making-ml-models-smaller",
        "/ml-model-compression-part1": "/blog/making-ml-models-smaller"
    }
});
