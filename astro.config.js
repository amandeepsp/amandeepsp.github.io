import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import partytown from "@astrojs/partytown";
import { remarkAlert } from "remark-github-blockquote-alert";
import { rehypeHeadingIds } from "@astrojs/markdown-remark";
import remarkGfm from "remark-gfm";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import mdx from "@astrojs/mdx";

export default defineConfig({
    site: "https://amandeepsp.github.io",
    integrations: [
        sitemap(),
        tailwind(),
        mdx(),
        partytown({
            config: {
                forward: ["dataLayer.push"]
            }
        })
    ],
    markdown: {
        remarkPlugins: [remarkMath, remarkAlert, remarkGfm],
        rehypePlugins: [
            rehypeHeadingIds,
            rehypeKatex,
            [
                rehypeAutolinkHeadings,
                {
                    behavior: "append",
                    content: {
                        type: "text",
                        value: "#"
                    },
                    headingProperties: {
                        className: ["anchor"]
                    },
                    properties: {
                        className: ["anchor-link"]
                    }
                }
            ]
        ],
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
