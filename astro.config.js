import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { remarkAlert } from "remark-github-blockquote-alert";
import { rehypeHeadingIds, unified } from "@astrojs/markdown-remark";
import remarkGfm from "remark-gfm";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import mdx from "@astrojs/mdx";

export default defineConfig({
    site: "https://amandeepsp.github.io",
    integrations: [sitemap(), mdx()],
    markdown: {
        processor: unified({
            remarkPlugins: [remarkMath, remarkAlert, remarkGfm],
            rehypePlugins: [
                rehypeHeadingIds,
                [rehypeKatex, { strict: false }],
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
            ]
        }),
        syntaxHighlight: "shiki",
        shikiConfig: {
            themes: {
                light: "min-light",
                dark: "kanagawa-dragon"
            }
        }
    },
    redirects: {
        "/making-models-smaller-1": "/blog/making-ml-models-smaller",
        "/ml-model-compression-part1": "/blog/making-ml-models-smaller"
    }
});
