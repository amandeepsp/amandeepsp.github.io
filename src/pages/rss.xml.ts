import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import sanitizeHtml from "sanitize-html";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { remarkAlert } from "remark-github-blockquote-alert";
import remarkRehype from "remark-rehype";
import rehypeStringify from "rehype-stringify";
import siteConfig from "../data/site-config.ts";
import { sortItemsByDateDesc, filterDrafts } from "../utils/data-utils.ts";
import type { APIContext } from "astro";

const processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkAlert)
    .use(remarkRehype)
    .use(rehypeStringify);

async function renderMarkdown(body: string): Promise<string> {
    const result = await processor.process(body);
    return sanitizeHtml(String(result), {
        allowedTags: sanitizeHtml.defaults.allowedTags.concat(["img"]),
    });
}

export async function GET(context: APIContext) {
    const posts = filterDrafts(await getCollection("blog")).sort(sortItemsByDateDesc);
    const items = await Promise.all(
        posts.map(async (item) => ({
            title: item.data.title,
            description: item.data.excerpt ?? "",
            link: `/blog/${item.id}/`,
            pubDate: item.data.publishDate,
            content: await renderMarkdown(item.body ?? ""),
        }))
    );
    return rss({
        title: siteConfig.title,
        description: siteConfig.description ?? "",
        site: context.site as URL,
        items,
    });
}
