import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import siteConfig from "../data/site-config.ts";
import { sortItemsByDateDesc, filterDrafts } from "../utils/data-utils.ts";
import type { APIContext } from "astro";

export async function GET(context: APIContext) {
    const posts = filterDrafts(await getCollection("blog")).sort(sortItemsByDateDesc);
    return rss({
        title: siteConfig.title,
        description: siteConfig.description ?? "",
        site: context.site as URL,
        items: posts.map((item) => ({
            title: item.data.title,
            description: item.data.excerpt,
            link: `/blog/${item.id}/`,
            pubDate: item.data.publishDate
        }))
    });
}
