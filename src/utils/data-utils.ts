import { type CollectionEntry } from "astro:content";
import { slugify } from "./common-utils";

/**
 * Filters out draft posts based on environment and user preference
 * In production: always filter out drafts
 * In development: filter out drafts unless SHOW_DRAFTS=true
 */
export function filterDrafts(posts: CollectionEntry<"blog">[]): CollectionEntry<"blog">[] {
    const isDev = import.meta.env.DEV;
    const showDrafts = import.meta.env.SHOW_DRAFTS === "true";

    // In production, always filter out drafts
    // In development, show drafts only if SHOW_DRAFTS=true
    if (!isDev || !showDrafts) {
        return posts.filter((post) => !post.data.draft);
    }

    return posts;
}

export function sortItemsByDateDesc(
    itemA: CollectionEntry<"blog" | "projects">,
    itemB: CollectionEntry<"blog" | "projects">
) {
    return new Date(itemB.data.publishDate).getTime() - new Date(itemA.data.publishDate).getTime();
}

export function getAllTags(posts: CollectionEntry<"blog">[]) {
    const tags: string[] = [...new Set(posts.flatMap((post) => post.data.tags || []).filter(Boolean))];
    return tags
        .map((tag) => {
            return {
                name: tag,
                id: slugify(tag)
            };
        })
        .filter((obj, pos, arr) => {
            return arr.map((mapObj) => mapObj.id).indexOf(obj.id) === pos;
        });
}

export function getPostsByTag(posts: CollectionEntry<"blog">[], tagId: string) {
    const filteredPosts: CollectionEntry<"blog">[] = posts.filter((post) =>
        (post.data.tags || []).map((tag) => slugify(tag)).includes(tagId)
    );
    return filteredPosts;
}

export function calculateReadingTime(content?: string): string {
    // Remove HTML tags and get plain text
    const plainText = content?.replace(/<[^>]*>/g, "") ?? "";

    // Count words (split by whitespace and filter out empty strings)
    const words = plainText.split(/\s+/).filter((word) => word.length > 0);
    const wordCount = words.length;

    // Average reading speed is 200-250 words per minute, using 225 as middle ground
    const wordsPerMinute = 225;
    const minutes = Math.ceil(wordCount / wordsPerMinute);

    return `${minutes} min read`;
}
