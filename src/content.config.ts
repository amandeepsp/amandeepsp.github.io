import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const seoSchema = z.object({
    title: z.string().min(5).max(120).optional(),
    description: z.string().min(15).max(160).optional(),
    image: z
        .object({
            src: z.string(),
            alt: z.string().optional()
        })
        .optional(),
    pageType: z.enum(["website", "article"]).default("website"),
    keywords: z.array(z.string()).optional(),
    noindex: z.boolean().default(false)
});

const blog = defineCollection({
    loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/blog" }),
    schema: z.object({
        title: z.string(),
        subTitle: z.string().optional(),
        excerpt: z.string().optional(),
        publishDate: z.coerce.date(),
        updatedDate: z.coerce.date().optional(),
        featured: z.boolean().default(false),
        draft: z.boolean().default(false),
        tags: z.array(z.string()).default([]),
        seo: seoSchema.optional()
    })
});

const pages = defineCollection({
    loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/pages" }),
    schema: z.object({
        title: z.string(),
        seo: seoSchema.optional()
    })
});     
export const collections = { blog, pages };
