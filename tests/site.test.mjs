import assert from "node:assert/strict";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import sharp from "sharp";
import YAML from "yaml";

const ROOT = path.resolve(import.meta.dirname, "..");
const DIST = path.join(ROOT, "dist");
const CONTENT = path.join(ROOT, "src/content/blog");
const SITE = "https://amandeepsp.github.io";
const PUBLISHED_IDS = [
    "clojure-aoc21",
    "fs-watcher",
    "high-dims",
    "hnsw",
    "layout-algebra",
    "making-ml-models-smaller",
    "nvfp4-blackwell-gemv",
    "power-of-2"
];

async function filesUnder(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    const files = await Promise.all(
        entries.map(async (entry) => {
            const absolute = path.join(directory, entry.name);
            return entry.isDirectory() ? filesUnder(absolute) : [absolute];
        })
    );
    return files.flat();
}

function frontmatter(source) {
    const match = source.match(/^---\s*\n([\s\S]*?)\n---(?:\s*\n|$)/);
    assert.ok(match, "content file is missing YAML frontmatter");
    return YAML.parse(match[1]);
}

function contentId(file) {
    const relative = path.relative(CONTENT, file).replaceAll(path.sep, "/");
    return relative.replace(/\.(md|mdx)$/, "").replace(/\/index$/, "");
}

async function contentEntries() {
    const files = (await filesUnder(CONTENT)).filter((file) => /\.mdx?$/.test(file));
    return Promise.all(
        files.map(async (file) => ({
            id: contentId(file),
            data: frontmatter(await readFile(file, "utf8"))
        }))
    );
}

function pageUrl(htmlFile) {
    const relative = path.relative(DIST, htmlFile).replaceAll(path.sep, "/");
    if (relative === "index.html") return `${SITE}/`;
    if (relative.endsWith("/index.html")) return `${SITE}/${relative.slice(0, -10)}`;
    return `${SITE}/${relative}`;
}

async function exists(file) {
    try {
        return (await stat(file)).isFile();
    } catch {
        return false;
    }
}

async function resolvesFrom(reference, fromUrl) {
    if (!reference || reference.startsWith("#") || /^(?:data|mailto|tel|javascript):/i.test(reference)) {
        return true;
    }

    let url;
    try {
        url = new URL(reference, fromUrl);
    } catch {
        return false;
    }
    if (url.origin !== SITE) return true;

    const decodedPath = decodeURIComponent(url.pathname);
    const relative = decodedPath.replace(/^\/+/, "");
    const candidates = decodedPath.endsWith("/")
        ? [path.join(DIST, relative, "index.html")]
        : [path.join(DIST, relative), path.join(DIST, relative, "index.html")];
    return (await Promise.all(candidates.map(exists))).some(Boolean);
}

function attributeValues(html, attribute) {
    return [...html.matchAll(new RegExp(`\\b${attribute}=["']([^"']+)["']`, "gi"))].map((match) => match[1]);
}

test("generated routes preserve every published post, tag, redirect, and index", async () => {
    const entries = await contentEntries();
    const published = entries.filter(({ data }) => !data.draft);
    assert.deepEqual(
        entries.map(({ id }) => id).sort(),
        PUBLISHED_IDS,
        "draft content should not live in the site repository"
    );
    assert.deepEqual(published.map(({ id }) => id).sort(), PUBLISHED_IDS, "published post IDs changed");
    assert.deepEqual(
        entries.filter(({ data }) => data.series).map(({ id, data }) => ({ id, series: data.series })),
        [],
        "series assignments changed"
    );

    const tags = [...new Set(published.flatMap(({ data }) => data.tags ?? []))].sort();
    const expected = new Set([
        "404.html",
        "index.html",
        "blog/index.html",
        "contact-me/index.html",
        "tags/index.html",
        "making-models-smaller-1/index.html",
        "ml-model-compression-part1/index.html",
        ...PUBLISHED_IDS.map((id) => `blog/${id}/index.html`),
        ...tags.map((tag) => `tags/${tag}/index.html`)
    ]);
    const actual = new Set(
        (await filesUnder(DIST))
            .filter((file) => file.endsWith(".html"))
            .map((file) => path.relative(DIST, file).replaceAll(path.sep, "/"))
    );
    assert.deepEqual([...actual].sort(), [...expected].sort(), "generated HTML route inventory changed");
});

test("all internal links, images, styles, and scripts resolve", async () => {
    const htmlFiles = (await filesUnder(DIST)).filter((file) => file.endsWith(".html"));
    const broken = [];
    for (const file of htmlFiles) {
        const html = await readFile(file, "utf8");
        const references = [...attributeValues(html, "href"), ...attributeValues(html, "src")];
        for (const srcset of attributeValues(html, "srcset")) {
            references.push(...srcset.split(",").map((candidate) => candidate.trim().split(/\s+/)[0]));
        }
        for (const reference of references) {
            if (!(await resolvesFrom(reference, pageUrl(file)))) {
                broken.push(`${path.relative(DIST, file)} -> ${reference}`);
            }
        }
    }
    assert.deepEqual(broken, []);
});

test("email addresses are absent from generated HTML", async () => {
    const protectedAddresses = [["amandeepspdhr", "gmail.com"].join("@"), ["amandeepsp", "gmail.com"].join("@")];
    const htmlFiles = (await filesUnder(DIST)).filter((file) => file.endsWith(".html"));

    for (const file of htmlFiles) {
        const html = await readFile(file, "utf8");
        for (const address of protectedAddresses) {
            assert.doesNotMatch(
                html,
                new RegExp(address, "i"),
                `${path.relative(DIST, file)} exposes an email address`
            );
        }
    }

    const contactPage = await readFile(path.join(DIST, "contact-me", "index.html"), "utf8");
    assert.match(contactPage, /data-contact-email/);
    assert.doesNotMatch(contactPage, /href=["']mailto:/i);
});

test("published posts have canonical URLs and one deterministic social card", async () => {
    const cards = (await readdir(path.join(DIST, "og"))).filter((file) => file.endsWith(".png")).sort();
    assert.deepEqual(cards, PUBLISHED_IDS.map((id) => `${id}.png`).sort());

    for (const id of PUBLISHED_IDS) {
        const html = await readFile(path.join(DIST, "blog", id, "index.html"), "utf8");
        assert.match(html, new RegExp(`<link rel="canonical" href="${SITE}/blog/${id}/?"`));
        assert.match(html, new RegExp(`<meta property="og:image" content="${SITE}/og/${id}\\.png"`));
        assert.equal(await exists(path.join(DIST, "og", id, "index.html")), false);

        const metadata = await sharp(path.join(DIST, "og", `${id}.png`)).metadata();
        assert.equal(metadata.width, 1200);
        assert.equal(metadata.height, 630);
        assert.equal(metadata.format, "png");
    }
});

test("RSS is a summary feed with every published post", async () => {
    const rss = await readFile(path.join(DIST, "rss.xml"), "utf8");
    const items = [...rss.matchAll(/<item>([\s\S]*?)<\/item>/g)].map((match) => match[1]);
    const entries = await contentEntries();
    assert.equal(items.length, PUBLISHED_IDS.length);
    assert.doesNotMatch(rss, /<content:encoded/i);

    for (const id of PUBLISHED_IDS) {
        const item = items.find((candidate) => candidate.includes(`${SITE}/blog/${id}/`));
        const entry = entries.find((candidate) => candidate.id === id);
        const summary = entry?.data.excerpt ?? entry?.data.seo?.description ?? entry?.data.subTitle;
        assert.ok(item, `RSS entry missing for ${id}`);
        assert.match(item, /<title>.+<\/title>/s);
        assert.match(item, /<pubDate>.+<\/pubDate>/s);
        if (summary) assert.match(item, /<description>.+<\/description>/s);
        else assert.doesNotMatch(item, /<description>/s);
    }
});

test("sitemap URLs resolve and include all posts and tags", async () => {
    const sitemap = await readFile(path.join(DIST, "sitemap-0.xml"), "utf8");
    const urls = [...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map((match) => match[1]);
    const entries = await contentEntries();
    const tags = [...new Set(entries.filter(({ data }) => !data.draft).flatMap(({ data }) => data.tags ?? []))];
    for (const expected of [
        `${SITE}/`,
        `${SITE}/blog/`,
        `${SITE}/tags/`,
        ...PUBLISHED_IDS.map((id) => `${SITE}/blog/${id}/`),
        ...tags.map((tag) => `${SITE}/tags/${tag}/`)
    ]) {
        assert.ok(urls.includes(expected), `sitemap URL missing: ${expected}`);
    }
    for (const url of urls) assert.ok(await resolvesFrom(url, `${SITE}/`), `sitemap URL is broken: ${url}`);
});

test("resume export is the only resume source in the site", async () => {
    const pdf = await readFile(path.join(DIST, "resume.pdf"));
    assert.equal(pdf.subarray(0, 5).toString(), "%PDF-");
    const siteFiles = await filesUnder(ROOT);
    const resumeSources = siteFiles.filter(
        (file) => !file.includes(`${path.sep}node_modules${path.sep}`) && file.endsWith(".typ")
    );
    assert.deepEqual(resumeSources, []);
});
