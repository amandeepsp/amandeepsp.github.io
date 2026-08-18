import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import YAML from "yaml";

const root = process.cwd();
const sharp = (await import("sharp")).default;
const contentRoot = path.join(root, "src/content/blog");
const outputRoot = path.join(root, "dist/og");
const font = await fs.readFile(path.join(root, "public/fonts/source-serif-4-variable.woff2"));
const fontData = font.toString("base64");

function escapeXml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&apos;");
}

function parsePost(source) {
    if (!source.startsWith("---")) throw new Error("post is missing frontmatter");
    const end = source.indexOf("\n---", 3);
    if (end < 0) throw new Error("post has unterminated frontmatter");
    return {
        data: YAML.parse(source.slice(4, end)),
        body: source.slice(end + 4)
    };
}

function wrapWords(value, maxCharacters) {
    const lines = [];
    let current = "";
    for (const word of value.split(/\s+/)) {
        const candidate = current ? `${current} ${word}` : word;
        if (candidate.length > maxCharacters && current) {
            lines.push(current);
            current = word;
        } else {
            current = candidate;
        }
    }
    if (current) lines.push(current);
    if (lines.length <= 3) return lines;
    return [
        ...lines.slice(0, 2),
        `${lines
            .slice(2)
            .join(" ")
            .slice(0, maxCharacters - 1)
            .trimEnd()}…`
    ];
}

function readingTime(body) {
    const words = body
        .replace(/<[^>]*>/g, " ")
        .trim()
        .split(/\s+/)
        .filter(Boolean).length;
    return `${Math.max(1, Math.ceil(words / 225))} min read`;
}

function formatDate(value) {
    return new Date(value).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric"
    });
}

function cardSvg(post) {
    const title = String(post.data.title);
    const fontSize = title.length > 64 ? 62 : title.length > 42 ? 72 : 82;
    const lines = wrapWords(title, title.length > 64 ? 34 : 29);
    const titleStart = lines.length === 1 ? 250 : lines.length === 2 ? 205 : 165;
    const titleLines = lines
        .map(
            (line, index) =>
                `<text x="72" y="${titleStart + index * (fontSize * 1.08)}" class="title" font-size="${fontSize}">${escapeXml(line)}</text>`
        )
        .join("");
    const subtitle = post.data.subTitle
        ? `<text x="72" y="${Math.min(478, titleStart + lines.length * fontSize * 1.08 + 34)}" class="subtitle">${escapeXml(post.data.subTitle)}</text>`
        : "";

    return `
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <style>
    @font-face {
      font-family: "Source Serif 4";
      src: url("data:font/woff2;base64,${fontData}") format("woff2");
    }
    text { fill: #171717; font-family: "Source Serif 4", Georgia, serif; }
    .title { font-weight: 650; }
    .subtitle { fill: #666; font-size: 34px; font-style: italic; }
    .meta { font-size: 24px; }
    .site { font-size: 28px; font-weight: 600; }
  </style>
  <rect width="1200" height="630" fill="#f2f1ec" />
  <line x1="72" x2="1128" y1="76" y2="76" stroke="#171717" stroke-width="2" stroke-dasharray="8 8" />
  <text x="72" y="56" class="site">amandeep singh</text>
  ${titleLines}
  ${subtitle}
  <text x="72" y="574" class="meta">${escapeXml(formatDate(post.data.publishDate))} · ${escapeXml(readingTime(post.body))}</text>
</svg>`;
}

async function findPosts(directory) {
    const posts = [];
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
        const fullPath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            posts.push(...(await findPosts(fullPath)));
        } else if (/\.mdx?$/.test(entry.name)) {
            const source = await fs.readFile(fullPath, "utf8");
            const post = parsePost(source);
            if (post.data.draft) continue;
            const relative = path.relative(contentRoot, fullPath);
            post.slug = path.basename(relative).startsWith("index.")
                ? path.dirname(relative).split(path.sep).join("/")
                : relative
                      .replace(/\.mdx?$/, "")
                      .split(path.sep)
                      .join("/");
            posts.push(post);
        }
    }
    return posts;
}

await fs.mkdir(outputRoot, { recursive: true });
const posts = await findPosts(contentRoot);
await Promise.all(
    posts.map(async (post) => {
        const output = path.join(outputRoot, `${post.slug}.png`);
        await fs.mkdir(path.dirname(output), { recursive: true });
        await sharp(Buffer.from(cardSvg(post)))
            .png()
            .toFile(output);
    })
);

console.log(`Generated ${posts.length} social cards in ${path.relative(root, outputRoot)}`);
