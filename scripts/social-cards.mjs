import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import process from "node:process";
import YAML from "yaml";

const root = process.cwd();
process.env.FONTCONFIG_FILE = path.join(root, "scripts/fonts/fontconfig.conf");
const sharp = (await import("sharp")).default;
const contentRoot = path.join(root, "src/content/blog");
const outputRoot = path.join(root, "dist/og");
const card = { width: 1200, height: 600, padding: 72, textTop: 105, textHeight: 360 };

// Sharp's Pango renderer needs TTF rather than the site's WOFF2 files.
const cardFont = path.join(root, "scripts/fonts/source-serif-4-variable.ttf");

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

async function renderText(text, height) {
    return sharp({
        text: {
            text: `<span foreground="#171717">${text}</span>`,
            font: "Source Serif 4 Variable",
            fontfile: cardFont,
            width: card.width - 2 * card.padding,
            ...(height ? { height } : { dpi: 72 }),
            wrap: "word-char",
            rgba: true
        }
    })
        .png()
        .toBuffer({ resolveWithObject: true });
}

async function renderCard(post) {
    // Pango wraps and fits the entire block, preserving the title/subtitle size ratio.
    const subtitle = post.data.subTitle
        ? `\n<span size="28pt">\n</span><span size="34pt" style="italic" foreground="#666666">${escapeXml(post.data.subTitle)}</span>`
        : "";
    const content = await renderText(
        `<span size="82pt" weight="semibold">${escapeXml(post.data.title)}</span>${subtitle}`,
        card.textHeight
    );
    const site = await renderText('<span size="28pt" weight="semibold">amandeep singh</span>');
    const meta = await renderText(
        `<span size="24pt">${escapeXml(formatDate(post.data.publishDate))} · ${escapeXml(readingTime(post.body))}</span>`
    );
    const background = `<svg xmlns="http://www.w3.org/2000/svg" width="${card.width}" height="${card.height}">
      <rect width="100%" height="100%" fill="#f2f1ec" />
      <line x1="${card.padding}" x2="${card.width - card.padding}" y1="76" y2="76" stroke="#171717" stroke-width="2" stroke-dasharray="8 8" />
    </svg>`;
    return sharp(Buffer.from(background))
        .composite([
            { input: site.data, left: card.padding, top: 30 },
            {
                input: content.data,
                left: card.padding,
                top: card.textTop + Math.floor((card.textHeight - content.info.height) / 2)
            },
            { input: meta.data, left: card.padding, top: 522 }
        ])
        .png();
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
        const png = await (await renderCard(post)).toBuffer();
        await fs.writeFile(output, png);

        // Hash the rendered bytes so font and layout changes also invalidate cached cards.
        const version = createHash("sha256").update(png).digest("hex").slice(0, 16);
        const page = path.join(root, "dist/blog", post.slug, "index.html");
        const html = await fs.readFile(page, "utf8");
        const versioned = html.replace(
            /(<meta property="(?:og:image|twitter:image)" content=")([^"]+)(")/g,
            (tag, prefix, source, suffix) => {
                const url = new URL(source);
                if (url.pathname !== `/og/${post.slug}.png`) return tag;
                url.searchParams.set("v", version);
                return `${prefix}${url.toString()}${suffix}`;
            }
        );
        await fs.writeFile(page, versioned);
    })
);

console.log(`Generated ${posts.length} social cards in ${path.relative(root, outputRoot)}`);
