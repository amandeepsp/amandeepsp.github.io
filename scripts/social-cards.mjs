import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import process from "node:process";
import { preview } from "astro";
import { chromium } from "playwright";

const root = process.cwd();
const outputRoot = path.join(root, "dist/og");

const slugs = (await fs.readdir(outputRoot, { recursive: true }))
    .filter((file) => file.endsWith(`${path.sep}index.html`))
    .map((file) => path.dirname(file).split(path.sep).join("/"));
const server = await preview({ server: { host: "127.0.0.1", port: 0, open: false } });
let browser;
try {
    browser = await chromium.launch();
    const card = await browser.newPage({ viewport: { width: 1200, height: 600 }, deviceScaleFactor: 1 });
    for (const slug of slugs) {
        await card.goto(`http://127.0.0.1:${server.port}/og/${slug}/`);
        await card.evaluate(() => document.fonts.ready);
        // Reject overflow instead of silently clipping or shrinking future titles.
        const fits = await card.evaluate(() => {
            const main = document.querySelector("main");
            return (
                main.getBoundingClientRect().bottom <=
                    innerHeight - parseFloat(getComputedStyle(document.body).paddingBottom) &&
                document.documentElement.scrollWidth === innerWidth
            );
        });
        if (!fits) throw new Error(`${slug}: social card overflows its overlay-safe area`);
        const output = path.join(outputRoot, `${slug}.png`);
        await fs.mkdir(path.dirname(output), { recursive: true });
        const png = await card.screenshot();
        await fs.writeFile(output, png);

        // Hash the rendered bytes so font and layout changes also invalidate cached cards.
        const version = createHash("sha256").update(png).digest("hex").slice(0, 16);
        const page = path.join(root, "dist/blog", slug, "index.html");
        const html = await fs.readFile(page, "utf8");
        const versioned = html.replace(
            /(<meta property="(?:og:image|twitter:image)" content=")([^"]+)(")/g,
            (tag, prefix, source, suffix) => {
                const url = new URL(source);
                if (url.pathname !== `/og/${slug}.png`) return tag;
                url.searchParams.set("v", version);
                return `${prefix}${url.toString()}${suffix}`;
            }
        );
        await fs.writeFile(page, versioned);
    }
} finally {
    await browser?.close();
    await server.stop();
}

console.log(`Generated ${slugs.length} social cards in ${path.relative(root, outputRoot)}`);
