import { chromium } from "playwright";
import fs from "fs";
import path from "path";

// List out dir src/content/blog to get all slugs,
// both dirs and files can be slugs
const slugs = fs.readdirSync("./src/content/blog")
  .map((file) => file.replace(/\.mdx?$/, ""));

const OUTPUT_DIR = "./public/og";
const SITE_URL = "http://localhost:4321";

fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const browser = await chromium.launch({ headless: true});
const page = await browser.newPage();

for (const slug of slugs) {

    const url = `${SITE_URL}/og/${slug}`;
    console.log(`Generating OG for ${slug}: ${url}`);

    await page.goto(url, { waitUntil: "load" });

    // Select your OG container by CSS selector
    const element = await page.$("#og-card"); // or '.og-container'

    if (!element) {
        console.warn(`No element found for ${slug}`);
        continue;
    }

    await element.screenshot({
        path: path.join(OUTPUT_DIR, `${slug}.png`)
    });

}

await browser.close();
