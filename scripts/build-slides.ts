import { execSync } from "node:child_process";
import { globSync } from "node:fs";
import { resolve, join, basename } from "node:path";

const slidesDir = "slides";
const entries = globSync("*.md", { cwd: slidesDir });

for (const entry of entries) {
    const slug = entry === basename(entry) ? basename(entry, ".md") : entry.split("/")[0];
    const input = join(slidesDir, entry);
    const base = `/slides/${slug}/`;
    const out = resolve("dist", "slides", slug);
    console.log(`Building slide deck: ${slug}`);
    execSync(`bunx slidev build ${input} --base ${base} --out ${out}`, {
        stdio: "inherit"
    });
}

console.log(`Built ${entries.length} slide deck(s).`);
