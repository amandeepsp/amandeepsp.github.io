import assert from "node:assert/strict";
import test from "node:test";
import { chromium } from "playwright";

// Run against the dev server: node --test tests/responsive.test.mjs
test("article layout survives resizing and desktop text reflow", async () => {
    const browser = await chromium.launch({
        executablePath: process.env.CHROMIUM_PATH
    });
    try {
        const page = await browser.newPage();
        await page.addInitScript(() => {
            window.resizeRegistrations = 0;
            const addEventListener = window.addEventListener;
            window.addEventListener = function (type, ...args) {
                if (type === "resize") window.resizeRegistrations++;
                return addEventListener.call(this, type, ...args);
            };
        });
        await page.goto(`${process.env.SITE_URL ?? "http://localhost:4321"}/blog/spec-decode/`);
        await page.evaluate(() => document.fonts.ready);
        const settle = () =>
            page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await settle();
        const registrations = await page.evaluate(() => window.resizeRegistrations);

        for (const width of [390, 1280, 1279, 1440, 320, 1280, 768, 1440]) {
            await page.setViewportSize({ width, height: 900 });
            await settle();
            const layout = await page.evaluate(() => ({
                width: document.documentElement.scrollWidth,
                notes: document.querySelectorAll(".sidenote").length,
                references: document.querySelectorAll("a[data-footnote-ref]").length,
                footnotesHidden: document.querySelector(".footnotes").classList.contains("footnotes-hidden"),
                registrations: window.resizeRegistrations
            }));
            assert.ok(layout.width <= width, `horizontal overflow at ${width}px`);
            assert.equal(layout.notes, width >= 1280 ? layout.references : 0);
            assert.equal(layout.footnotesHidden, width >= 1280);
            assert.equal(layout.registrations, registrations, "resize listeners must not accumulate");
        }

        // Model text reflow (for example, a font finishing loading) without crossing a breakpoint.
        await page.locator(".prose-blog").evaluate((element) => (element.style.fontSize = "24px"));
        await page.setViewportSize({ width: 1400, height: 900 });
        await settle();
        const overlaps = await page.locator(".sidenote").evaluateAll(
            (notes) =>
                notes.slice(1).filter((note, index) => {
                    const previous = notes[index].getBoundingClientRect();
                    return note.getBoundingClientRect().top < previous.bottom + 15;
                }).length
        );
        assert.equal(overlaps, 0, "sidenotes must be repositioned after text reflow");
    } finally {
        await browser.close();
    }
});

test("scroll-to-top visibility and keyboard activation respect motion preferences", async () => {
    const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
    try {
        const page = await browser.newPage();
        await page.goto(`${process.env.SITE_URL ?? "http://localhost:4321"}/blog/spec-decode/`);
        const button = page.getByRole("button", { name: "Scroll to top", exact: true });
        await page.waitForFunction(() => document.querySelector(".scroll-to-top")?.hidden);
        assert.equal(await button.isVisible(), false);

        for (const reducedMotion of ["no-preference", "reduce"]) {
            await page.emulateMedia({ reducedMotion });
            await page.evaluate(() => window.scrollTo(0, 400));
            await page.waitForFunction(() => window.scrollY === 400);
            assert.equal(await button.isVisible(), false, "hidden at the threshold");
            await page.evaluate(() => window.scrollTo(0, 401));
            await button.waitFor({ state: "visible" });
            await button.focus();
            await page.keyboard.press("Enter");
            await page.waitForFunction(() => window.scrollY === 0);
            await button.waitFor({ state: "hidden" });
        }
    } finally {
        await browser.close();
    }
});
