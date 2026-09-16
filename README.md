# amandeep singh

A small static Astro blog. Essays live in `src/content/blog`; finished media lives in `public`.

```sh
npm ci
npx playwright install --with-deps chromium
npm run dev
npm run check
npm run format:check
npm run build
```

`npm run build` creates the site and one social card per published post in `dist`. Drafts stay out of routes, feeds, sitemaps, and social cards.

Social cards are Astro pages at `/og/<post-id>/`, defined in `src/pages/og/[id].astro`
using the blog collection and the site's web fonts. The postbuild script starts a temporary
Astro preview server, screenshots those pages with Chromium, and stops the server.
Keep the bottom 80px clear for social-app title overlays. OG pages are excluded from the sitemap and marked noindex.

Generated article media is maintained outside this repository and exported as finished files when needed:

```sh
cd ../blog-artifacts
uv run blog-artifacts export --site ../amandeepsp.github.io
```

The resume follows the same boundary:

```sh
cd ../resume
just export
```
