# amandeep singh

A small static Astro blog. Essays live in `src/content/blog`; finished media lives in `public`.
Small authored diagrams live in `src/diagrams` and are compiled from Typst/CeTZ to inline SVG during the Astro build.

```sh
npm ci
npm run dev
npm run check
npm run format:check
npm run build
```

`npm run build` creates the site and one social card per published post in `dist`. Drafts stay out of routes, feeds, sitemaps, and social cards.

Generated data-driven article media is maintained outside this repository and exported as finished files when needed:

```sh
cd ../blog-artifacts
uv run blog-artifacts export --site ../amandeepsp.github.io
```

The resume follows the same boundary:

```sh
cd ../resume
just export
```
