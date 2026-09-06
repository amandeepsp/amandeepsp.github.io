# Social-card fonts

These TTF files are decompressed copies of the matching Source Serif 4 Variable
WOFF2 files in `public/fonts`. Sharp's bundled Pango/FreeType renderer falls back
to a system font with those WOFF2 files, so social cards use TTF instead.
Both upright and italic faces are registered before rendering cards.

To regenerate after updating the web fonts, run from the repository root:

```sh
uv run --with fonttools --with brotli python - <<'PY'
from fontTools.ttLib import TTFont

for style in ("", "-italic"):
    font = TTFont(f"public/fonts/source-serif-4-variable{style}.woff2")
    font.flavor = None
    font.save(f"scripts/fonts/source-serif-4-variable{style}.ttf")
PY
```

This is a one-time asset conversion, not a build dependency.
