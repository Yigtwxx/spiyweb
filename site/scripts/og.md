# Regenerating the social card

One card serves both link previews: the site's `og:image` / `twitter:image`
(`public/social-preview.jpg`) and the GitHub repository's social preview
(`.github/social-preview.jpg`). It is the `/social` page, 1280×640, rendered in a
real browser so it uses the site's own fonts and halftones. It is captured at 2×
and scaled down, which keeps the dithered engraving sharp without moiré and the
file under GitHub's 1 MB limit:

1. `npm run build && npm run preview`, then open `http://localhost:4321/social` in Chrome.
2. Set the viewport to 1280×640 at device pixel ratio 2 and capture it (2560×1280).
3. Scale it to 1280×640 and save it as JPEG, quality 90, in both places:
   `sips -z 640 1280 capture.png --out small.png && sips -s format jpeg -s formatOptions 90 small.png --out public/social-preview.jpg && cp public/social-preview.jpg ../.github/social-preview.jpg`
4. Upload `.github/social-preview.jpg` in the repository's Settings → General →
   Social preview. GitHub has no API for this.

The page is `noindex` and excluded from the sitemap.
