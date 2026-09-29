# SYDE 671 website

Jekyll site based on [academicpages](https://github.com/academicpages/academicpages.github.io) (upstream commit `3d28cd2`), with a Tokyo Night theme (Moon by default, Day via the toggle).

Published at https://gabecasciano.github.io/SYDE671/ by `.github/workflows/pages.yml` on every push to `main` that touches `website/`.

## Local preview

```sh
cd website
docker compose up
```

Open http://localhost:4000/SYDE671/. Edits rebuild automatically; restart after changing `_config.yml`.

Without Docker, this needs `ruby-devel` (`sudo dnf install ruby-devel`), then:

```sh
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve
```

## Adding an assignment

1. Copy `_assignments/assignment-1.md` to `_assignments/assignment-N.md` and edit the front matter.
2. Put its media in `images/assignment-N/`.
3. It is listed automatically on the home and Assignments pages, at `/assignment-N/`.

Assignment files are plain Markdown with a little HTML and no Liquid, so the same file renders on the site, on GitHub, in an editor preview, or with pandoc. Reference media relative to the file:

```markdown
![Alt text](../images/assignment-1/photo.jpg)

<figure>
  <img src="../images/assignment-1/photo.jpg" alt="">
  <figcaption>Caption</figcaption>
</figure>

<div class="media-grid">
  <figure><img src="../images/assignment-1/a.jpg" alt=""><figcaption>A</figcaption></figure>
  <figure><img src="../images/assignment-1/b.jpg" alt=""><figcaption>B</figcaption></figure>
</div>

<video src="../images/assignment-1/clip.mp4" poster="../images/assignment-1/clip.jpg" controls loop muted playsinline></video>

$$ \frac{a}{\|a\|} \cdot \frac{b}{\|b\|} $$
```

`media-grid` fits as many columns as there is room for (min 200px each). GIFs are regular images.

Keep media small: JPEG around 2000px on the long side, and H.264 mp4 under 50 MB per file (GitHub rejects files over 100 MB).

```sh
magick in.jpg -resize 2000x2000\> -quality 85 out.jpg
ffmpeg -i in.mov -vcodec libx264 -crf 28 -pix_fmt yuv420p -an out.mp4
```

## Hand-in

- PDF: use the "Save as PDF" button on an assignment page, or Ctrl+P. Print uses the light palette and hides the navigation and sidebar. Videos print as their poster frame and GIFs as a single frame.
- Markdown: hand in `_assignments/assignment-N.md` together with `images/assignment-N/`.

## Where things are

- `_config.yml`: site settings and the author sidebar (name, bio, links).
- `_data/navigation.yml`: header links.
- `_pages/`: home (`about.md`), CV, assignments listing.
- `_sass/theme/_tokyonight_{dark,light}.scss`: palette.
- `_sass/layout/_custom.scss`: Tokyo Night tweaks, media grid, print styles.
- `assets/js/_main.js`: theme toggle (dark default). After editing it, run `npm install && npm run build:js` to rebuild `main.min.js`.
