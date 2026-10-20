# Assignment 2 Deliverables Checklist

Source: `SYDE671 assn2.pdf`, sections DELIVERABLES and SCORING. Assignment 2 says to submit "as described in the first assignment", so the Assignment 1 Deliverables section (`../Assignment 1/SYDE671 assn1.pdf`) also applies.

Lesson from Assignment 1 (85 instead of 95): the website only has to show the required results, but the submission package must contain every result produced, not just the ones on the page. Check the package against this file before uploading.

## 1. What gets submitted to Learn

Package built 2026-10-20 in `Assignment 2/submission/` (ignored by git): `SYDE671_Assignment_2.zip` (README, PDF, 10 scripts, 14 input files, 280 images) and `SYDE671_Assignment_2.pdf`. The unchecked items below are the uploads to Learn.

- [ ] Zip file uploaded to Learn, containing:
  - [x] All code: everything in `Assignment 2/src/` (`common.py`, `conv.py`, `finite_diff.py`, `dog.py`, `orientation.py`, `unsharp.py`, `hybrid.py`, `stacks.py`, `blend.py`, `show_points.py`)
  - [x] All generated images: everything in `website/images/assignment-2/` (280 files as of 2026-10-20, in folders `1.1`, `1.2`, `1.3`, `bw1`, `2.1`, `2.2`, `2.3`, `2.4`, `points`), including those not shown on the page (see section 3)
  - [x] A PDF of the webpage
  - [x] A README that lists the contents and maps each script to its output folder (same format as `../Assignment 1/submission/README.md`)
  - [x] All original input images from `Assignment 2/data/`, in a `data/` folder next to `src/` so the code runs as is (13 images plus `points.json`, about 11 MB): `big-cat.jpeg`, `burt_apple.png`, `burt_orange.png`, `camera-man.jpg`, `crazy-cat.jpg`, `DerekPicture.jpg`, `father.JPG`, `little-cat.jpeg`, `nutmeg.jpg`, `pasta-boy.jpg`, `sister-1.JPG`, `sister-2.JPG`, `taj.jpg`, `points.json`
- [ ] Link to the hosted webpage: https://gabecasciano.github.io/SYDE671/assignment-2/
- [ ] PDF of the webpage (uploaded separately as well, as in Assignment 1)

## 2. What the website must show

Website: `website/_assignments/assignment-2.md`. "Code snippets" are required on the website for 1.1 and asked for in the notes for 2.1, 2.3 and 2.4.

### Part 1: Filters and Edges (30 points)

- [x] **1.1 Convolutions from scratch (5)**
  - [x] Four-loop and two-loop convolution, numpy only, with zero padding
  - [x] Comparison with `scipy.signal.convolve2d`
  - [x] Commentary on runtime
  - [x] Commentary on boundary handling
  - [x] Picture of yourself, in grayscale, convolved with a 9x9 box filter
  - [x] Same picture convolved with Dx and Dy
  - [x] Code snippets on the website
- [x] **1.2 Finite difference operator (5)**
  - [x] Partial derivatives in x and y of cameraman
  - [x] Gradient magnitude image
  - [x] Binarized edge image
  - [x] Justification of the chosen threshold (tradeoff between edges and noise)
- [x] **1.3 Derivative of Gaussian (10)**
  - [x] Gaussian built with `cv2.getGaussianKernel` (outer product)
  - [x] Blurred cameraman, then repeat 1.2 on it
  - [x] "What differences do you see?" answered
  - [x] DoG filters (G convolved with Dx, Dy) shown as images
  - [x] DoG applied to cameraman, shown to match blur-then-difference
  - [x] Comparison with the finite difference results from 1.2
- [x] **Bells and whistles: gradient orientation (10, mandatory)**
  - [x] Orientation computed without built-in angle functions
  - [x] Visualized with HSV / cyclic colormap
  - [x] Process explained

### Part 2: Applications (65 points)

- [x] **2.1 Unsharp masking (15)**
  - [x] Single-convolution unsharp mask filter implemented
  - [x] Explanation of how it relates to blur filters and high frequencies
  - [x] Taj Mahal: blurred, high-frequency, and sharpened versions
  - [x] Another image of your choice: blurred, high-frequency, and sharpened versions
  - [x] Varying the sharpening amount and how it changes the result
  - [x] Sharp image, blurred, then re-sharpened; compared with the original, observations reported
- [x] **2.2 Hybrid images (20, bells and whistles mandatory)**
  - [x] Three hybrids: Derek + Nutmeg and two of your own
  - [x] For one hybrid, the full process: original images, aligned images, log-magnitude FFTs of both inputs, both filtered images and the hybrid, cutoff frequency choice, final image
  - [x] For the others: the input images and the final hybrid
  - [x] Bells and whistles: colour in the low component, high component, or both; which works better, with justification
- [x] **2.3 Gaussian and Laplacian stacks (10)**
  - [x] Stacks implemented from scratch (no `cv2.pyrDown`, no `skimage.transform.pyramid_gaussian`)
  - [x] Gaussian and Laplacian stacks shown for the apple and orange
  - [x] Recreation of Szeliski Figure 3.42 (a) through (l)
- [x] **2.4 Multiresolution blending (20, bells and whistles mandatory)**
  - [x] Oraple with a vertical or horizontal seam, using a Gaussian-stacked mask
  - [x] Two additional custom blends of your own images
  - [x] At least one with an irregular mask (the PDF details ask for two pairs with irregular masks)
  - [x] "Crazy ideas of your own" blend
  - [x] Laplacian stack of the favourite result with the masked inputs, like Figure 10 of Burt and Adelson
  - [x] Bells and whistles: colour, explored and justified

### General

- [x] "What I Learned": the most important thing learned from the project
- [x] Clarity (5): visualization quality, clear explanations, organization
- [x] Final PDF print of the page checked for missing images and cut-off figures

## 3. Generated images not shown on the website

These must still go in the zip. As of 2026-10-18:

- `2.4/oraple_a_l0.jpg`, `oraple_a_l2.jpg`, `oraple_a_l4.jpg`, `oraple_a_sum.jpg`
- `2.4/oraple_b_l0.jpg`, `oraple_b_l2.jpg`, `oraple_b_l4.jpg`, `oraple_b_sum.jpg`
- `2.4/oraple_blend_l0.jpg`, `oraple_blend_l2.jpg`, `oraple_blend_l4.jpg`, `oraple_blend_sum.jpg`
- `2.4/father_sister1_seam_mask.png`, `father_sister2_seam_mask.png`, `sister1_sister2_seam_mask.png`

To regenerate this list, run from `website/`:

```
for f in $(cd images/assignment-2 && find . -type f ! -name .gitkeep | sed 's|^\./||'); do
  grep -q "assignment-2/$f" _assignments/assignment-2.md || echo "$f"
done
```

## 4. Final check before upload

- [x] `unzip -l` on the zip: every file in `src/`, `data/` and `website/images/assignment-2/` is present
- [x] File counts match: `find website/images/assignment-2 -type f ! -name .gitkeep | wc -l` equals the image count in the zip
- [x] Every item in section 2 is checked
- [ ] Webpage link opens and the live page matches the PDF
