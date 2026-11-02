# Assignment 3 Deliverables Checklist

Source: `syde671 Assignment 3-1.pdf`, the Deliverables lines under each step, plus SCORING. Part B's "detailed description of the requirements and scoring" is `CS180_280A_ Intro to Computer Vision and Computational Photography.pdf` (adds B.5 and says the Part B submission includes Part A and Part B). Assignment 3 says "Please follow submissions as per the first 2 assignments", so the Deliverables sections of `../Assignment 1/SYDE671 assn1.pdf` and `../Assignment 2/SYDE671 assn2.pdf` also apply.

Lesson from Assignment 1 (85 instead of 95): the website only has to show the required results, but the submission package must contain every result produced, not just the ones on the page. Check the package against this file before uploading.

## 1. What gets submitted to Learn

Package goes in `Assignment 3/submission/` (ignored by git).

- [ ] Zip file uploaded to Learn, containing:
  - [ ] All code: everything in `Assignment 3/src/`
  - [ ] All generated images: everything in `website/images/assignment-3/`, including those not shown on the page (see section 3)
  - [ ] A PDF of the webpage
  - [ ] A README that lists the contents and maps each script to its output folder (same format as `../Assignment 2/submission/` README)
  - [ ] All input images and correspondence point files from `Assignment 3/data/`, in a `data/` folder next to `src/` so the code runs as is
  - [ ] Source videos as the converted MP4s in `website/images/assignment-3/videos/`, not the original iPhone .MOV files
- [ ] Link to the hosted webpage: https://gabecasciano.github.io/SYDE671/assignment-3/
- [ ] PDF of the webpage (uploaded separately as well, as in Assignments 1 and 2)

## 2. What the website must show

Website: `website/_assignments/assignment-3.md`.

Banned functions (Part A, and reused in Part B): `cv2.findHomography`, `cv2.warpPerspective`, `cv2.getPerspectiveTransform`, `skimage.transform.ProjectiveTransform`, `skimage.transform.warp`, `scipy.interpolate.griddata`, and similar high-level homography, warping or interpolation functions. Linear algebra helpers (solve, lstsq, inv) are allowed.

### Part A: Image Warping and Mosaicing (100 points)

- [ ] **A.1 Shoot the pictures (25)**
  - [ ] At least 2 sets of own photos with projective transforms between them (fixed centre of projection, rotate camera)
  - [ ] In practice 3 sets are needed, since A.4 and B.4 each need 3 mosaics
  - [ ] 40 to 70 percent overlap, shot close together in time, no fisheye
- [ ] **A.2 Recover homographies (25)**
  - [x] `computeH(im1_pts, im2_pts)` implemented, n-by-2 inputs, returns 3x3 H
  - [x] More than 4 correspondences, solved with least squares
  - [x] Correspondences visualized on the images
  - [x] System of equations shown
  - [x] Recovered homography matrix shown
- [ ] **A.3 Warp the images (25)**
  - [x] `warpImageNearestNeighbor(im, H)` implemented with inverse warping
  - [x] `warpImageBilinear(im, H)` implemented with inverse warping
  - [x] Both interpolations from scratch (no `griddata` or similar)
  - [x] Output size from the four corners pushed through H; alpha mask for empty pixels
  - [x] Rectification of at least 2 images (known rectangular object made rectangular)
  - [x] Results for both interpolation methods shown and compared on quality
  - [ ] Trade-offs between speed and quality discussed
- [ ] **A.4 Blend into a mosaic (25)**
  - [ ] 3 mosaics
  - [x] Source images shown for each mosaic
  - [x] Weighted averaging (feathering) or Laplacian pyramid blending to reduce edge artifacts
  - [ ] Procedure explained

### Part B: Feature Matching for Autostitching (100 points)

Follows Brown et al., "Multi-Image Matching using Multi-Scale Oriented Patches" (`MOPS.pdf`). Provided code: `starter/harris.py`.

- [x] **B.1 Harris corner detection (20)**
  - [x] Single-scale Harris (provided `harris.py` allowed)
  - [x] Figure of Harris corners overlaid on the image (without ANMS)
  - [x] Adaptive Non-Maximal Suppression (Section 3) implemented
  - [x] Figure of ANMS-chosen corners overlaid on the image
- [x] **B.2 Feature descriptor extraction (20)**
  - [x] Axis-aligned 8x8 patches sampled from a blurred 40x40 window
  - [x] Bias/gain normalized
  - [x] Several extracted features shown
- [ ] **B.3 Feature matching (20)**
  - [x] Lowe ratio test (first to second nearest neighbour)
  - [ ] Threshold chosen with reference to Figure 6b of the paper (0.5 used; justification is part of the write-up)
  - [x] Matched features shown between image pairs
- [ ] **B.4 RANSAC for robust homography (40)**
  - [x] 4-point RANSAC implemented from scratch
  - [x] Mosaics produced by reusing Part A code
  - [x] Manual and automatic stitching shown side by side
  - [ ] At least 3 automatic mosaics
- [ ] **B.5 Bells & Whistles (extra for CS180, mandatory for CS280A; all three done, write-up pending)**
  - [x] Multiscale processing for corner detection and description
  - [x] Rotation invariance for the descriptors
  - [x] Panorama recognition from an unordered set of images

### General

- [ ] Part A and Part B both on the one page (CS180 doc: Part B submission includes Part A and B)
- [ ] "What I Learned": the most important thing learned from the project (required in Assignment 2)
- [ ] Clear explanations of new concepts or tricks, with improved results shown (Assignment 1)
- [ ] Visually appealing and organized; images not oversized (Assignment 1)
- [ ] Final PDF print of the page checked for missing images and cut-off figures

## 3. Generated images not shown on the website

These must still go in the zip. To list them, run from `website/`:

```
for f in $(cd images/assignment-3 && find . -type f ! -name .gitkeep | sed 's|^\./||'); do
  grep -q "assignment-3/$f" _assignments/assignment-3.md || echo "$f"
done
```

## 4. Final check before upload

- [ ] `unzip -l` on the zip: every file in `src/`, `data/` and `website/images/assignment-3/` is present
- [ ] File counts match: `find website/images/assignment-3 -type f ! -name .gitkeep | wc -l` equals the image count in the zip
- [ ] Every item in section 2 is checked
- [ ] Webpage link opens and the live page matches the PDF
