# Assignment 4 Deliverables Checklist

Source: `syde671 Assignment 4.pdf` (borrowed from CS231A PSET 2). The handout has no Deliverables section. Items marked with a computer icon are code, items marked with a document icon go in the "written report". Submission format follows the first 3 assignments (Assignment 1 Deliverables: code zip, webpage link, PDF of the webpage), so the webpage is the written report.

Open question: the handout says "written report", not webpage. Confirm with the instructor or Learn whether the webpage PDF is acceptable as the report.

Lesson from Assignment 1 (85 instead of 95): the website only has to show the required results, but the submission package must contain every result produced, not just the ones on the page. Check the package against this file before uploading.

## 1. What gets submitted to Learn

Package goes in `Assignment 4/submission/` (ignored by git).

- [ ] Zip file uploaded to Learn, containing:
  - [ ] All code: everything in `Assignment 4/src/` (p1.py to p4.py, epipolar_utils.py, sfm_utils.py)
  - [ ] All generated figures: everything in `website/images/assignment-4/`, including those not shown on the page (see section 3)
  - [ ] A PDF of the webpage
  - [ ] A README that lists the contents and maps each script to its outputs (same format as `../Assignment 3/submission/`)
  - [ ] The provided data in a `data/` folder next to `src/` so the code runs as is
- [ ] Link to the hosted webpage: https://gabecasciano.github.io/SYDE671/assignment-4/
- [ ] PDF of the webpage (uploaded separately as well, as in Assignments 1 to 3)

## 2. What the report must contain

Website: `website/_assignments/assignment-4.md`. Run every script from `Assignment 4/` (e.g. `python src/p1.py`); the scripts load `data/...` relative to the working directory.

### 1. Fundamental Matrix Estimation (30 points), `p1.py`

- [ ] (a) Code: `lls_eight_point_alg()`, linear least squares eight-point, rank 2 enforced by SVD (15)
- [ ] (b) Report: resulting F to at least 4 decimal places, and a brief description of the implementation (5)
- [ ] (c) Code: `normalized_eight_point_alg()`, rank 2 enforced by SVD (2.5)
- [ ] (d) Report: normalized F to at least 4 decimal places (2.5)
- [ ] (e) Code: `compute_distance_to_epipolar_lines()`, average point-to-epipolar-line distance (2.5)
- [ ] (f) Report: brief description of `compute_distance_to_epipolar_lines()` (2.5)
- Run on both `data/set1` and `data/set2`; report F (both methods) and the distances for both sets. Show the epipolar line plots for both methods (not explicitly required, but the script produces them and they support the report).

### 2. Image Rectification (20 points), `p2.py` (uses `data/set1`)

- [ ] (a) Code: `compute_epipole()`, solve F^T e = 0 by SVD (2)
- [ ] (b) Code: `compute_H()` (handout calls it `find_H()`): H2 = T^-1 G R T (5)
  - T moves the image centre to the origin: translate by (-width/2, -height/2)
  - R rotates the translated epipole (e1, e2, 1) onto the x axis: rows [a e1/r, a e2/r, 0], [-a e2/r, a e1/r, 0], [0, 0, 1], r = sqrt(e1^2 + e2^2), a = 1 if e1 >= 0 else -1
  - G sends (f, 0, 1) to infinity: rows [1, 0, 0], [0, 1, 0], [-1/f, 0, 1]
- [ ] (c) Code: `compute_matching_homographies()` (10)
  - H1 = HA H2 M, M = [e]x F + e v^T, v = (1, 1, 1)
  - HA rows [a1, a2, a3], [0, 1, 0], [0, 0, 1]; solve W a = b by least squares, where W rows are (x_hat_i, y_hat_i, 1) and b entries are x_hat'_i, with x_hat = H2 M p and x_hat' = H2 p'
- [ ] (d) Report: the pair of rectified images, and a brief comment on why rectification makes correspondence search easier (3)

### 3. Factorization Method (20 points), `p3.py`

- [ ] (a) Code: `factorization_method()`, Tomasi-Kanade (8)
- [ ] (b) Report: brief description of the implementation (2)
- [ ] (c) Report: plot of the reconstruction vs ground truth (`data/set1`); explain why they match up to scale and rotation (3)
- [ ] (d) Report: the 4 singular values; why 4 are non-zero; how many would be non-zero in the ideal case and why (2)
- [ ] (e) Report: subset result (`data/set1_subset`) vs ground truth; explain why it no longer matches as well (3)
- [ ] (f) Report: the new singular values compared with the previous ones; explain major changes (2)

### 4. Triangulation in Structure From Motion (30 points), `p4.py`

- [ ] (a) Code: `estimate_initial_RT()`, the 4 candidate (R, T) from E = U D V^T, with Z, W as in the handout, R = det(Q) Q, T = +/- u3; the given example RT must be among them (5)
- [ ] (b) Code: `linear_estimate_3d_point()`, DLT with rows p_i,1 m_i^3T - m_i^1T and p_i,2 m_i^3T - m_i^2T, solved by SVD (5)
- [ ] (c) Code: `reprojection_error()` and `jacobian()`, J is 2K x 3 (5)
- [ ] (d) Code: `nonlinear_estimate_3d_point()`, Gauss-Newton, exactly 10 iterations of P = P - (J^T J)^-1 J^T e from the linear estimate (5)
- [ ] (e) Code: `estimate_RT_from_E()`, pick the (R, T) with the most points at positive depth in both camera frames (5)
- [ ] (f) Report: final plot of the reconstructed statue (`run_pipeline = True`); comment on why a larger scene such as a house would be harder (5)
- Show the unit test printouts for parts A to E (expected vs estimated) as evidence the code works.

### General

- [ ] Clear explanations of methods, with results shown (Assignment 1)
- [ ] "What I Learned": the most important thing learned (Assignment 2)
- [ ] Visually appealing and organized; images not oversized (Assignment 1)
- [ ] Final PDF print of the page checked for missing images, cut-off figures and matrices

## 3. Notes on the provided code

- `starter/` holds the untouched provided code and `PSET2.ipynb` (Colab version, not needed). Work happens in `src/`.
- `src/p4.py` line 216: `fig.gca(projection='3d')` was changed to `fig.add_subplot(projection='3d')`, since `gca` no longer takes `projection` in current Matplotlib.
- The scripts end with `plt.show()`; figures for the website need to be saved to `website/images/assignment-4/`.
- `data/statue/copy_dense_matches.npy` is not referenced by `p4.py`.

## 4. Generated figures not shown on the website

These must still go in the zip. To list them, run from `website/`:

```
for f in $(cd images/assignment-4 && find . -type f ! -name .gitkeep | sed 's|^\./||'); do
  grep -q "assignment-4/$f" _assignments/assignment-4.md || echo "$f"
done
```

## 5. Final check before upload

- [ ] `unzip -l` on the zip: every file in `src/`, `data/` and `website/images/assignment-4/` is present
- [ ] File counts match: `find website/images/assignment-4 -type f ! -name .gitkeep | wc -l` equals the figure count in the zip
- [ ] Every item in section 2 is checked
- [ ] Webpage link opens and the live page matches the PDF
