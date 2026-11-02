---
title: "Assignment 3: [Auto]Stitching Photo Mosaics"
date: 2026-11-01
excerpt: "Homographies, image warping, and mosaics, stitched by hand and then automatically with Harris corners, feature matching, and RANSAC."
---

## Procedure: Capturing the Source Images

The source images come from videos taken on an iPhone at the Grand Canyon. The frames were pulled out of the videos afterwards, which gives many closely spaced views of each scene to choose from.

<div class="media-grid" style="grid-template-columns: 1.78fr 0.56fr">
  <figure>
    <video src="../images/assignment-3/videos/IMG_0852.mp4" poster="../images/assignment-3/videos/IMG_0852_poster.jpg" controls loop muted playsinline></video>
    <figcaption>IMG_0852, landscape, 34 s</figcaption>
  </figure>
  <figure>
    <video src="../images/assignment-3/videos/IMG_0861.mp4" poster="../images/assignment-3/videos/IMG_0861_poster.jpg" controls loop muted playsinline></video>
    <figcaption>IMG_0861, portrait, 87 s</figcaption>
  </figure>
</div>

### Extracting frames

`Assignment 3/src/extract_frames.py` turns each video into a folder of still frames:

1. ffmpeg decodes the video (iPhone HEVC) and samples it at a fixed rate, 2 frames per second by default. ffmpeg also applies the rotation stored in the file, so portrait videos come out upright.
2. iPhones can record video in HDR (HLG). When a file is tagged as HDR, the frames are tone mapped to standard range so they do not look washed out. Both videos here are standard range, so this step was skipped.
3. Each frame is compared to the last frame that was kept, using a 128-pixel-wide blurred grayscale copy of both. If the mean absolute difference is below a threshold (0.1 on a 0 to 1 scale), the frame is too similar and is dropped.

```python
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")

vf = f"fps={fps}" + ("," + TONEMAP if is_hdr(video) else "")
subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf,
                "-q:v", "2", f"{tmp}/%05d.jpg"], check=True)

last, kept = None, 0
for f in frames:
    t = thumb(f)
    if last is None or np.abs(t - last).mean() > diff:
        kept += 1
        shutil.move(f, out_dir / f"{video.stem}_{kept:04d}.jpg")
        last = t
```

| Video | Sampled frames | Kept frames |
|---|---|---|
| IMG_0852 | 68 | 13 |
| IMG_0861 | 174 | 81 |

Because the camera pans continuously in both videos, almost every pair of frames half a second apart differed by more than 0.04, so a low threshold kept nearly everything. Raising it to 0.1 keeps a frame only after the view has moved a fair amount, which left neighbouring kept frames overlapping by roughly half.
