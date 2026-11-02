---
title: "Assignment 3: [Auto]Stitching Photo Mosaics"
date: 2026-11-01
excerpt: "Homographies, image warping, and mosaics, stitched by hand and then automatically with Harris corners, feature matching, and RANSAC."
---

## Procedure: Capturing the Source Images

The source images come from videos taken on an iPhone at the Grand Canyon. The frames were pulled out of the videos afterwards, which gives many closely spaced views of each scene to choose from.

### Extracting frames

`Assignment 3/src/extract_frames.py` turns each video into a folder of still frames:

1. ffmpeg decodes the video (iPhone HEVC) and samples it at a fixed rate, 2 frames per second by default. ffmpeg also applies the rotation stored in the file, so portrait videos come out upright.
2. iPhone video is recorded in HDR (HLG). When the file is tagged as HDR, the frames are tone mapped to standard range so they do not look washed out.
3. Each frame is compared to the last frame that was kept, using a 128-pixel-wide blurred grayscale copy of both. If the mean absolute difference is below a threshold (0.04 on a 0 to 1 scale by default), the frame is a near-duplicate and is dropped.

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

Pruning only removes frames where the camera barely moved, such as the start and end of a clip or pauses during the pan. Frames taken while the camera is moving are kept, which leaves a manageable set for choosing mosaic pairs.
