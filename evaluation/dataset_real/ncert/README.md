# Real slice: NCERT worksheet pages (verification pending)

20 page images supplied by the project owner on 2026-10-03: 18 screenshots of
NCERT mathematics textbook / exemplar pages (Classes 7–10: perimeter and area,
quadrilaterals, triangles, areas related to circles) and 2 downloaded
worksheets (triangle angle sum, area of an irregular shape).

- **Not redistributed.** NCERT pages carry a "not to be republished" watermark,
  so the images are git-ignored. Place the files in `images/` as
  `ncert_01.png` … `ncert_20.png`; `manifest.json` lists their SHA-256 hashes so
  results can be matched to the exact files.
- **No ground truth.** These are kept separate from the 100 synthetic images and
  are never pooled with them. Only measurements that need no annotation are
  reported (`python -m evaluation.run_real --slice ncert`). Accuracy remains
  NOT MEASURED until a teacher verifies annotations for these pages.
- **Stressors present:** diagonal watermark text over figures, multi-figure
  pages with dense prose, blue/cyan ink, hatched and shaded regions, arcs and
  semicircles, angle arcs with degree labels, small fonts.
