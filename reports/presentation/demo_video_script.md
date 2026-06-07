# Demo video script — static-frame product walkthrough

Format: **~35 seconds** (fast-cut), no spoken narration, **background music
only**. Every shot is a **still screenshot** of the live Gradio UI (no
live cursor movement, no typing, no waiting on CPU inference). The
narrative is carried by short on-screen text overlays and gentle motion
(Ken Burns zoom / pan) added in the editor, following the tone of
contemporary technology-company product launches (Apple keynote, Google
I/O, Microsoft Build).

Why static frames: the classifier runs on CPU, so live SHAP and
long-document inference are slow; capturing finished screenshots keeps
the pace fast, hides the cursor, and removes any risk of a mistyped
input on camera. The editor and the music supply the motion — not the
screen recorder.

Recording target: 1920 x 1080, 30 fps, H.264 video + AAC stereo audio
for the background music track.

## 1. Music selection

Pick a single royalty-free instrumental track around 95 - 110 BPM at
moderate volume (-18 LUFS integrated). The track should be:

- Cinematic, lo-fi piano or ambient electronic - never a vocal track
  or sound-effect heavy.
- Quiet enough that every text overlay still reads clearly.
- Looped or trimmed so the final cut fades cleanly to silence on the
  end card.

Suggested sources: YouTube Audio Library (Creative Commons / no
attribution), Pixabay Music, Free Music Archive. Always credit the
artist in the video description even if attribution is not required.

## 2. Assets to capture first (nine screenshots)

Capture every screenshot once, at leisure, from the live UI at
`http://localhost:7861` before opening the editor. Use **Win + Shift +
S**, save all PNGs into `reports/presentation/demo_frames/`. Browser at
100 % zoom (Ctrl+0), full-screen (F11) so no tab bar or address bar is
visible.

| ID | Screenshot to capture                                                                 | Used in |
|----|---------------------------------------------------------------------------------------|---------|
| A  | App home, title + both logos visible (hero frame)                                       | Shot 4  |
| B  | Classify result for the **English** input — model picker shows ModernBERT-large (EN), result Sci/Tech | Shot 5 |
| C  | Classify result for the **Vietnamese** input — model picker auto-switches to XLM-R-large vi+BT, result Sports | Shot 6 |
| D  | Classify result for the **French** input — model picker auto-switches to XLM-R-large fr-only, result Business | Shot 7 |
| E  | Long-document result, with the **Mode:** line reporting the sliding-window path         | Shot 8  |
| F  | Tight crop of the **Class probabilities** panel (cropped from B in the editor)          | Shot 9  |
| G  | **Explain** tab, finished SHAP heat-map with red / blue token weights                  | Shot 10 |
| H  | The **Model comparison** table, three bold rows (one per language) in view             | Shot 11 |

The **model picker is visible** in every Classify screenshot and the
checkpoint name auto-switches with the input language (ModernBERT-large
→ XLM-R-large vi+BT → XLM-R-large fr-only). Screenshots B, C and D
therefore carry the auto-routing story on their own; no separate
"routing crop" frame is needed.

Sample inputs to paste while capturing the screenshots:

- **English (A→B):** "NASA Perseverance rover discovers ancient microbial life on Mars surface."
- **Vietnamese (C):** "Đội tuyển Việt Nam thắng Thái Lan 2-0 tại chung kết AFF Cup."
- **French (D):** "La Banque centrale européenne relève ses taux directeurs de 25 points de base."
- **Long document (E):** paste a long article that exceeds the active
  encoder's native token limit so the **Mode:** line flips from "single
  forward pass" to "sliding-window". Easiest trigger: keep a 512-token
  encoder active (Vietnamese or French) and paste a 700+ word article;
  ModernBERT (8190 tokens) needs a much longer text to switch.
- **SHAP (G):** "Federal Reserve cuts interest rates by 25 basis points to combat inflation."

> Confidence numbers (read off the captured UI, use these in overlays):
> English **Sci/Tech 90 %**, Vietnamese **Sports 82 %**, French
> **Business 90 %**. The predicted class is fixed by the model; the
> percentage can drift between runs, so always match the overlay to the
> number actually shown in the captured screenshot.

## 3. Shot list (static frames, fast cut, ~36 s)

Three acts. **Act I (00:00 - 00:10.6)** is an editor-built hook (no UI).
**Act II (00:10.6 - 00:33.6)** is the product walkthrough, one still
frame per capability. **Act III (00:33.6 - 00:35.6)** is the closing
card. The cut is deliberately fast — each capability gets ~2-3 s, so the
overlays are short and punchy. The effect column gives the CapCut
animation to apply per clip (Vietnamese CapCut names in parentheses).

| Shot | Time            | Duration | Source                    | CapCut effect                         | Text overlay                                                          |
|------|-----------------|---------:|---------------------------|---------------------------------------|----------------------------------------------------------------------|
| 1    | 00:00 - 00:02   | 2 s      | `card_01_title.png`       | Cardboard-hole reveal (lỗ bìa cứng)   | *(text baked into card)*                                              |
| 2    | 00:02 - 00:05.6 | 4 × 0.9 s| 4 newspaper images        | Zoom 1 (thu phóng 1) on each          | **Have you ever wondered** *(one word per image: Have / you ever / wondered ...)* |
| 3    | 00:05.6 - 00:10.6 | 5 s    | 1 headline image          | Zoom 1 (thu phóng 1)                   | **what topic an article belongs to,<br/>before you read it?**         |
| 4    | 00:10.6 - 00:13.6 | 3 s    | **Screenshot A**          | Zoom 1 (thu phóng 1)                   | **We built exactly what you need**                                   |
| 5    | 00:13.6 - 00:15.6 | 2 s    | **Screenshot B**          | Zoom 1 (thu phóng 1)                   | **On English news** · Sci/Tech 90 %                                  |
| 6    | 00:15.6 - 00:17.6 | 2 s    | **Screenshot C**          | Zoom 1 (thu phóng 1)                   | **On Vietnamese news** · Sports 82 %                                 |
| 7    | 00:17.6 - 00:19.6 | 2 s    | **Screenshot D**          | Zoom 1 (thu phóng 1)                   | **Or even on French** · Business 90 %                                |
| 8    | 00:19.6 - 00:22.6 | 3 s    | **Screenshot E**          | Stretch & distort (kéo dài và biến dạng) | **Short or long. It makes no difference**                          |
| 9    | 00:22.6 - 00:25.6 | 3 s    | **Screenshot F** (softmax crop) | Zoom 1 (thu phóng 1)             | **Every prediction, in full detail**                                |
| 10   | 00:25.6 - 00:28.6 | 3 s    | **Screenshot G** (SHAP)   | Bounce 2 (nảy 2)                       | **And see why**<br/>Red pushes toward the class. Blue pushes away.   |
| 11   | 00:28.6 - 00:33.6 | 5 s    | **Screenshot H** (table)  | Zoom 1 (thu phóng 1)                   | **Twelve models<br/>Three languages<br/>One interface**             |
| 12   | 00:33.6 - 00:35.6 | 2 s    | `card_12_end.png`         | Fade out 0.9 s (mờ dần)                | *(text baked into card)*                                             |

Total runtime: ~35.6 seconds. The auto-routing message rides on Shots
5-7 (the model picker visibly changes name with each language), so no
separate routing shot is needed. Because the cut is fast, keep every
overlay to one short line and let it appear with the clip (no 4-second
minimum at this pace).

## 4. Editing each static frame

- **Per-clip effect (CapCut → Animation tab):** apply the effect named in
  the shot-list "CapCut effect" column to each clip, not a single global
  motion. Most clips use **Zoom 1 (thu phóng 1)**; Shot 1 uses the
  **cardboard-hole reveal (lỗ bìa cứng)**, Shot 8 uses **Stretch &
  distort (kéo dài và biến dạng)**, Shot 10 uses **Bounce 2 (nảy 2)**,
  and Shot 12 ends on a **0.9 s fade-out (mờ dần)**.
- **Transitions:** short cross-fade (~0.2 s) between clips, or hard cuts —
  at this fast pace hard cuts on the beat also work well.
- **Overlay timing:** the cut is fast (2-3 s per capability), so keep
  each overlay to one short line and let it appear with the clip; do not
  enforce a 4-second hold.
- **Shot 2 word reveal:** the four newspaper images carry the phrase one
  fragment at a time ("Have" / "you ever" / "wondered" across the four
  0.9 s images), so the sentence builds as the montage flicks past.
- **Crops:** Shot 9 (softmax) is a crop of Screenshot B — crop in CapCut
  to the Predicted-class + Class-probabilities panels rather than
  recapturing; crop (not scale) keeps it sharp.
- **Highlight (optional):** on Shot 10, draw a soft rectangle or arrow
  over one strongly-weighted red token to direct the eye.

## 5. Typography and overlay style

- Single font family for every overlay (Inter or Segoe UI).
- White text on `rgba(0, 0, 0, 0.55)` rounded background.
- 36 px headings / 24 px body / 18 px sub-captions.
- Bottom-centre or lower-third placement so overlays never cover the
  predicted-class panel or the SHAP tokens.

## 6. Audio

- Fade music in over ~1 s on the title card, fade out over 0.9 s on the
  end card (matching Shot 12's fade-out).
- Keep the music at -18 LUFS so overlays stay the visual focus.
- No narration, no UI click sounds, no system sounds — one music stem
  only.

## 7. Pre-capture setup

1. Start the demo:

   ```bash
   python -m src.deployment.gradio_app
   ```

2. Open `http://localhost:7861`, press **F11** (full screen, no browser
   chrome), set zoom to 100 % (Ctrl+0).
3. Capture screenshots A-H per Section 2, pasting the matching sample
   input for each.
4. Pre-render the title card (Shot 1) and the end card (Shot 12) as
   1920 x 1080 PNGs with the project logo and the GitHub URL.
5. Source the newspaper-montage clips/images for Shots 2-3 (royalty-free
   stock) and drop everything into the editor with the music on a
   dedicated audio bus.

## 8. Post-production notes

- Export captions as an SRT file even though the video has no spoken
  narration; the SRT duplicates each text overlay so accessibility and
  auto-captioning pipelines do not generate noise.
- Credit the music track in the video description with the artist name
  and licence (for example "*Music: Hopeful Cinematic by AShamaluevMusic,
  free for use with credit*").
- Upload as **Unlisted** if the SIC submission portal asks for a link;
  do not publish publicly until the SIC review has accepted the report.
