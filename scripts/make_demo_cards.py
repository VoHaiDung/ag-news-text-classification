"""Render the demo video's title and end cards as 1920x1080 PNGs.

Two editor-built frames for the demo walkthrough
(reports/presentation/demo_video_script.md, Shots 1 and 12):

  card_01_title.png  — project + team logos, title, team line
  card_12_end.png    — call to action + GitHub URL

Output: reports/presentation/demo_frames/{card_01_title,card_12_end}.png

Run:  python -m scripts.make_demo_cards
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUT = ROOT / "reports" / "presentation" / "demo_frames"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080
BG = (11, 12, 16)          # near-black, matches the Gradio dark theme
WHITE = (245, 245, 247)
GREY = (150, 152, 160)
SIC_BLUE = (90, 130, 255)  # link accent

# Windows core fonts (verified present on this host).
F_BOLD = "C:/Windows/Fonts/arialbd.ttf"
F_SEMI = "C:/Windows/Fonts/seguisb.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def center_text(draw, text, fnt, y, fill, w=W):
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    draw.text(((w - tw) / 2, y), text, font=fnt, fill=fill)
    return box[3] - box[1]


def paste_logos(img: Image.Image, y: int, height: int = 150) -> None:
    """Paste project | team logos side by side, centred, with a divider."""
    logos = []
    for name in ("logo-ag-news-text-classification.png", "logo-aimer-pam.png"):
        p = ASSETS / name
        if p.exists():
            lg = Image.open(p).convert("RGBA")
            scale = height / lg.height
            lg = lg.resize((int(lg.width * scale), height), Image.LANCZOS)
            logos.append(lg)
    if not logos:
        return
    gap = 60
    total = sum(l.width for l in logos) + gap * (len(logos) - 1)
    x = (W - total) // 2
    for i, lg in enumerate(logos):
        img.paste(lg, (x, y), lg)
        x += lg.width + gap


def make_title() -> Path:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    paste_logos(img, y=300, height=150)
    center_text(d, "AG News Text Classification", font(F_BOLD, 96), 540, WHITE)
    center_text(d, "Aimer PAM", font(F_SEMI, 48), 680, GREY)
    center_text(d, "Samsung Innovation Campus — AI Course",
                font(F_SEMI, 34), 760, GREY)
    path = OUT / "card_01_title.png"
    img.save(path)
    return path


def make_end() -> Path:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    paste_logos(img, y=330, height=130)
    center_text(d, "Explore the project", font(F_BOLD, 84), 560, WHITE)
    center_text(d, "github.com/VoHaiDung/ag-news-text-classification",
                font(F_SEMI, 40), 700, SIC_BLUE)
    center_text(d, "Aimer PAM · Vo Hai Dung", font(F_SEMI, 32), 780, GREY)
    path = OUT / "card_12_end.png"
    img.save(path)
    return path


def main() -> None:
    t = make_title()
    e = make_end()
    print(f"wrote {t.relative_to(ROOT)}")
    print(f"wrote {e.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
