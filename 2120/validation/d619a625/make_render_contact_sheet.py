"""Assemble actual render captures without changing their pixel values."""

import argparse
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    shapes = ("sphere", "capsule", "cylinder", "cone", "arrow")
    scenarios = (
        ("render-cpu-visible", "CPU; CUDA visible"),
        ("render-cpu-hidden", "CPU; CUDA hidden"),
        ("render-cpu-ambient", "CPU; ambient CUDA context"),
        ("render-cuda", "CUDA native interoperability"),
    )
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 21)
    sheet = Image.new("RGB", (384 * 5, 80 + 300 * 4), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 10), "GH-2120: CPU/CUDA buffers; all drawing uses NVIDIA EGL/OpenGL", font=font, fill="black")
    for column, shape in enumerate(shapes):
        draw.text((column * 384 + 15, 45), shape, font=font, fill="black")
    for row, (name, label) in enumerate(scenarios):
        y = 80 + row * 300
        draw.text((15, y), label, font=font, fill="black")
        for column, shape in enumerate(shapes):
            image = Image.open(args.root / name / f"{shape}-cached-0-frame-0-segments-32.png").convert("RGB")
            if image.size != (384, 256):
                raise ValueError("Unexpected render capture size")
            sheet.paste(image, (column * 384, y + 35))
    sheet.save(args.root / "render-comparison.png")


if __name__ == "__main__":
    main()
