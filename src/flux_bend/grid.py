"""Image grid assembly using Pillow."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont


HEADER_HEIGHT = 40
TITLE_HEIGHT = 60
LABEL_WIDTH = 120


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font, falling back to default if no TTF is available."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def make_grid(
    images: list[Image.Image],
    row_labels: list[str],
    col_labels: list[str],
    row_param_name: str | None = None,
    col_param_name: str | None = None,
    title: str = "",
    cell_size: tuple[int, int] = (512, 512),
) -> Image.Image:
    """Assemble images into a labeled grid.

    Args:
        images: Flat list of images, row-major order. len = nrows * ncols.
        row_labels: Labels for each row.
        col_labels: Labels for each column.
        row_param_name: Name of the row parameter (shown in row header).
        col_param_name: Name of the column parameter (shown in col header).
        title: Title string for the top bar.
        cell_size: (width, height) to resize each cell image.
    """
    cell_w, cell_h = cell_size
    n_cols = max(len(col_labels), 1)
    n_rows = max(len(row_labels), 1)

    has_row_labels = bool(row_labels and any(row_labels))
    has_col_labels = bool(col_labels and any(col_labels))
    has_title = bool(title)

    left_margin = LABEL_WIDTH if has_row_labels else 0
    top_margin = (TITLE_HEIGHT if has_title else 0) + (HEADER_HEIGHT if has_col_labels else 0)

    grid_w = left_margin + n_cols * cell_w
    grid_h = top_margin + n_rows * cell_h

    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = _get_font(14)
    title_font = _get_font(12)

    # Title bar
    if has_title:
        truncated = title[:100] + "..." if len(title) > 100 else title
        draw.text((10, 10), truncated, fill=(0, 0, 0), font=title_font)

    # Column headers
    if has_col_labels:
        y = TITLE_HEIGHT if has_title else 0
        for c, label in enumerate(col_labels):
            x = left_margin + c * cell_w + cell_w // 2
            header = f"{col_param_name}={label}" if col_param_name else label
            bbox = draw.textbbox((0, 0), header, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x - text_w // 2, y + 10), header, fill=(0, 0, 0), font=font)

    # Row headers
    if has_row_labels:
        for r, label in enumerate(row_labels):
            y = top_margin + r * cell_h + cell_h // 2
            header = f"{row_param_name}={label}" if row_param_name else label
            bbox = draw.textbbox((0, 0), header, font=font)
            text_h = bbox[3] - bbox[1]
            draw.text((5, y - text_h // 2), header, fill=(0, 0, 0), font=font)

    # Place images
    for idx, img in enumerate(images):
        r = idx // n_cols
        c = idx % n_cols
        resized = img.resize(cell_size, Image.Resampling.LANCZOS)
        x = left_margin + c * cell_w
        y = top_margin + r * cell_h
        grid.paste(resized, (x, y))

    return grid
