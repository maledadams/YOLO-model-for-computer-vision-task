from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageColor, ImageDraw, ImageFont


BOARD_FILES = tuple("abcdefgh")
BOARD_RANKS = tuple("12345678")

PIECE_FAMILY_COLORS: dict[str, str] = {
    "bishop": "#00B894",
    "king": "#FF6B6B",
    "knight": "#4D96FF",
    "pawn": "#F4A261",
    "queen": "#D63384",
    "rook": "#7B2CBF",
}


@dataclass
class DetectionRow:
    label: str
    side: str
    family: str
    confidence: float
    square: str
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["confidence"] = round(self.confidence, 4)
        return row


def family_from_label(label: str) -> str:
    normalized = label.lower().strip()
    return normalized.split("-", 1)[-1]


def side_from_label(label: str) -> str:
    normalized = label.lower().strip()
    return normalized.split("-", 1)[0]


def normalize_squares(values: Iterable[str]) -> list[str]:
    seen = set()
    normalized: list[str] = []
    for value in values:
        square = value.lower().strip()
        if len(square) != 2 or square[0] not in BOARD_FILES or square[1] not in BOARD_RANKS:
            continue
        if square in seen:
            continue
        seen.add(square)
        normalized.append(square)
    return normalized


def parse_square_text(raw_value: str) -> list[str]:
    cleaned = raw_value.replace("\n", ",").replace(" ", ",")
    pieces = [piece for piece in cleaned.split(",") if piece]
    return normalize_squares(pieces)


def square_bounds(
    image_width: int,
    image_height: int,
    square: str,
    orientation: str = "white_bottom",
    board_margin_ratio: float = 0.0,
) -> tuple[float, float, float, float]:
    margin_x = image_width * board_margin_ratio
    margin_y = image_height * board_margin_ratio
    board_left = margin_x
    board_top = margin_y
    board_width = image_width - (margin_x * 2)
    board_height = image_height - (margin_y * 2)
    tile_width = board_width / 8
    tile_height = board_height / 8

    file_index = BOARD_FILES.index(square[0])
    rank_index = BOARD_RANKS.index(square[1])

    if orientation == "white_bottom":
        column = file_index
        row = 7 - rank_index
    else:
        column = 7 - file_index
        row = rank_index

    x1 = board_left + (column * tile_width)
    y1 = board_top + (row * tile_height)
    x2 = x1 + tile_width
    y2 = y1 + tile_height
    return x1, y1, x2, y2


def point_to_square(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    orientation: str = "white_bottom",
    board_margin_ratio: float = 0.0,
) -> str:
    margin_x = image_width * board_margin_ratio
    margin_y = image_height * board_margin_ratio
    board_left = margin_x
    board_top = margin_y
    board_width = image_width - (margin_x * 2)
    board_height = image_height - (margin_y * 2)
    tile_width = board_width / 8
    tile_height = board_height / 8

    clamped_x = min(max(x, board_left), board_left + board_width - 1)
    clamped_y = min(max(y, board_top), board_top + board_height - 1)

    column = int((clamped_x - board_left) / tile_width)
    row = int((clamped_y - board_top) / tile_height)
    column = max(0, min(7, column))
    row = max(0, min(7, row))

    if orientation == "white_bottom":
        file_name = BOARD_FILES[column]
        rank_name = BOARD_RANKS[7 - row]
    else:
        file_name = BOARD_FILES[7 - column]
        rank_name = BOARD_RANKS[row]
    return f"{file_name}{rank_name}"


def build_detection_rows(
    result,
    image_width: int,
    image_height: int,
    orientation: str = "white_bottom",
    board_margin_ratio: float = 0.0,
) -> list[DetectionRow]:
    detections: list[DetectionRow] = []
    names = result.names
    boxes = result.boxes

    if boxes is None:
        return detections

    xyxy_values = boxes.xyxy.cpu().tolist()
    conf_values = boxes.conf.cpu().tolist()
    cls_values = boxes.cls.cpu().tolist()

    for xyxy, confidence, class_id in zip(xyxy_values, conf_values, cls_values):
        x1, y1, x2, y2 = xyxy
        label = names[int(class_id)]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        detections.append(
            DetectionRow(
                label=label,
                side=side_from_label(label),
                family=family_from_label(label),
                confidence=float(confidence),
                square=point_to_square(
                    center_x,
                    center_y,
                    image_width=image_width,
                    image_height=image_height,
                    orientation=orientation,
                    board_margin_ratio=board_margin_ratio,
                ),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )

    return detections


def consolidate_board_detections(
    detections: list[DetectionRow],
    max_pieces: int = 32,
) -> list[DetectionRow]:
    
    by_square: dict[str, DetectionRow] = {}
    for detection in detections:
        existing = by_square.get(detection.square)
        if existing is None or detection.confidence > existing.confidence:
            by_square[detection.square] = detection

    consolidated = sorted(by_square.values(), key=lambda row: row.confidence, reverse=True)
    return consolidated[:max_pieces]


def detection_matches(
    detection: DetectionRow,
    families: set[str],
    sides: set[str],
    squares: set[str],
) -> bool:
    family_ok = not families or detection.family in families
    side_ok = not sides or detection.side in sides
    square_ok = not squares or detection.square in squares
    return family_ok and side_ok and square_ok


def _line_width(image_width: int, image_height: int) -> int:
    return max(3, min(image_width, image_height) // 170)


def _square_outline_width(image_width: int, image_height: int) -> int:
    return max(5, min(image_width, image_height) // 115)


def _rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    red, green, blue = ImageColor.getrgb(hex_color)
    return red, green, blue, alpha


def draw_highlights(
    image: Image.Image,
    detections: list[DetectionRow],
    selected_families: Iterable[str],
    selected_sides: Iterable[str],
    selected_squares: Iterable[str],
    orientation: str = "white_bottom",
    board_margin_ratio: float = 0.0,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    width, height = canvas.size
    families = set(selected_families)
    sides = set(selected_sides)
    squares = set(normalize_squares(selected_squares))
    filtered = [d for d in detections if detection_matches(d, families, sides, squares)]

    for square in normalize_squares(selected_squares):
        x1, y1, x2, y2 = square_bounds(
            width,
            height,
            square,
            orientation=orientation,
            board_margin_ratio=board_margin_ratio,
        )
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 255, 255, 255),
            width=_square_outline_width(width, height),
        )
        text_draw.text((x1 + 8, y1 + 8), square, fill=(255, 255, 255, 255), font=font)

    visible_rows = filtered if (families or sides or squares) else detections
    for detection in visible_rows:
        color = PIECE_FAMILY_COLORS.get(detection.family, "#00A8E8")
        draw.rectangle(
            [detection.x1, detection.y1, detection.x2, detection.y2],
            outline=_rgba(color, 255),
            fill=_rgba(color, 72),
            width=_line_width(width, height),
        )
        label = f"{detection.label} {detection.square} {detection.confidence:.2f}"
        text_x = detection.x1 + 4
        text_y = max(4, detection.y1 - 14)
        text_draw.text((text_x, text_y), label, fill=_rgba(color, 255), font=font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def list_image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)
