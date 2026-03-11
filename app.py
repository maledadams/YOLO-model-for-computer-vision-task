from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.chessvision import (
    BOARD_FILES,
    BOARD_RANKS,
    OPENING_DEFINITIONS,
    PIECE_FAMILY_COLORS,
    build_detection_rows,
    consolidate_board_detections,
    draw_highlights,
    match_openings,
    opening_table_rows,
    parse_square_text,
)
from src.chessvision.core import detection_matches, list_image_files


st.set_page_config(page_title="Chess Piece Highlighter", layout="wide")


ALL_SQUARES = [f"{file_name}{rank_name}" for rank_name in reversed(BOARD_RANKS) for file_name in BOARD_FILES]
ALL_FAMILIES = sorted(PIECE_FAMILY_COLORS)
ALL_SIDES = ["white", "black"]
OPENING_OPTIONS = {opening.key: opening for opening in OPENING_DEFINITIONS}


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def read_uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def read_local_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def infer_image(model: YOLO, image: Image.Image, conf: float, orientation: str, board_margin_ratio: float):
    result = model.predict(image, conf=conf, iou=0.45, max_det=40, verbose=False)[0]
    raw_detections = build_detection_rows(
        result,
        image_width=image.width,
        image_height=image.height,
        orientation=orientation,
        board_margin_ratio=board_margin_ratio,
    )
    detections = consolidate_board_detections(raw_detections)
    return result, detections


def selected_square_values(manual_input: str, selected_from_ui: list[str]) -> list[str]:
    merged = list(selected_from_ui)
    merged.extend(parse_square_text(manual_input))
    seen = set()
    normalized = []
    for square in merged:
        if square not in seen:
            seen.add(square)
            normalized.append(square)
    return normalized


def detection_table(detections):
    if not detections:
        return pd.DataFrame(columns=["label", "side", "family", "square", "confidence"])
    return pd.DataFrame([row.to_dict() for row in detections])


def opening_table(openings):
    if not openings:
        return pd.DataFrame(columns=["opening", "moves", "description"])
    return pd.DataFrame(
        [{"opening": opening.name, "moves": opening.moves, "description": opening.description} for opening in openings]
    )


def render_opening_summary(matched_openings, selected_openings) -> None:
    if matched_openings:
        opening_names = ", ".join(opening.name for opening in matched_openings)
        st.success(f"Opening identified: {opening_names}")
    elif selected_openings:
        selected_names = ", ".join(OPENING_OPTIONS[key].name for key in selected_openings)
        st.warning(f"No match for selected opening filter: {selected_names}")
    else:
        st.info("No built-in opening pattern was identified in this image.")


def sidebar_filters():
    st.sidebar.title("Controls")
    weights_path = st.sidebar.text_input(
        "YOLO weights",
        value="runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt",
        help="Use your trained best.pt, or any compatible Ultralytics checkpoint.",
    )
    confidence = st.sidebar.slider("Confidence", 0.05, 0.95, 0.40, 0.05)
    orientation = st.sidebar.selectbox(
        "Board orientation",
        options=["white_bottom", "black_bottom"],
        format_func=lambda value: "White pieces at bottom" if value == "white_bottom" else "Black pieces at bottom",
    )
    board_margin_percent = st.sidebar.slider(
        "Board margin (%)",
        min_value=0,
        max_value=20,
        value=0,
        help="Increase this when the board does not fully occupy the image frame.",
    )
    selected_families = st.sidebar.multiselect("Piece type filter", ALL_FAMILIES)
    selected_sides = st.sidebar.multiselect("Piece color filter", ALL_SIDES)
    square_quick_pick = st.sidebar.multiselect("Square filter", ALL_SQUARES)
    square_text = st.sidebar.text_input("Square search", value="", help="Examples: e4 or a2, b2, c2")
    selected_openings = st.sidebar.multiselect(
        "Opening filter",
        options=list(OPENING_OPTIONS),
        format_func=lambda key: OPENING_OPTIONS[key].name,
        help="This is logic-based filtering only. It does not add any highlights.",
    )
    return {
        "weights_path": weights_path,
        "confidence": confidence,
        "orientation": orientation,
        "board_margin_ratio": board_margin_percent / 100,
        "selected_families": selected_families,
        "selected_sides": selected_sides,
        "selected_squares": selected_square_values(square_text, square_quick_pick),
        "selected_openings": selected_openings,
    }


def filter_matches(detections, selected_families, selected_sides, selected_squares):
    families = set(selected_families)
    sides = set(selected_sides)
    squares = set(selected_squares)
    return [row for row in detections if detection_matches(row, families, sides, squares)]


def opening_matches_filter(detections, selected_openings):
    matches = match_openings(detections)
    if not selected_openings:
        return matches
    selected = set(selected_openings)
    return [opening for opening in matches if opening.key in selected]


def render_image_result(title: str, image: Image.Image, detections, filters: dict[str, object]) -> None:
    filtered = filter_matches(
        detections,
        filters["selected_families"],
        filters["selected_sides"],
        filters["selected_squares"],
    )
    matched_openings = opening_matches_filter(detections, filters["selected_openings"])
    highlighted = draw_highlights(
        image,
        detections=detections,
        selected_families=filters["selected_families"],
        selected_sides=filters["selected_sides"],
        selected_squares=filters["selected_squares"],
        orientation=filters["orientation"],
        board_margin_ratio=filters["board_margin_ratio"],
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader(title)
        st.image(highlighted, use_container_width=True)
    with right:
        st.subheader("Matching detections")
        st.dataframe(detection_table(filtered), use_container_width=True, hide_index=True)
        if filters["selected_squares"] and not filtered:
            st.caption("Selected squares stay outlined even when no matching piece is present.")
        st.subheader("Opening matches")
        st.dataframe(opening_table(matched_openings), use_container_width=True, hide_index=True)
        if filters["selected_openings"]:
            st.caption("Opening filters change search results only. They do not create highlights.")


def single_image_tab(model: YOLO, filters: dict[str, object]) -> None:
    source_mode = st.radio("Image source", options=["Upload image", "Local file path"], horizontal=True)

    image = None
    image_label = None

    if source_mode == "Upload image":
        uploaded_file = st.file_uploader("Upload a chessboard image", type=["jpg", "jpeg", "png", "webp"])
        if not uploaded_file:
            st.info("Upload one image to run piece detection, highlighting, and opening identification.")
            return
        image = read_uploaded_image(uploaded_file)
        image_label = uploaded_file.name
    else:
        local_path = st.text_input("Local image path", value="")
        if not local_path:
            st.info("Enter a local image path to run piece detection, highlighting, and opening identification.")
            return
        image_path = Path(local_path)
        if not image_path.exists():
            st.warning("That image path does not exist.")
            return
        image = read_local_image(str(image_path))
        image_label = str(image_path)

    _, detections = infer_image(
        model,
        image,
        conf=filters["confidence"],
        orientation=filters["orientation"],
        board_margin_ratio=filters["board_margin_ratio"],
    )
    matched_openings = opening_matches_filter(detections, filters["selected_openings"])
    render_opening_summary(matched_openings, filters["selected_openings"])
    render_image_result(image_label, image, detections, filters)


def folder_search_tab(model: YOLO, filters: dict[str, object]) -> None:
    default_folder = Path("data/chess-pieces/images/val")
    folder_input = st.text_input("Image folder", value=str(default_folder))
    max_images = st.slider("Images to scan", min_value=1, max_value=200, value=24)

    folder_path = Path(folder_input)
    if not folder_path.exists():
        st.warning("Folder does not exist yet. Run dataset preparation first, or point this to a local image folder.")
        return

    image_paths = list_image_files(folder_path)[:max_images]
    if not image_paths:
        st.warning("No supported images were found in that folder.")
        return

    matches = []
    progress = st.progress(0.0)

    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        _, detections = infer_image(
            model,
            image,
            conf=filters["confidence"],
            orientation=filters["orientation"],
            board_margin_ratio=filters["board_margin_ratio"],
        )
        filtered = filter_matches(
            detections,
            filters["selected_families"],
            filters["selected_sides"],
            filters["selected_squares"],
        )
        matched_openings = opening_matches_filter(detections, filters["selected_openings"])

        piece_filter_active = any([filters["selected_families"], filters["selected_sides"], filters["selected_squares"]])
        piece_filter_ok = bool(filtered) or not piece_filter_active
        opening_filter_ok = bool(matched_openings) or not filters["selected_openings"]

        if piece_filter_ok and opening_filter_ok:
            matches.append((image_path, image, detections, filtered, matched_openings))
        progress.progress(index / len(image_paths))

    st.write(f"Found {len(matches)} matching image(s) out of {len(image_paths)} scanned.")
    for image_path, image, detections, filtered, matched_openings in matches:
        with st.expander(image_path.name, expanded=False):
            render_image_result(str(image_path), image, detections, filters)
            if filtered:
                st.caption(
                    "Matched squares: "
                    + ", ".join(sorted({row.square for row in filtered}))
                )
            if matched_openings:
                st.caption(
                    "Matched openings: "
                    + ", ".join(opening.name for opening in matched_openings)
                )


def main() -> None:
    st.title("Chess Piece Highlighter")
    st.write(
        "Detect pieces in an existing board image, then highlight only the piece types, piece colors, "
        "and board positions you care about."
    )

    filters = sidebar_filters()
    try:
        model = load_model(filters["weights_path"])
    except Exception as exc:
        st.error(f"Could not load YOLO weights: {exc}")
        st.stop()

    with st.expander("Highlight colors", expanded=False):
        color_rows = pd.DataFrame(
            [{"piece_type": family, "highlight_color": color} for family, color in PIECE_FAMILY_COLORS.items()]
        )
        st.dataframe(color_rows, use_container_width=True, hide_index=True)

    with st.expander("Opening filters", expanded=False):
        st.write("These filters are rule-based board-state checks. They do not add any highlights to the image.")
        st.dataframe(pd.DataFrame(opening_table_rows()), use_container_width=True, hide_index=True)

    single_tab, search_tab = st.tabs(["Single image", "Search folder"])
    with single_tab:
        single_image_tab(model, filters)
    with search_tab:
        folder_search_tab(model, filters)


if __name__ == "__main__":
    main()
