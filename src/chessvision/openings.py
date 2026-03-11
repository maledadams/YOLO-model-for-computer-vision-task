from __future__ import annotations

from dataclasses import dataclass

from .core import DetectionRow


@dataclass(frozen=True)
class PieceRule:
    side: str
    family: str
    square: str

    @property
    def label(self) -> str:
        return f"{self.side}-{self.family}"

    def describe(self) -> str:
        return f"{self.label} on {self.square}"


@dataclass(frozen=True)
class OpeningDefinition:
    key: str
    name: str
    moves: str
    description: str
    required: tuple[PieceRule, ...]
    forbidden: tuple[PieceRule, ...] = ()


OPENING_DEFINITIONS: tuple[OpeningDefinition, ...] = (
    OpeningDefinition(
        key="ruy_lopez",
        name="Ruy Lopez",
        moves="1.e4 e5 2.Nf3 Nc6 3.Bb5",
        description="Classic king-pawn opening with White's bishop pinning the c6 knight.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("white", "knight", "f3"),
            PieceRule("white", "bishop", "b5"),
            PieceRule("black", "pawn", "e5"),
            PieceRule("black", "knight", "c6"),
        ),
        forbidden=(PieceRule("white", "bishop", "c4"),),
    ),
    OpeningDefinition(
        key="italian_game",
        name="Italian Game",
        moves="1.e4 e5 2.Nf3 Nc6 3.Bc4",
        description="White develops the bishop to c4 to pressure f7 immediately.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("white", "knight", "f3"),
            PieceRule("white", "bishop", "c4"),
            PieceRule("black", "pawn", "e5"),
            PieceRule("black", "knight", "c6"),
        ),
        forbidden=(PieceRule("white", "bishop", "b5"),),
    ),
    OpeningDefinition(
        key="sicilian_defense",
        name="Sicilian Defense",
        moves="1.e4 c5",
        description="Black answers e4 with the asymmetrical c-pawn strike.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("black", "pawn", "c5"),
        ),
        forbidden=(
            PieceRule("black", "pawn", "e5"),
            PieceRule("black", "pawn", "d5"),
        ),
    ),
    OpeningDefinition(
        key="french_defense",
        name="French Defense",
        moves="1.e4 e6 2.d4 d5",
        description="Black builds a solid center with e6 and d5 against e4.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("white", "pawn", "d4"),
            PieceRule("black", "pawn", "e6"),
            PieceRule("black", "pawn", "d5"),
        ),
        forbidden=(PieceRule("black", "pawn", "c6"),),
    ),
    OpeningDefinition(
        key="caro_kann_defense",
        name="Caro-Kann Defense",
        moves="1.e4 c6 2.d4 d5",
        description="Black supports ...d5 with the c-pawn before contesting the center.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("white", "pawn", "d4"),
            PieceRule("black", "pawn", "c6"),
            PieceRule("black", "pawn", "d5"),
        ),
        forbidden=(PieceRule("black", "pawn", "e6"),),
    ),
    OpeningDefinition(
        key="scandinavian_defense",
        name="Scandinavian Defense",
        moves="1.e4 d5",
        description="Black challenges the e4 pawn immediately with the d-pawn.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("black", "pawn", "d5"),
        ),
        forbidden=(
            PieceRule("black", "pawn", "c5"),
            PieceRule("black", "pawn", "c6"),
            PieceRule("black", "pawn", "e6"),
        ),
    ),
    OpeningDefinition(
        key="pirc_defense",
        name="Pirc Defense",
        moves="1.e4 d6 2.d4 Nf6",
        description="Black uses a flexible setup with d6 and Nf6 instead of an immediate center claim.",
        required=(
            PieceRule("white", "pawn", "e4"),
            PieceRule("white", "pawn", "d4"),
            PieceRule("black", "pawn", "d6"),
            PieceRule("black", "knight", "f6"),
        ),
        forbidden=(
            PieceRule("black", "pawn", "d5"),
            PieceRule("black", "pawn", "c5"),
            PieceRule("black", "pawn", "e5"),
        ),
    ),
    OpeningDefinition(
        key="queens_gambit",
        name="Queen's Gambit",
        moves="1.d4 d5 2.c4",
        description="White offers the c-pawn to undermine Black's d5 center.",
        required=(
            PieceRule("white", "pawn", "d4"),
            PieceRule("white", "pawn", "c4"),
            PieceRule("black", "pawn", "d5"),
        ),
        forbidden=(
            PieceRule("black", "pawn", "c6"),
            PieceRule("black", "pawn", "e6"),
        ),
    ),
    OpeningDefinition(
        key="slav_defense",
        name="Slav Defense",
        moves="1.d4 d5 2.c4 c6",
        description="Black supports the d5 pawn with c6 and keeps the light-squared bishop flexible.",
        required=(
            PieceRule("white", "pawn", "d4"),
            PieceRule("white", "pawn", "c4"),
            PieceRule("black", "pawn", "d5"),
            PieceRule("black", "pawn", "c6"),
        ),
        forbidden=(PieceRule("black", "pawn", "e6"),),
    ),
    OpeningDefinition(
        key="kings_indian_defense",
        name="King's Indian Defense",
        moves="1.d4 Nf6 2.c4 g6",
        description="Black fianchettos the king-side bishop and attacks the center later.",
        required=(
            PieceRule("white", "pawn", "d4"),
            PieceRule("white", "pawn", "c4"),
            PieceRule("black", "knight", "f6"),
            PieceRule("black", "pawn", "g6"),
        ),
        forbidden=(
            PieceRule("black", "pawn", "d5"),
            PieceRule("black", "pawn", "c5"),
        ),
    ),
    OpeningDefinition(
        key="english_opening",
        name="English Opening",
        moves="1.c4",
        description="White opens with the c-pawn to avoid immediate king-pawn and queen-pawn symmetry.",
        required=(PieceRule("white", "pawn", "c4"),),
        forbidden=(
            PieceRule("white", "pawn", "d4"),
            PieceRule("white", "pawn", "e4"),
        ),
    ),
    OpeningDefinition(
        key="london_system",
        name="London System",
        moves="1.d4 and Bf4 with Nf3",
        description="White develops into a dependable setup centered on d4, Nf3, and Bf4.",
        required=(
            PieceRule("white", "pawn", "d4"),
            PieceRule("white", "knight", "f3"),
            PieceRule("white", "bishop", "f4"),
        ),
        forbidden=(PieceRule("white", "pawn", "c4"),),
    ),
)


OPENING_BY_KEY = {opening.key: opening for opening in OPENING_DEFINITIONS}


def detections_to_board_state(detections: list[DetectionRow]) -> dict[str, DetectionRow]:
    board_state: dict[str, DetectionRow] = {}
    for detection in detections:
        existing = board_state.get(detection.square)
        if existing is None or detection.confidence > existing.confidence:
            board_state[detection.square] = detection
    return board_state


def board_has_rule(board_state: dict[str, DetectionRow], rule: PieceRule) -> bool:
    detection = board_state.get(rule.square)
    if detection is None:
        return False
    return detection.side == rule.side and detection.family == rule.family


def match_openings(detections: list[DetectionRow]) -> list[OpeningDefinition]:
    board_state = detections_to_board_state(detections)
    matches: list[OpeningDefinition] = []

    for opening in sorted(OPENING_DEFINITIONS, key=lambda item: len(item.required), reverse=True):
        required_ok = all(board_has_rule(board_state, rule) for rule in opening.required)
        forbidden_ok = all(not board_has_rule(board_state, rule) for rule in opening.forbidden)
        if required_ok and forbidden_ok:
            matches.append(opening)

    return matches


def opening_filter_matches(detections: list[DetectionRow], selected_opening_keys: list[str]) -> list[OpeningDefinition]:
    matches = match_openings(detections)
    if not selected_opening_keys:
        return matches
    selected = set(selected_opening_keys)
    return [opening for opening in matches if opening.key in selected]


def opening_table_rows() -> list[dict[str, str]]:
    rows = []
    for opening in OPENING_DEFINITIONS:
        rows.append(
            {
                "name": opening.name,
                "moves": opening.moves,
                "description": opening.description,
                "required_signature": ", ".join(rule.describe() for rule in opening.required),
            }
        )
    return rows
