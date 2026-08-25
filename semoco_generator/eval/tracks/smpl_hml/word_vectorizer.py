"""HumanML3D ``WordVectorizer`` (GloVe / our_vab) for ``text_mot_match``.

Vendored from EricGuo5513/text-to-motion ``utils/word_vectorizer.py``.
Expects three files under ``meta_root`` with prefix ``our_vab`` by default::

    our_vab_data.npy
    our_vab_words.pkl
    our_vab_idx.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

POS_enumerator = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

Loc_list = (
    "left", "right", "clockwise", "counterclockwise", "anticlockwise", "forward",
    "back", "backward", "up", "down", "straight", "curve",
)
Body_list = (
    "arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg", "waist",
    "eye", "knee", "shoulder", "thigh",
)
Obj_List = (
    "stair", "dumbbell", "chair", "window", "floor", "car", "ball", "handrail",
    "baseball", "basketball",
)
Act_list = (
    "walk", "run", "swing", "pick", "bring", "kick", "put", "squat", "throw",
    "hop", "dance", "jump", "turn", "stumble", "stop", "sit", "lift", "lower",
    "raise", "wash", "stand", "kneel", "stroll", "rub", "bend", "balance",
    "flap", "jog", "shuffle", "lean", "rotate", "spin", "spread", "climb",
)
Desc_list = (
    "slowly", "carefully", "fast", "careful", "slow", "quickly", "happy",
    "angry", "sad", "happily", "angrily", "sadly",
)
VIP_dict = {
    "Loc_VIP": Loc_list,
    "Body_VIP": Body_list,
    "Obj_VIP": Obj_List,
    "Act_VIP": Act_list,
    "Desc_VIP": Desc_list,
}


class WordVectorizer:
    def __init__(self, meta_root: str | Path, prefix: str = "our_vab") -> None:
        root = Path(meta_root)
        vectors = np.load(root / f"{prefix}_data.npy")
        with open(root / f"{prefix}_words.pkl", "rb") as f:
            words = pickle.load(f)
        with open(root / f"{prefix}_idx.pkl", "rb") as f:
            word2idx = pickle.load(f)
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos: str) -> np.ndarray:
        pos_vec = np.zeros(len(POS_enumerator), dtype=np.float32)
        key = pos if pos in POS_enumerator else "OTHER"
        pos_vec[POS_enumerator[key]] = 1.0
        return pos_vec

    def __len__(self) -> int:
        return len(self.word2vec)

    def __getitem__(self, item: str):
        word, pos = item.split("/")
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            pos_vec = self._get_pos_ohot(vip_pos if vip_pos is not None else pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._get_pos_ohot("OTHER")
        return word_vec, pos_vec


def resolve_glove_root(candidates: list[str | Path] | None = None) -> Path | None:
    """Return the first existing GloVe/our_vab root, or ``None``."""
    from ....paths import glove_root

    defaults = [
        glove_root(),
    ]
    paths = [Path(p) for p in (candidates or [])] + defaults
    for root in paths:
        if (
            (root / "our_vab_data.npy").is_file()
            and (root / "our_vab_words.pkl").is_file()
            and (root / "our_vab_idx.pkl").is_file()
        ):
            return root
    return None


def load_word_vectorizer(
    glove_root: str | Path | None = None,
    *,
    prefix: str = "our_vab",
) -> WordVectorizer:
    root = Path(glove_root) if glove_root else resolve_glove_root()
    if root is None:
        raise FileNotFoundError(
            "HumanML WordVectorizer assets not found. Expected "
            "{our_vab_data.npy, our_vab_words.pkl, our_vab_idx.pkl} under "
            "<data-root>/glove or --glove-root."
        )
    return WordVectorizer(root, prefix=prefix)


__all__ = [
    "POS_enumerator",
    "WordVectorizer",
    "load_word_vectorizer",
    "resolve_glove_root",
]
