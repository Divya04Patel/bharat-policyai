from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def hash_embed_texts(texts: Iterable[str], dim: int = 384) -> np.ndarray:
    """Create deterministic normalized embeddings using a hashing trick.

    This keeps the prototype fully local and avoids heavyweight model installs.
    """
    text_list = list(texts)
    vectors = np.zeros((len(text_list), dim), dtype=np.float32)

    for row, text in enumerate(text_list):
        tokens = _tokenize(text)
        if not tokens:
            continue

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % dim
            sign = -1.0 if digest[0] & 1 else 1.0
            vectors[row, bucket] += sign

        norm = float(np.linalg.norm(vectors[row]))
        if norm > 0:
            vectors[row] /= norm

    return vectors
