#!/usr/bin/env python3
"""Tests the expert decision rules with synthetic vote patterns. Not expert validation.

Thresholds are read from app.config, and each case is built from them, so the tests
stay valid if PH_PHOTO_MIN_VOTES or PH_MAJORITY_SHARE change in .env.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import HEALTHY_CLASS, MAJORITY_SHARE, PHOTO_MIN_VOTES  # noqa: E402
from app.services import expert  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(("PASS  " if ok else "FAIL  ") + f"{name}: got {got!r}, want {want!r}")
    if not ok:
        failures.append(name)


def dominant_votes(n: int, top: str, other: str, top_share: float) -> Counter:
    """n neighbours where `top` holds at least top_share of them. Built from the config values."""
    top_n = max(PHOTO_MIN_VOTES, -(-int(n * top_share * 1000) // 1000))  # ceiling-ish, never below the floor
    top_n = min(n, max(top_n, int(n * top_share) + (1 if n * top_share % 1 else 0)))
    return Counter({top: top_n, other: n - top_n})


N = max(10, PHOTO_MIN_VOTES * 2)  # neighbour pool large enough for any reasonable config

# A clear majority, comfortably above MAJORITY_SHARE
strong = dominant_votes(N, "Little Leaf", "Healthy", min(1.0, MAJORITY_SHARE + 0.2))
# A split vote, with no class reaching MAJORITY_SHARE
split_n = {"Mosaic Virus": 0, "Cercospora Leaf Spot": 0, "Little Leaf": 0, "Healthy": 0}
per = N // 4
split = Counter({k: per for k in split_n})
split["Mosaic Virus"] += N - per * 4

# Photo alone, strong vote -> possible, never likely
r = expert.decide(strong, N, [], False)
check("photo alone never likely", r["verdict"], "possible")

# Photo and description agree -> likely
r = expert.decide(strong, N, ["Little Leaf"], True)
check("photo and text agree -> likely", r["verdict"], "likely")

# Photo and description disagree -> cannot_determine
r = expert.decide(strong, N, ["Mosaic Virus"], True)
check("photo and text disagree -> cannot_determine", r["verdict"], "cannot_determine")

# Split photo vote, no text -> cannot_determine
r = expert.decide(split, N, [], False)
check("split photo, no text -> cannot_determine", r["verdict"], "cannot_determine")

# Too few neighbours to meet PHOTO_MIN_VOTES -> not supported
few = max(0, PHOTO_MIN_VOTES - 1)
r = expert.decide(Counter({"Little Leaf": few}), few, [], False)
check("below PHOTO_MIN_VOTES -> cannot_determine", r["verdict"], "cannot_determine")

# No evidence at all -> cannot_determine
r = expert.decide(Counter(), 0, [], False)
check("no evidence -> cannot_determine", r["verdict"], "cannot_determine")

# Agreeing rival with enough votes -> possible, not likely
r = expert.decide(Counter({"Little Leaf": N - 4, "Mosaic Virus": 4}), N,
                  ["Little Leaf", "Mosaic Virus"], True)
check("agreeing rival -> possible", r["verdict"], "possible")

# Healthy photo -> possible, and Healthy is the only listed candidate
healthy = dominant_votes(N, HEALTHY_CLASS, "Little Leaf", min(1.0, MAJORITY_SHARE + 0.2))
r = expert.decide(healthy, N, [], False)
check("healthy photo -> possible", r["verdict"], "possible")
check("healthy photo lists only Healthy as candidate", r["candidates"], [HEALTHY_CLASS])

print("\nALL EXPERT RULE TESTS PASSED" if not failures else f"\n{len(failures)} FAILED")
sys.exit(1 if failures else 0)