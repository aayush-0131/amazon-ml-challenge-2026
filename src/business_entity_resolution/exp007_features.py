"""Additional EXP007 representations; the EXP003 feature contract is untouched."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .features import FEATURE_NAMES as OLD_FEATURE_NAMES, build_feature_row

NORMALIZATION_VERSION = "anyascii-0.3.3-secondary-v1"
SUFFIXES = frozenset("""inc incorporated corporation corp co company ltd limited llc llp plc gmbh ag kg bv nv sa sas sarl srl spa pty pte pvt private public liability enterprises enterprise holdings holding group solutions services technologies technology international intl udyog societe anonyme sro oy ab aps""".split())
SPACE = re.compile(r"[^a-z0-9]+")
SPLIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def normalize(text: str) -> str:
    raw = anyascii(str(text or "")).casefold()
    raw = re.sub(r"https?://|www\.", " ", raw)
    raw = re.sub(r"(?<=[a-z0-9])\.(?:com|org|net|in|fr|co)\b", " ", raw)
    return " ".join(SPACE.sub(" ", SPLIT.sub(" ", raw)).split())


def core_name(text: str) -> str:
    full = normalize(text).split()
    core = full[:]
    while core and core[-1] in SUFFIXES:
        core.pop()
    return " ".join(core or full)


def canonical_number(token: str) -> str:
    return str(int(token))


@dataclass(frozen=True)
class Address:
    text: str
    numbers: frozenset[str]
    primary: str
    postal: frozenset[str]


def address(text: str) -> Address:
    value = normalize(text)
    tokens = value.split()
    numeric = [canonical_number(t) for t in tokens if t.isdigit()]
    # Extract postal-like tokens before digit-letter splitting (e.g. SW1A).
    raw_tokens = re.findall(r"[a-z0-9]+", anyascii(str(text or "")).casefold())
    postal = frozenset((t.lstrip("0") or "0") if t.isdigit() else t for t in raw_tokens
                       if 4 <= len(t) <= 8 and any(c.isdigit() for c in t))
    return Address(value, frozenset(numeric), numeric[0] if numeric else "", postal)


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def grams(value: str) -> frozenset[str]:
    value = value.replace(" ", "")
    return frozenset(value[i:i+3] for i in range(max(0, len(value)-2)))


def ratio(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100 if a and b else 0.0


ENHANCED_FEATURE_NAMES = (
    "trans_name_exact", "trans_name_ratio", "trans_name_token_set", "trans_name_token_sort", "trans_name_jw",
    "core_name_exact", "core_name_token_set", "core_name_token_sort", "core_name_jaccard", "core_name_containment",
    "core_name_char3", "significant_first_token_similarity", "significant_first_token_conflict", "trans_name_length_ratio",
    "legal_suffix_only_difference", "trans_address_ratio", "trans_address_token_set", "trans_address_token_sort",
    "trans_address_jw", "trans_address_char3", "numeric_token_overlap", "numeric_token_containment",
    "numeric_sets_disjoint", "primary_number_exact", "primary_number_mismatch", "primary_number_log_difference",
    "postal_like_agreement", "postal_like_conflict", "core_name_primary_conflict", "high_name_numeric_conflict",
    "left_address_missing_x_name", "right_address_missing_x_name",
)
FEATURE_NAMES = OLD_FEATURE_NAMES + ENHANCED_FEATURE_NAMES


def enhanced_row(left_name: str, right_name: str, left_address: str, right_address: str) -> dict[str, float]:
    ln, rn = normalize(left_name), normalize(right_name)
    lc, rc = core_name(left_name), core_name(right_name)
    la, ra = address(left_address), address(right_address)
    lt, rt = set(lc.split()), set(rc.split())
    first_l, first_r = next(iter(lc.split()), ""), next(iter(rc.split()), "")
    name_sim = ratio(lc, rc)
    primary_conflict = bool(la.primary and ra.primary and la.primary != ra.primary)
    values = (
        float(bool(ln and ln == rn)), ratio(ln, rn), fuzz.token_set_ratio(ln, rn)/100 if ln and rn else 0,
        fuzz.token_sort_ratio(ln, rn)/100 if ln and rn else 0, JaroWinkler.normalized_similarity(ln, rn) if ln and rn else 0,
        float(bool(lc and lc == rc)), fuzz.token_set_ratio(lc, rc)/100 if lc and rc else 0,
        fuzz.token_sort_ratio(lc, rc)/100 if lc and rc else 0, jaccard(lt, rt),
        len(lt & rt)/min(len(lt), len(rt)) if lt and rt else 0, jaccard(grams(lc), grams(rc)),
        ratio(first_l, first_r), float(bool(first_l and first_r and first_l != first_r and ratio(first_l, first_r) < .5)),
        min(len(ln), len(rn))/max(len(ln), len(rn)) if ln and rn else 0,
        float(bool(ln and rn and ln != rn and lc == rc)),
        ratio(la.text, ra.text), fuzz.token_set_ratio(la.text, ra.text)/100 if la.text and ra.text else 0,
        fuzz.token_sort_ratio(la.text, ra.text)/100 if la.text and ra.text else 0,
        JaroWinkler.normalized_similarity(la.text, ra.text) if la.text and ra.text else 0,
        jaccard(grams(la.text), grams(ra.text)), jaccard(la.numbers, ra.numbers),
        len(la.numbers & ra.numbers)/min(len(la.numbers), len(ra.numbers)) if la.numbers and ra.numbers else 0,
        float(bool(la.numbers and ra.numbers and not la.numbers & ra.numbers)),
        float(bool(la.primary and la.primary == ra.primary)), float(primary_conflict),
        math.log1p(abs(int(la.primary)-int(ra.primary))) if la.primary and ra.primary else 0,
        float(bool(la.postal and ra.postal and la.postal & ra.postal)),
        float(bool(la.postal and ra.postal and not la.postal & ra.postal)),
        float(bool(lc and lc == rc and primary_conflict)), float(name_sim >= .9 and primary_conflict),
        float(not la.text) * name_sim, float(not ra.text) * name_sim,
    )
    return dict(zip(ENHANCED_FEATURE_NAMES, map(float, values), strict=True))


def feature_row(source, candidate) -> dict[str, float]:
    old = build_feature_row(source, candidate)
    extra = enhanced_row(source.business_name, candidate.candidate_name, source.business_address, candidate.candidate_address)
    result = {**old, **extra}
    if tuple(result) != FEATURE_NAMES:
        raise AssertionError("EXP007 feature order mismatch")
    return result
