"""Deterministic listing normalization and watch-rule filters."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from .domain import Listing, WatchRule


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    reason: str


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def normalize_listing(listing: Listing) -> Listing:
    """Return a normalized copy without mutating the immutable input value."""
    return listing.model_copy(
        update={
            "title": _normalize_text(listing.title),
            "description": _normalize_text(listing.description),
        }
    )


def evaluate_rule(listing: Listing, rule: WatchRule) -> FilterResult:
    """Evaluate a watch rule in the required deterministic order."""
    include_keywords = tuple(
        keyword
        for raw_keyword in rule.include_keywords
        if (keyword := _normalize_text(raw_keyword))
    )
    if not include_keywords:
        return FilterResult(False, "blank_include_keywords")

    searchable = f"{listing.title} {listing.description}".casefold()
    for raw_keyword in rule.exclude_keywords:
        keyword = _normalize_text(raw_keyword)
        if keyword and keyword.casefold() in searchable:
            return FilterResult(False, f"excluded_keyword:{keyword}")

    if not any(keyword.casefold() in searchable for keyword in include_keywords):
        return FilterResult(False, "missing_include_keyword")

    if rule.max_price_jpy is not None and listing.price_jpy > rule.max_price_jpy:
        return FilterResult(False, "price_above_maximum")

    return FilterResult(True, "accepted")
