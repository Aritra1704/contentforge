"""Static checks for the fixed golden-set evaluation briefs."""

from __future__ import annotations

import json
from pathlib import Path


def test_golden_set_briefs_cover_core_mvp_scenarios() -> None:
    """The golden set should stay broad enough to cover the current MVP and known failures."""

    path = Path(__file__).parent / "data" / "golden_set_briefs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload) >= 20
    ids = {item["id"] for item in payload}
    assert len(ids) == len(payload)
    assert any(item["theme_name"] == "Valentine Special" for item in payload)
    assert any(item["theme_name"] == "Holi Week" for item in payload)
    assert any(item["theme_name"] == "Ramadan Month" for item in payload)
    assert any(item["theme_name"] == "Diwali Week" for item in payload)
    assert any(item["theme_name"] == "Netflix and Chill" for item in payload)
    assert any(item["id"].startswith("failure_") for item in payload)
