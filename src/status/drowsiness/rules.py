from __future__ import annotations

from typing import Any, Optional


def log_drowsy_episode(logger, user: Optional[Any], duration_sec: float, min_ear: float, frame: Any, severity: str) -> None:
    logger.log_event(
        getattr(user, "user_id", 0),
        "DROWSY",
        duration_sec,
        min_ear,
        frame,
        alert_category="Drowsiness",
        alert_detail="Eyes Closed Too Long",
        severity=severity,
    )


def log_yawn(logger, user: Optional[Any], mar: float, frame: Any, alert_detail: str, severity: str) -> None:
    logger.log_event(
        getattr(user, "user_id", 0),
        "YAWN",
        0.0,
        mar,
        frame,
        alert_category="Drowsiness",
        alert_detail=alert_detail,
        severity=severity,
    )


def log_drowsy_score_on(
    logger,
    user: Optional[Any],
    score: float,
    perclos: float,
    frame: Any,
    severity: str = "Medium",
) -> None:
    logger.log_event(
        getattr(user, "user_id", 0),
        "DROWSY_SCORE_ON",
        0.0,
        float(score),
        frame,
        alert_category="Drowsiness",
        alert_detail=f"Weighted score ON (score={score:.2f}, perclos={perclos:.2f})",
        severity=severity,
    )