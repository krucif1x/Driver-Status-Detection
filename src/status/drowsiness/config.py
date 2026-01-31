from __future__ import annotations

from typing import Any, Dict

from src.utils.config.yaml_loader import load_yaml_section
from src.utils.config.parsing import as_float, as_int, get_section, sec_to_frames


def load_drowsiness_config(path: str, fps: float) -> Dict[str, Any]:
    root = load_yaml_section(path, "detectors.drowsiness")

    episode = get_section(root, "episode")
    ear = get_section(root, "ear")
    blink = get_section(root, "blink")
    yawn = get_section(root, "yawn")
    sup = get_section(root, "expression_suppression")
    head_pose = get_section(root, "head_pose")

    perclos = get_section(root, "perclos")
    score = get_section(root, "score")
    weights = get_section(score, "weights")

    sev_root = get_section(root, "severity")
    sev = get_section(sev_root, "drowsiness")

    # EAR thresholds
    ear_low = as_float(ear.get("low_threshold"), 0.22)
    ear_high = as_float(ear.get("high_threshold"), 0.26)
    ear_high_ratio = (ear_high / ear_low) if ear_low > 0 else 1.2

    # PERCLOS
    perclos_window_sec = as_float(perclos.get("window_sec"), 30.0)

    # Score timing
    hold_frames = sec_to_frames(score.get("hold_sec"), fps, 0.5)
    release_frames = sec_to_frames(score.get("release_sec"), fps, 1.0)
    hard_close_frames = sec_to_frames(score.get("hard_close_sec"), fps, 2.0)

    # Expression suppression:
    # Your YAML uses explicit frames: smile_frames / laugh_frames
    # Keep backward compatibility with *_sec if present.
    smile_frames = as_int(sup.get("smile_frames"), -1)
    laugh_frames = as_int(sup.get("laugh_frames"), -1)
    if smile_frames < 0:
        smile_frames = int(sec_to_frames(sup.get("smile_suppress_sec"), fps, 0.5))
    if laugh_frames < 0:
        laugh_frames = int(sec_to_frames(sup.get("laugh_suppress_sec"), fps, 0.67))

    return {
        "episode": {
            "start_frames": sec_to_frames(episode.get("start_threshold_sec"), fps, 0.5),
            "end_frames": sec_to_frames(episode.get("end_grace_sec"), fps, 1.5),
            "min_episode_sec": as_float(episode.get("min_episode_sec"), 2.0),
            "drop_start_multiplier": as_float(episode.get("drop_start_multiplier"), 0.6),
        },
        "ear": {
            "low": ear_low,
            "high": ear_high,
            "high_ratio": float(ear_high_ratio),
            "drop": as_float(ear.get("drop_threshold"), 0.10),
            "drop_window_frames": as_int(ear.get("drop_window_frames"), 15),
            "history_frames": sec_to_frames(ear.get("history_sec"), fps, 1.0),
            # Not in your YAML; detector uses it -> default safely
            "ema_alpha": as_float(ear.get("ema_alpha"), 0.35),
        },
        "perclos": {
            "window_frames": sec_to_frames(perclos_window_sec, fps, 30.0),
            "threshold": as_float(perclos.get("threshold"), 0.25),
        },
        "head_pose": {
            "pitch_abs_threshold_deg": as_float(head_pose.get("pitch_abs_threshold_deg"), 18.0),
        },
        "score": {
            "on_threshold": as_float(score.get("on_threshold"), 0.65),
            "off_threshold": as_float(score.get("off_threshold"), 0.45),
            "hold_frames": int(hold_frames),
            "release_frames": int(release_frames),
            "hard_close_frames": int(hard_close_frames),
            "yawn_saturate_count": as_int(score.get("yawn_saturate_count"), 3),
            "weights": {
                "perclos": as_float(weights.get("perclos"), 0.55),
                "eyes_closed": as_float(weights.get("eyes_closed"), 0.25),
                "yawn": as_float(weights.get("yawn"), 0.15),
                "pitch": as_float(weights.get("pitch"), 0.05),
            },
        },
        "blink": {
            "min_closed_frames": as_int(blink.get("min_closed_frames"), 1),
            "max_closed_frames": as_int(blink.get("max_closed_frames"), 10),
        },
        "yawn": {
            "thresh_frames": sec_to_frames(yawn.get("threshold_sec"), fps, 0.27),
            "cooldown_frames": sec_to_frames(yawn.get("cooldown_sec"), fps, 2.0),
            "hand_cover_distance_norm": as_float(yawn.get("hand_cover_distance_norm"), 0.15),
            "frequency_window_sec": as_float(yawn.get("frequency_window_sec"), 120.0),
            "high_frequency_count": as_int(yawn.get("high_frequency_count"), 3),
            "timestamps_max": as_int(yawn.get("timestamps_max"), 10),
            # Not in your YAML; detector uses it -> default safely
            "covered_mar_min": as_float(yawn.get("covered_mar_min"), 0.45),
        },
        "expression_suppression": {
            "smile_frames": int(smile_frames),
            "laugh_frames": int(laugh_frames),
        },
        "severity": {
            "drowsiness": {
                "medium_sec": as_float(sev.get("medium_sec"), 2.0),
                "high_sec": as_float(sev.get("high_sec"), 3.0),
                "critical_sec": as_float(sev.get("critical_sec"), 5.0),
            }
        },
    }