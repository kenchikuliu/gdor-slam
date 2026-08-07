#!/usr/bin/env python3
"""Evaluate final-map renders on non-keyframe held-out RGB observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def masked_psnr(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    squared = (reference.astype(np.float32) - estimate.astype(np.float32)) ** 2
    values = squared[mask]
    if values.size == 0:
        return float("nan")
    mse = float(values.mean())
    return float("inf") if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse)


def masked_ssim(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    reference = reference.astype(np.float32)
    estimate = estimate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    channel_scores = []
    support = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if not support.any():
        support = mask
    for channel in range(3):
        x = reference[:, :, channel]
        y = estimate[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        score = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        channel_scores.append(float(score[support].mean()))
    return float(np.mean(channel_scores))


class LpipsEvaluator:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.error = ""
        self.torch = None
        self.model = None
        if not enabled:
            return
        try:
            import lpips
            import torch

            self.torch = torch
            self.model = lpips.LPIPS(net="alex", spatial=True).eval()
        except Exception as exc:  # dependency/model availability is recorded, never hidden
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"

    def evaluate_regions(
        self,
        reference_bgr: np.ndarray,
        estimate_bgr: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> dict[str, float]:
        if not self.enabled or self.model is None or self.torch is None:
            return {name: float("nan") for name in masks}
        torch = self.torch
        reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        estimate = cv2.cvtColor(estimate_bgr, cv2.COLOR_BGR2RGB)
        reference_tensor = torch.from_numpy(reference).permute(2, 0, 1).float()[None]
        estimate_tensor = torch.from_numpy(estimate).permute(2, 0, 1).float()[None]
        reference_tensor = reference_tensor / 127.5 - 1.0
        estimate_tensor = estimate_tensor / 127.5 - 1.0
        with torch.no_grad():
            distance = self.model(reference_tensor, estimate_tensor)
        results = {}
        for name, mask in masks.items():
            mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor, size=distance.shape[-2:], mode="nearest")
            denominator = float(mask_tensor.sum())
            results[name] = (
                float("nan") if denominator == 0.0
                else float((distance * mask_tensor).sum() / denominator)
            )
        return results


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Held-out manifest does not exist: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "ground_truth", "static_mask"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Held-out manifest must contain {sorted(required)}: {path}")
    frame_ids = [int(row["frame"]) for row in rows]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Held-out manifest contains duplicate frame IDs")
    if any(not row["ground_truth"] or not row["static_mask"] for row in rows):
        raise ValueError(
            "Every held-out manifest row must name ground_truth and static_mask")
    return rows


def load_manifest_triplets(
    manifest_path: Path,
    heldout_dir: Path,
    render_dir: Path,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    manifest_rows = load_manifest(manifest_path)
    expected_render_names = {
        f"{int(row['frame']):06d}.png" for row in manifest_rows
    }
    observed_render_names = {path.name for path in render_dir.glob("*.png")}
    missing = sorted(expected_render_names - observed_render_names)
    unexpected = sorted(observed_render_names - expected_render_names)
    if missing or unexpected:
        raise RuntimeError(
            "Render set does not exactly match the common held-out manifest; "
            f"missing={missing}, unexpected={unexpected}")

    triplets = []
    errors = []
    for row in manifest_rows:
        frame_id = int(row["frame"])
        reference_path = heldout_dir / row["ground_truth"]
        estimate_path = render_dir / f"{frame_id:06d}.png"
        mask_path = heldout_dir / row["static_mask"]
        for label, path, hash_field in (
            ("ground_truth", reference_path, "ground_truth_sha256"),
            ("static_mask", mask_path, "static_mask_sha256"),
        ):
            expected_hash = row.get(hash_field, "")
            if expected_hash and path.is_file() and sha256(path) != expected_hash:
                errors.append(
                    f"frame {frame_id}: {label} hash does not match manifest")
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        estimate = cv2.imread(str(estimate_path), cv2.IMREAD_COLOR)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unreadable = [
            label for label, image in (
                ("ground_truth", reference),
                ("render", estimate),
                ("static_mask", mask_image),
            )
            if image is None
        ]
        if unreadable:
            errors.append(f"frame {frame_id}: unreadable {','.join(unreadable)}")
            continue
        if reference.shape != estimate.shape:
            errors.append(
                f"frame {frame_id}: GT/render shape mismatch "
                f"{reference.shape} != {estimate.shape}")
            continue
        if reference.shape[:2] != mask_image.shape:
            errors.append(
                f"frame {frame_id}: GT/mask shape mismatch "
                f"{reference.shape[:2]} != {mask_image.shape}")
            continue
        if not np.any(mask_image > 0):
            errors.append(f"frame {frame_id}: static_mask has no positive pixels")
            continue
        triplets.append((frame_id, reference, estimate, mask_image))
    if errors:
        raise RuntimeError(
            "Held-out manifest validation failed:\n" + "\n".join(errors))
    return triplets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("heldout_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        default="final online Gaussian map rendered at held-out poses")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    heldout_dir = args.heldout_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest else heldout_dir / "manifest.csv"
    )
    render_dir = args.render_dir or args.heldout_dir / "render"
    output_dir = args.output_dir or args.heldout_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    triplets = load_manifest_triplets(
        manifest_path, heldout_dir, render_dir.resolve())

    lpips_evaluator = LpipsEvaluator(not args.skip_lpips)
    rows = []
    for frame_id, reference, estimate, mask_image in triplets:
        static_mask = mask_image > 0
        full_mask = np.ones(static_mask.shape, dtype=bool)
        lpips_scores = lpips_evaluator.evaluate_regions(
            reference, estimate, {"full": full_mask, "static": static_mask})
        rows.append({
            "frame": frame_id,
            "static_fraction": float(static_mask.mean()),
            "psnr_full": masked_psnr(reference, estimate, full_mask),
            "ssim_full": masked_ssim(reference, estimate, full_mask),
            "lpips_full": lpips_scores["full"],
            "psnr_static": masked_psnr(reference, estimate, static_mask),
            "ssim_static": masked_ssim(reference, estimate, static_mask),
            "lpips_static": lpips_scores["static"],
        })

    csv_path = output_dir / "heldout_metrics_per_frame.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": args.protocol,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "frame_ids": [row["frame"] for row in rows],
        "regions": {
            "full": "all image pixels",
            "static": "positive pixels in the fixed evaluation mask",
        },
        "frames": len(rows),
        "psnr_full_mean": finite_mean([row["psnr_full"] for row in rows]),
        "ssim_full_mean": finite_mean([row["ssim_full"] for row in rows]),
        "lpips_full_mean": finite_mean([row["lpips_full"] for row in rows]),
        "psnr_static_mean": finite_mean([row["psnr_static"] for row in rows]),
        "ssim_static_mean": finite_mean([row["ssim_static"] for row in rows]),
        "lpips_static_mean": finite_mean([row["lpips_static"] for row in rows]),
        "lpips_available": lpips_evaluator.enabled,
        "lpips_error": lpips_evaluator.error or None,
    }
    # Backward-compatible aliases used by the benchmark aggregator.
    summary["psnr_mean"] = summary["psnr_static_mean"]
    summary["ssim_mean"] = summary["ssim_static_mean"]
    summary["lpips_mean"] = summary["lpips_static_mean"]
    with (output_dir / "heldout_metrics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
