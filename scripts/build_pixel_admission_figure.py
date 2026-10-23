#!/usr/bin/env python3
"""Build a traceable same-frame GDOR pixel-admission visualization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != expected_size:
        raise ValueError(
            f"Mask {path} has size {image.size}, expected {expected_size}"
        )
    return np.asarray(image) > 127


def load_frame_row(path: Path, frame: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["frame"]) == frame:
                return row
    raise ValueError(f"Frame {frame} not found in {path}")


def overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    output = rgb.astype(np.float32).copy()
    paint = np.empty_like(output)
    paint[:] = color
    output[mask] = output[mask] * (1.0 - alpha) + paint[mask] * alpha
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    output = image.copy()
    output.thumbnail(size, Image.Resampling.LANCZOS)
    return output


def make_panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    accent: str,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline="#cbd7d3", width=2)
    draw.rectangle((0, 0, width - 1, 8), fill=accent)
    draw.text((18, 24), title, fill="#17282e", font=get_font(25, True))
    draw.text((18, 62), subtitle, fill="#617176", font=get_font(16))
    fitted = fit_image(image, (width - 36, height - 116))
    x = (width - fitted.width) // 2
    y = 99 + (height - 116 - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def resolve_sources(evidence_dir: Path, frame: int) -> dict[str, Path]:
    stem = f"frame_{frame:06d}"
    paths = {
        "rgb": evidence_dir / f"{stem}_rgb.png",
        "raw_dynamic": evidence_dir / f"{stem}_raw_dynamic_mask.png",
        "recovered": evidence_dir / f"{stem}_recovered_tracking_support.png",
        "admission": evidence_dir / f"{stem}_mapping_admission.png",
        "overlap": evidence_dir / f"{stem}_recovered_admission_overlap.png",
        "refined_static": evidence_dir / f"{stem}_refined_static_mask.png",
        "metadata": evidence_dir / "instrumentation_metadata.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replay evidence:\n" + "\n".join(missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a five-panel figure from runtime-exported pixel masks."
    )
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sequence", default="tum_sitting_halfsphere")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--original-frame-metrics", type=Path, required=True)
    parser.add_argument("--replay-frame-metrics", type=Path, required=True)
    parser.add_argument("--mask-config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    metadata_path = args.evidence_dir / "instrumentation_metadata.json"
    metadata = load_json(metadata_path)
    frame = int(metadata["current_frame"])
    paths = resolve_sources(args.evidence_dir, frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb_image = Image.open(paths["rgb"]).convert("RGB")
    rgb = np.asarray(rgb_image)
    expected_size = rgb_image.size
    raw_dynamic = load_binary_mask(paths["raw_dynamic"], expected_size)
    recovered = load_binary_mask(paths["recovered"], expected_size)
    admission = load_binary_mask(paths["admission"], expected_size)
    exported_overlap = load_binary_mask(paths["overlap"], expected_size)
    computed_overlap = recovered & admission
    if not np.array_equal(exported_overlap, computed_overlap):
        raise ValueError("Exported overlap differs from recovered AND admission")

    counts = {
        "image_pixels": int(raw_dynamic.size),
        "raw_dynamic": int(raw_dynamic.sum()),
        "recovered_tracking_support": int(recovered.sum()),
        "persistent_mapping_admission": int(admission.sum()),
        "recovered_admission_overlap": int(computed_overlap.sum()),
    }
    metadata_counts = {
        "raw_dynamic": int(metadata["raw_dynamic_pixels"]),
        "recovered_tracking_support": int(metadata["recovered_tracking_pixels"]),
        "persistent_mapping_admission": int(metadata["mapping_admission_pixels"]),
        "recovered_admission_overlap": int(
            metadata["recovered_admission_overlap_pixels"]
        ),
    }
    for key, value in metadata_counts.items():
        if counts[key] != value:
            raise ValueError(
                f"Pixel count mismatch for {key}: PNG={counts[key]}, metadata={value}"
            )

    original_row = load_frame_row(args.original_frame_metrics, frame)
    replay_row = load_frame_row(args.replay_frame_metrics, frame)

    colors = {
        "raw": (219, 61, 49),
        "recovered": (15, 190, 102),
        "admission": (27, 132, 218),
        "overlap": (226, 33, 179),
    }
    panel_size = (350, 430)
    panels = [
        make_panel(
            rgb_image,
            "A  真实 RGB",
            f"frame {frame} · seed {args.seed}",
            "#66777c",
            panel_size,
        ),
        make_panel(
            overlay(rgb, raw_dynamic, colors["raw"], 0.62),
            "B  恢复前排除",
            f"raw dynamic · {counts['raw_dynamic']:,} px",
            "#db3d31",
            panel_size,
        ),
        make_panel(
            overlay(rgb, recovered, colors["recovered"], 0.82),
            "C  Tracking 恢复",
            f"temporary use · {counts['recovered_tracking_support']:,} px",
            "#0fbe66",
            panel_size,
        ),
        make_panel(
            overlay(rgb, admission, colors["admission"], 0.62),
            "D  Mapping 准入",
            f"persistent admission · {counts['persistent_mapping_admission']:,} px",
            "#1b84da",
            panel_size,
        ),
        make_panel(
            Image.fromarray(
                np.where(
                    computed_overlap[..., None],
                    np.asarray(colors["overlap"], dtype=np.uint8),
                    np.asarray((18, 31, 36), dtype=np.uint8),
                )
            ),
            "E  直接准入交集",
            f"recovered ∩ admission · {counts['recovered_admission_overlap']:,} px",
            "#e221b3",
            panel_size,
        ),
    ]

    width, height = 1900, 720
    canvas = Image.new("RGB", (width, height), "#f2f5f3")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 27),
        "GDOR-SLAM：同一真实帧上的 Tracking 使用权与 Mapping 准入权",
        fill="#17282e",
        font=get_font(37, True),
    )
    draw.text(
        (50, 79),
        "DYN-19 Full diagnostic replay · TUM sitting_halfsphere · not an original release mask artifact",
        fill="#607176",
        font=get_font(19),
    )
    gap = 16
    x = 43
    for panel in panels:
        canvas.paste(panel, (x, 125))
        x += panel_size[0] + gap

    draw.rounded_rectangle(
        (43, 579, 1857, 686), radius=6, fill="#ffffff", outline="#cbd7d3", width=2
    )
    draw.text(
        (68, 598),
        "route invariant",
        fill="#a43b32",
        font=get_font(19, True),
    )
    draw.text(
        (68, 630),
        "recovered_support = raw_dynamic ∩ tracking_static;  mapping_admission = staticWeight ≥ 0.5;  overlap = 0 px",
        fill="#17282e",
        font=get_font(22, True),
    )
    draw.text(
        (1125, 600),
        f"Original run selector: {int(original_row['tracking_recovery_audit_pixels']):,} px",
        fill="#607176",
        font=get_font(17),
    )
    draw.text(
        (1125, 630),
        f"Replay observation: {counts['recovered_tracking_support']:,} px · direct overlap {counts['recovered_admission_overlap']:,} px",
        fill="#607176",
        font=get_font(17),
    )

    figure_path = args.output_dir / "fig_pixel_admission_real_frame.png"
    canvas.save(figure_path, optimize=True)

    public_source_dir = args.output_dir / "source"
    public_source_dir.mkdir(exist_ok=True)
    public_source_files = {}
    for key, source_path in paths.items():
        target_path = public_source_dir / source_path.name
        shutil.copy2(source_path, target_path)
        public_source_files[key] = {
            "path": str(target_path.relative_to(args.output_dir)),
            "sha256": sha256_path(target_path),
        }

    source_hashes = {
        key: sha256_path(path)
        for key, path in paths.items()
        if key != "metadata"
    }
    provenance = {
        "schema": "gdor-pixel-admission-figure-v1",
        "figure": figure_path.name,
        "figure_sha256": sha256_path(figure_path),
        "evidence_kind": "diagnostic replay",
        "sequence": args.sequence,
        "seed": args.seed,
        "processed_frame_index": frame,
        "frame_index_basis": metadata["frame_index_basis"],
        "selection": {
            "policy": "maximum tracking_recovery_audit_pixels in the original DYN-19 Full sitting_halfsphere seed-0 frame log",
            "original_frame_metrics": str(args.original_frame_metrics.resolve()),
            "original_frame_metrics_sha256": sha256_path(
                args.original_frame_metrics
            ),
            "original_tracking_recovery_audit_pixels": int(
                original_row["tracking_recovery_audit_pixels"]
            ),
            "original_mapping_leak_pixels": int(
                original_row["tracking_recovery_mapping_leak_pixels"]
            ),
        },
        "replay": {
            "evidence_dir": str(args.evidence_dir.resolve()),
            "replay_frame_metrics": str(args.replay_frame_metrics.resolve()),
            "replay_frame_metrics_sha256": sha256_path(args.replay_frame_metrics),
            "binary": str(args.binary.resolve()),
            "binary_sha256": sha256_path(args.binary),
            "source_file": str(args.source_file.resolve()),
            "source_file_sha256": sha256_path(args.source_file),
            "source_commit_before_diagnostic_patch": args.source_commit,
            "mask_config": str(args.mask_config.resolve()),
            "mask_config_sha256": sha256_path(args.mask_config),
            "association": metadata["sources"]["association"],
            "association_sha256": sha256_path(
                Path(metadata["sources"]["association"])
            ),
            "metadata": str(metadata_path.resolve()),
            "metadata_sha256": sha256_path(metadata_path),
        },
        "pixel_counts": counts,
        "public_source_files": public_source_files,
        "source_png_sha256": source_hashes,
        "definitions": metadata["evidence_contract"],
        "claim_boundary": (
            "This same-frame replay checks direct route admission using the exact "
            "runtime masks. It is not an original DYN-19 release mask artifact, "
            "does not establish independent zero-ghost ground truth, and does not "
            "exclude indirect Mapping changes through an improved camera pose."
        ),
    }
    provenance_path = (
        args.output_dir / "fig_pixel_admission_real_frame.provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure": str(figure_path), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
