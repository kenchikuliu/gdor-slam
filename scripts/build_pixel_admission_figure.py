#!/usr/bin/env python3
"""Build a traceable same-frame GDOR pixel-admission visualization."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_from_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[Image.Image, bytes]:
    data = tf.extractfile(member).read()
    return Image.open(io.BytesIO(data)).convert("L"), data


def frame_id(name: str) -> int | None:
    stem = Path(name).stem
    hits = re.findall(r"(?<!\d)(\d{3,7})(?!\d)", stem)
    return int(hits[-1]) if hits else None


def classify(name: str) -> str | None:
    value = name.lower()
    if not value.endswith(IMAGE_EXTS):
        return None
    if any(key in value for key in ("raw_static", "raw_mask", "pre_recovery", "mapping_mask")):
        return "raw"
    if any(key in value for key in ("recovered_support", "recovery_support", "recovered_mask")):
        return "recovered"
    if any(key in value for key in ("static_mask", "refined_mask", "tracking_mask")):
        return "refined"
    if any(key in value for key in ("admission", "mapping_weight", "map_weight")):
        return "admission"
    if any(key in value for key in ("rgb", "color", "frame")):
        return "rgb"
    return None


def find_sequence_root() -> Path | None:
    patterns = (
        "**/rgbd_dataset_freiburg3_sitting_halfsphere",
        "**/sitting_halfsphere",
    )
    roots = [Path("/home/slam"), Path("/media/slam"), Path("/mnt")]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if (path / "rgb").is_dir() or (path / "associations.txt").is_file():
                    return path
    return None


def rgb_for_frame(sequence: Path, index: int) -> Path | None:
    association_files = [sequence / "associations.txt", sequence / "associate.txt"]
    for association in association_files:
        if not association.is_file():
            continue
        rows = []
        for line in association.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            rgb_tokens = [token for token in tokens if token.lower().startswith("rgb/")]
            if rgb_tokens:
                rows.append(rgb_tokens[0])
        if rows:
            index = max(0, min(index, len(rows) - 1))
            candidate = sequence / rows[index]
            if candidate.is_file():
                return candidate
    images = sorted((sequence / "rgb").glob("*")) if (sequence / "rgb").is_dir() else []
    images = [path for path in images if path.suffix.lower() in IMAGE_EXTS]
    if images:
        return images[max(0, min(index, len(images) - 1))]
    return None


def bool_mask(image: Image.Image, shape: tuple[int, int] | None = None) -> np.ndarray:
    if shape and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    array = np.asarray(image.convert("L"))
    return array > 127


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    paint = np.zeros_like(out)
    paint[:] = color
    out[mask] = out[mask] * (1.0 - alpha) + paint[mask] * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_panel(image: Image.Image, title: str, subtitle: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), title, fill="#17282e", font=font(24, True))
    draw.text((18, 49), subtitle, fill="#66777c", font=font(15))
    available = (width - 36, height - 92)
    fitted = image.copy()
    fitted.thumbnail(available, Image.Resampling.LANCZOS)
    x = (width - fitted.width) // 2
    y = 82 + (available[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    draw.rectangle((0, 0, width - 1, height - 1), outline="#d4ddda", width=2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = None
    inventory = []
    with tarfile.open(args.archive, "r:gz") as tf:
        grouped: dict[int, dict[str, tarfile.TarInfo]] = {}
        for member in tf.getmembers():
            lower = member.name.lower()
            if not member.isfile() or "sitting_halfsphere" not in lower:
                continue
            if not any(token in lower for token in ("seed0", "seed_0", "seed-0", "/0/")):
                continue
            kind = classify(member.name)
            fid = frame_id(member.name)
            if kind and fid is not None:
                grouped.setdefault(fid, {})[kind] = member
                inventory.append({"frame": fid, "kind": kind, "member": member.name, "bytes": member.size})

        candidates = []
        for fid, layers in grouped.items():
            if "raw" not in layers and "refined" not in layers:
                continue
            raw_image, _ = image_from_member(tf, layers.get("raw", layers["refined"]))
            raw = bool_mask(raw_image)
            if "recovered" in layers:
                recovered_image, _ = image_from_member(tf, layers["recovered"])
                recovered = bool_mask(recovered_image, raw.shape)
            elif "raw" in layers and "refined" in layers:
                refined_image, _ = image_from_member(tf, layers["refined"])
                refined = bool_mask(refined_image, raw.shape)
                recovered = refined & ~raw
            else:
                continue
            count = int(recovered.sum())
            if count:
                candidates.append((count, fid, layers, raw, recovered))
        if not candidates:
            raise RuntimeError("No same-frame raw/refined or recovered mask pair found in the archive")
        candidates.sort(reverse=True, key=lambda item: item[0])
        recovered_count, fid, layers, raw_static, recovered = candidates[0]

        if "admission" in layers:
            admission_image, admission_bytes = image_from_member(tf, layers["admission"])
            admission = bool_mask(admission_image, raw_static.shape)
            admission_source = layers["admission"].name
        else:
            admission = raw_static.copy()
            admission_bytes = admission.astype(np.uint8).tobytes()
            admission_source = "derived from raw pre-recovery static mask"

        rgb_source = None
        if "rgb" in layers:
            rgb_data = tf.extractfile(layers["rgb"]).read()
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")
            rgb_source = layers["rgb"].name
        else:
            sequence = find_sequence_root()
            rgb_path = rgb_for_frame(sequence, fid) if sequence else None
            if not rgb_path:
                raise RuntimeError(f"No RGB frame found for processed frame index {fid}")
            rgb_data = rgb_path.read_bytes()
            rgb_image = Image.open(io.BytesIO(rgb_data)).convert("RGB")
            rgb_source = str(rgb_path)

        h, w = raw_static.shape
        rgb_image = rgb_image.resize((w, h), Image.Resampling.LANCZOS)
        rgb = np.asarray(rgb_image)
        excluded = ~raw_static
        overlap = recovered & admission

        layers_out = {
            "rgb": rgb_image,
            "raw_exclusion": Image.fromarray((excluded * 255).astype(np.uint8)),
            "recovered_support": Image.fromarray((recovered * 255).astype(np.uint8)),
            "mapping_admission": Image.fromarray((admission * 255).astype(np.uint8)),
            "overlap": Image.fromarray((overlap * 255).astype(np.uint8)),
        }
        layer_hashes = {}
        for name, image in layers_out.items():
            path = args.output_dir / f"pixel_admission_{name}.png"
            image.save(path)
            layer_hashes[name] = sha256(path.read_bytes())

        panel_size = (700, 490)
        panels = [
            make_panel(rgb_image, "A 真实 RGB", f"TUM sitting_halfsphere · processed frame {fid}", panel_size),
            make_panel(overlay(rgb, excluded, (210, 55, 45), .58), "B 恢复前排除区域", f"红色：raw mask 排除 · {int(excluded.sum()):,} px", panel_size),
            make_panel(overlay(rgb, recovered, (20, 210, 105), .82), "C Tracking 恢复支持", f"绿色：临时 tracking-only · {recovered_count:,} px", panel_size),
            make_panel(overlay(rgb, admission, (20, 145, 220), .42), "D Mapping 准入", f"蓝色：persistent admission · {int(admission.sum()):,} px", panel_size),
            make_panel(overlay(rgb, overlap, (245, 35, 190), .95), "E 像素级 overlap audit", f"洋红：recovered ∩ admission · {int(overlap.sum()):,} px", panel_size),
        ]
        canvas = Image.new("RGB", (1480, 1640), "#f4f6f5")
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 28), "GDOR-SLAM：同一真实帧上的像素使用权与地图准入", fill="#17282e", font=font(36, True))
        draw.text((40, 78), "Tracking 可以临时借用绿色像素；Mapping 仍由恢复前的蓝色准入边界控制。", fill="#4f6267", font=font(21))
        positions = [(30, 125), (750, 125), (30, 635), (750, 635), (390, 1145)]
        for panel, position in zip(panels, positions):
            canvas.paste(panel, position)
        figure_path = args.output_dir / "fig_pixel_admission_real_frame.png"
        canvas.save(figure_path)

        provenance = {
            "figure": figure_path.name,
            "sequence": "tum_sitting_halfsphere",
            "seed": 0,
            "processed_frame_index": fid,
            "archive": str(args.archive),
            "archive_sha256": sha256(args.archive.read_bytes()),
            "rgb_source": rgb_source,
            "rgb_sha256": sha256(rgb_data),
            "raw_mask_member": layers.get("raw", layers.get("refined")).name,
            "refined_mask_member": layers.get("refined").name if layers.get("refined") else None,
            "recovered_member": layers.get("recovered").name if layers.get("recovered") else "derived as refined AND NOT raw",
            "mapping_admission_source": admission_source,
            "pixel_counts": {
                "image_pixels": int(h * w),
                "raw_excluded": int(excluded.sum()),
                "recovered_tracking_support": recovered_count,
                "persistent_mapping_admission": int(admission.sum()),
                "recovered_admission_overlap": int(overlap.sum()),
            },
            "layer_sha256": layer_hashes,
            "inventory": inventory,
            "claim_boundary": "The overlap panel audits direct pixel admission on this route. It is not independent ground truth for zero ghost contamination.",
        }
        (args.output_dir / "fig_pixel_admission_real_frame.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
