"""
One-time (idempotent) setup: reset flowers_corrupted demo to the intended state.

After running this:
  - data/flowers_corrupted/   → exactly 16 original images (no synthetic/augmented)
  - flowers_corrupted_features.csv → 20 rows: 16 real + 4 NaN (the cascade)
  - flowers_postfix_features.csv  → 20 rows: all filled (post-fix state)
  - flowers_dalle/                → 4 pre-generated PIL images (DALL-E placeholders)
  - flowers_corrupted_manifest.json
  - cache/flowers_corrupted.json  → quality numbers updated to reflect NaN rows

Run: python data/setup_flowers_demo.py
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DATA_DIR   = Path(__file__).parent
FLOWERS_DIR = DATA_DIR / "flowers_corrupted"
DALLE_DIR   = DATA_DIR / "flowers_dalle"
MANIFEST    = DATA_DIR / "flowers_corrupted_manifest.json"
FEATURES    = DATA_DIR / "flowers_corrupted_features.csv"
POSTFIX     = DATA_DIR / "flowers_postfix_features.csv"
CACHE_JSON  = DATA_DIR / "cache" / "flowers_corrupted.json"

# The 4 images that were "deleted" in the corruption step.
# These filenames must NOT exist in flowers_corrupted/.
MISSING = [
    {"filename": "daisy_flower_003.png",     "flower_class": "daisy"},
    {"filename": "lavender_flower_007.png",  "flower_class": "lavender"},
    {"filename": "rose_flower_012.png",      "flower_class": "rose"},
    {"filename": "sunflower_flower_019.png", "flower_class": "sunflower"},
]

FEATURE_COLS = [f"feature_{i}" for i in range(96)]
META_COLS    = ["filename", "file_size_kb", "width_px", "height_px",
                "aspect_ratio", "mode", "format"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hist_features(pil_img: Image.Image) -> list:
    """Color histogram: 3 channels × 32 bins = 96 features."""
    arr = np.array(pil_img.convert("RGB").resize((224, 224)))
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 256))
        feats.extend(hist.tolist())
    return feats


def _make_flower_image(flower: str) -> Image.Image:
    """Draw a synthetic flower with PIL (serves as DALL-E placeholder)."""
    palette = {
        "daisy":     [(255, 255, 200), (255, 255, 100), (200, 200, 0)],
        "lavender":  [(200, 140, 240), (220, 180, 255), (140, 80, 190)],
        "rose":      [(230, 30,  60),  (255, 90, 110),  (180, 0, 40)],
        "sunflower": [(255, 200, 0),   (240, 170, 0),   (160, 120, 0)],
        "tulip":     [(255, 60, 160),  (230, 10, 130),  (200, 0, 100)],
    }
    colors = palette.get(flower, palette["daisy"])
    rng = np.random.RandomState(abs(hash(flower)) % 2**31)

    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = 128, 128

    n_petals = 6
    for k in range(n_petals):
        angle = (2 * np.pi * k) / n_petals
        px = cx + 65 * np.cos(angle)
        py = cy + 65 * np.sin(angle)
        draw.ellipse((px - 32, py - 32, px + 32, py + 32),
                     fill=colors[0], outline=colors[2])

    draw.ellipse((cx - 22, cy - 22, cx + 22, cy + 22),
                 fill=(255, 210, 30), outline=(200, 160, 0))

    arr = np.array(img)
    noise = rng.randint(-12, 12, arr.shape)
    img = Image.fromarray(np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8))
    return img


def _img_to_b64(img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_cleanup():
    print("1. Cleaning up synthetic/augmented images...")
    removed = []
    for p in FLOWERS_DIR.glob("*.png"):
        if p.stem.startswith(("synthetic_", "augmented_", "dalle_")):
            p.unlink()
            removed.append(p.name)
    count = sum(1 for _ in FLOWERS_DIR.glob("*.png"))
    print(f"   Removed {len(removed)} extra files. Remaining: {count} images.")
    if count != 16:
        print(f"   WARNING: expected 16 images but found {count}. "
              "Re-run generate_sample_images.py if needed.", file=sys.stderr)


def step_manifest():
    print("2. Writing manifest...")
    manifest = {
        "expected_count": 20,
        "deleted_count": len(MISSING),
        "deleted_images": MISSING,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"   Saved {MANIFEST.name}")


def step_features_csv():
    """Regenerate features CSV from current images, then add 4 NaN rows."""
    print("3. Rebuilding features CSV with NaN rows for missing images...")

    # -- Regenerate clean rows from existing images --------------------------
    rows = []
    for img_path in sorted(FLOWERS_DIR.glob("*.png")):
        try:
            img = Image.open(img_path)
            feats = _hist_features(img)
            stat  = img_path.stat()
            parts = img_path.stem.split("_")
            flower_class = parts[0] if parts else "unknown"
            row = {"image_id": len(rows), "flower_class": flower_class}
            for i, v in enumerate(feats):
                row[f"feature_{i}"] = float(v)
            row.update({
                "filename":     img_path.name,
                "file_size_kb": round(stat.st_size / 1024, 4),
                "width_px":     float(img.size[0]),
                "height_px":    float(img.size[1]),
                "aspect_ratio": round(img.size[0] / img.size[1], 4),
                "mode":         img.mode,
                "format":       "PNG",
            })
            rows.append(row)
        except Exception as e:
            print(f"   WARNING: could not process {img_path.name}: {e}", file=sys.stderr)

    df_clean = pd.DataFrame(rows)

    # Column order: image_id, flower_class, feature_*, filename, meta...
    col_order = (["image_id", "flower_class"] + FEATURE_COLS + META_COLS)
    df_clean = df_clean.reindex(columns=col_order)

    # -- Append 4 NaN rows ---------------------------------------------------
    max_id = int(df_clean["image_id"].max())
    nan_rows = []
    for i, m in enumerate(MISSING):
        row = {
            "image_id":     max_id + 1 + i,
            "flower_class": m["flower_class"],
            "filename":     m["filename"],
        }
        # All feature + metadata columns → NaN
        for col in FEATURE_COLS + ["file_size_kb", "width_px", "height_px",
                                    "aspect_ratio", "mode", "format"]:
            row[col] = np.nan
        nan_rows.append(row)

    df_nan = pd.DataFrame(nan_rows, columns=col_order)
    df_full = pd.concat([df_clean, df_nan], ignore_index=True)
    df_full.to_csv(FEATURES, index=False)

    missing_cells = int(df_full.isnull().sum().sum())
    total_cells   = df_full.size
    completeness  = round((1 - missing_cells / total_cells) * 100, 1)
    print(f"   {FEATURES.name}: {len(df_full)} rows | "
          f"missing={missing_cells}/{total_cells} ({100-completeness:.1f}%) | "
          f"completeness={completeness}%")
    return df_full


def step_dalle_images():
    """Generate PIL synthetic flower images as DALL-E 2 placeholders."""
    print("4. Generating DALL-E placeholder images...")
    DALLE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate one per class (covers all 5 so UI selector always works)
    classes = ["daisy", "lavender", "rose", "sunflower", "tulip"]
    for cls in classes:
        img = _make_flower_image(cls)
        out = DALLE_DIR / f"dalle_{cls}.png"
        img.save(out)
        print(f"   Saved {out.name}")


def step_postfix_csv(df_full: pd.DataFrame):
    """Fill in NaN rows using features from DALL-E placeholder images."""
    print("5. Building post-fix CSV (all NaN rows filled)...")
    df_fixed = df_full.copy()

    nan_mask = df_fixed[FEATURE_COLS].isnull().all(axis=1)
    filled = 0
    for idx in df_fixed.index[nan_mask]:
        cls      = df_fixed.at[idx, "flower_class"]
        img_path = DALLE_DIR / f"dalle_{cls}.png"

        if not img_path.exists():
            # Fallback: flip an existing same-class image
            src = sorted(FLOWERS_DIR.glob(f"{cls}_*.png"))
            if not src:
                print(f"   WARNING: no source image for class {cls}", file=sys.stderr)
                continue
            img = Image.open(src[0]).transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = Image.open(img_path)

        feats = _hist_features(img)
        for i, v in enumerate(feats):
            df_fixed.at[idx, f"feature_{i}"] = float(v)
        df_fixed.at[idx, "file_size_kb"]  = 85.0
        df_fixed.at[idx, "width_px"]      = 224.0
        df_fixed.at[idx, "height_px"]     = 224.0
        df_fixed.at[idx, "aspect_ratio"]  = 1.0
        df_fixed.at[idx, "mode"]          = "RGB"
        df_fixed.at[idx, "format"]        = "PNG"
        filled += 1

    df_fixed.to_csv(POSTFIX, index=False)
    remaining_nan = int(df_fixed.isnull().sum().sum())
    print(f"   {POSTFIX.name}: {len(df_fixed)} rows | filled {filled} NaN rows | "
          f"remaining NaN={remaining_nan}")
    return df_fixed


def step_update_cache(df_full: pd.DataFrame):
    """Patch cache JSON so the quality numbers match the NaN-row CSV."""
    print("6. Updating cache JSON...")
    if not CACHE_JSON.exists():
        print("   WARNING: cache JSON not found — skipping.", file=sys.stderr)
        return

    with open(CACHE_JSON) as f:
        cache = json.load(f)

    results = cache["results"]

    # Recompute quality using the same cascade-factor formula as _compute_quality_from_df
    total_rows    = len(df_full)
    feature_cols  = [c for c in df_full.columns if c.startswith("feature_")]
    nan_rows      = int(df_full[feature_cols].isnull().all(axis=1).sum())
    completeness  = round(((total_rows - nan_rows) / total_rows) * 100, 1)
    uniqueness    = 100.0
    consistency   = 100.0   # no penalties with improved formula (skips feature_* and metadata)
    base          = completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3
    coverage      = (total_rows - nan_rows) / total_rows
    cascade_factor = 0.5 + 0.5 * coverage
    overall        = round(base * cascade_factor)

    total_cells   = df_full.size
    missing_cells = int(df_full.isnull().sum().sum())

    for tr in results.get("quality_tool_results", []):
        if tr.get("tool") == "compute_data_quality_score":
            tr["result"]["missing_values_count"] = missing_cells
            tr["result"]["total_cells"]          = total_cells
            tr["result"]["breakdown"]["completeness"] = completeness
            tr["result"]["breakdown"]["uniqueness"]   = uniqueness
            tr["result"]["breakdown"]["consistency"]  = consistency
            tr["result"]["overall_score"]             = overall
            for issue in tr["result"].get("issues", []):
                if issue.get("issue_type") == "image_coverage":
                    issue["issue"] = (
                        "Image coverage is only 80% (16/20 images). "
                        "4 images failed feature extraction → 4 NaN rows injected into "
                        "the features table (daisy_flower_003, lavender_flower_007, "
                        "rose_flower_012, sunflower_flower_019). "
                        "NaN rows silently bias any model trained on this dataset."
                    )

    # Keep the analysis_plan text aligned
    results["image_processing_metadata"] = {
        "processed_count":  16,
        "total_count":      20,
        "coverage_percent": 80.0,
        "missing_images":   [m["filename"] for m in MISSING],
    }

    CACHE_JSON.write_text(json.dumps(cache, indent=2))
    print(f"   completeness={completeness}%  overall_score={overall}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("AutoAnalyst — Flowers Demo Setup")
    print("=" * 60)

    if not FLOWERS_DIR.exists():
        sys.exit(f"ERROR: {FLOWERS_DIR} does not exist. "
                 "Run generate_sample_images.py first.")

    step_cleanup()
    step_manifest()
    df_full = step_features_csv()
    step_dalle_images()
    step_postfix_csv(df_full)
    step_update_cache(df_full)

    print()
    print("✓ Setup complete.")
    print(f"  Corrupted CSV : {FEATURES}  ({len(df_full)} rows, 4 NaN)")
    print(f"  Post-fix CSV  : {POSTFIX}")
    print(f"  DALL-E images : {DALLE_DIR}/dalle_{{class}}.png")
    print(f"  Manifest      : {MANIFEST}")
