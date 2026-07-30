#!/usr/bin/env python3
"""
s2_cloud_mask.py — Che mây (cloud masking) cho ảnh Sentinel-2 dạng folder `.SAFE`
gốc tải từ Copernicus (cả L1C và L2A).

Ý tưởng: **không tái tạo** pixel dưới mây (reconstruction), chỉ *đánh dấu và loại*
(masking) vùng mây + bóng mây — vừa đủ cho bài toán phát hiện tàu (tránh nhận nhầm
mây trắng thành tàu, tránh dự đoán trên vùng bị mây che).

Đầu vào : 1 folder `*.SAFE` (chứa 12–13 band JP2 + ảnh TCI như định dạng Copernicus).
Đầu ra  :
  - `<scene>_TCI_masked.tif` / `.png` : ảnh TCI đã che mây (pixel mây -> đen).
  - `<scene>_CLOUDMASK.tif` / `.png`   : mask nhị phân (255 = mây/bóng mây bị loại).
  - (tuỳ chọn) `tiles/images/*.png`     : cắt thành tile 800x800 để nạp thẳng YOLO,
    kèm `tiles/tiles_index.json` để ánh xạ toạ độ tile -> pixel trong scene gốc.

Phương pháp (`--method`):
  - `scl`          : dùng band SCL có sẵn trong sản phẩm **L2A** (nhanh, không cần model).
  - `omnicloudmask`: deep-learning SOTA (2025). Cần Red/Green/NIR. Chạy cho cả L1C & L2A.
  - `s2cloudless`  : ML single-scene cho **L1C** (cần band cirrus B10 nên không dùng cho L2A).
  - `auto`         : L2A -> `scl`; L1C -> `omnicloudmask` (fallback `s2cloudless`).

Phụ thuộc: `rasterio`, `numpy`, `Pillow` (bắt buộc). `omnicloudmask` và/hoặc
`s2cloudless` (chỉ cần cho đúng phương pháp tương ứng). Xem `requirements.txt`.

Dùng nhanh (CLI):
    python s2_cloud_mask.py --safe /duong/dan/S2A_MSIL1C_...SAFE --out out_dir \
        --method auto --tile 800 --tile-overlap 0 --skip-empty

Dùng như thư viện:
    from s2_cloud_mask import process_safe
    res = process_safe("S2A_..._.SAFE", "out_dir", method="auto", tile_size=800)
    print(res["masked_tci_png"], res["tiles_dir"])
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError as e:  # pragma: no cover - thông báo rõ ràng khi thiếu gói
    raise ImportError(
        "Cần cài `rasterio` để đọc file JP2 của Sentinel-2: pip install rasterio"
    ) from e

from PIL import Image

# --------------------------------------------------------------------------- #
# Bảng lớp SCL (Scene Classification Layer) trong sản phẩm L2A
#   0 NoData | 1 Saturated/Defective | 2 Dark area | 3 Cloud shadow | 4 Vegetation
#   5 Not-vegetated | 6 Water | 7 Unclassified | 8 Cloud medium prob
#   9 Cloud high prob | 10 Thin cirrus | 11 Snow/ice
# --------------------------------------------------------------------------- #
SCL_CLOUD = {8, 9}          # mây (trung bình + cao)
SCL_CIRRUS = {10}           # mây ti / cirrus mỏng
SCL_SHADOW = {3}            # bóng mây
SCL_DEFECT = {1}            # điểm bão hoà / lỗi (tuỳ chọn)

# Thứ tự 10 band mà s2cloudless cần (all_bands=False)
S2CLOUDLESS_BANDS = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]


# =========================================================================== #
# 1. Khám phá cấu trúc folder .SAFE
# =========================================================================== #
def detect_level(safe_dir: str) -> str:
    """Trả về 'L1C' hoặc 'L2A' dựa trên tên folder, fallback theo cấu trúc IMG_DATA."""
    name = os.path.basename(str(safe_dir).rstrip("/\\"))
    if "MSIL2A" in name:
        return "L2A"
    if "MSIL1C" in name:
        return "L1C"
    # Fallback: L2A có các thư mục con R10m/R20m/R60m trong IMG_DATA
    img = _find_granule_imgdata(safe_dir)
    if img and os.path.isdir(os.path.join(img, "R10m")):
        return "L2A"
    return "L1C"


def _find_granule_imgdata(safe_dir: str) -> Optional[str]:
    """Tìm thư mục GRANULE/*/IMG_DATA bên trong .SAFE."""
    cands = glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA"))
    return cands[0] if cands else None


def find_band_file(safe_dir: str, band: str, res: Optional[int] = None) -> Optional[str]:
    """
    Tìm file JP2 của một band trong .SAFE.

    band : 'B01'..'B12', 'B8A', 'TCI', hoặc 'SCL'.
    res  : độ phân giải mong muốn (10/20/60) — chỉ áp dụng cho L2A. Nếu None thì
           chọn độ phân giải cao nhất có sẵn.
    """
    img = _find_granule_imgdata(safe_dir)
    if not img:
        return None

    # L2A: có các thư mục con Rxxm
    if os.path.isdir(os.path.join(img, "R10m")):
        res_order = [res] if res else [10, 20, 60]
        for r in res_order:
            hits = sorted(glob.glob(os.path.join(img, f"R{r}m", f"*_{band}_{r}m.jp2")))
            if hits:
                return hits[0]
        return None

    # L1C: file phẳng trong IMG_DATA
    hits = sorted(glob.glob(os.path.join(img, f"*_{band}.jp2")))
    return hits[0] if hits else None


def scene_id(safe_dir: str) -> str:
    """ID scene gọn để đặt tên output."""
    return os.path.basename(str(safe_dir).rstrip("/\\")).replace(".SAFE", "")


# =========================================================================== #
# 2. Đọc raster
# =========================================================================== #
def read_band(path: str, out_shape: Optional[tuple[int, int]] = None,
              resampling: "Resampling" = Resampling.bilinear) -> np.ndarray:
    """Đọc band 1 của file, tuỳ chọn resample về out_shape=(h, w)."""
    with rasterio.open(path) as ds:
        if out_shape is not None:
            return ds.read(1, out_shape=out_shape, resampling=resampling).astype(np.float32)
        return ds.read(1).astype(np.float32)


def read_tci(safe_dir: str, level: str):
    """
    Đọc ảnh TCI (True Color Image) độ phân giải cao nhất.
    Trả về (rgb_uint8 [H,W,3], transform, crs). Nếu không có TCI, dựng từ B04/B03/B02.
    """
    tci_path = find_band_file(safe_dir, "TCI", res=10 if level == "L2A" else None)
    if tci_path:
        with rasterio.open(tci_path) as ds:
            rgb = ds.read([1, 2, 3])  # (3, H, W) uint8
            transform, crs = ds.transform, ds.crs
        return np.transpose(rgb, (1, 2, 0)).astype(np.uint8), transform, crs

    # Fallback: dựng TCI từ B04(R)/B03(G)/B02(B), scale kiểu quicklook.
    print("  [i] Không thấy TCI — dựng từ B04/B03/B02.")
    r_path = find_band_file(safe_dir, "B04", res=10 if level == "L2A" else None)
    g_path = find_band_file(safe_dir, "B03", res=10 if level == "L2A" else None)
    b_path = find_band_file(safe_dir, "B02", res=10 if level == "L2A" else None)
    if not (r_path and g_path and b_path):
        raise FileNotFoundError("Không tìm thấy TCI lẫn B02/B03/B04 trong .SAFE.")
    with rasterio.open(r_path) as ds:
        transform, crs = ds.transform, ds.crs
    r, g, b = read_band(r_path), read_band(g_path), read_band(b_path)
    stack = np.stack([r, g, b], axis=-1)
    rgb = np.clip(stack / 3000.0 * 255.0, 0, 255).astype(np.uint8)  # ~TOA quicklook
    return rgb, transform, crs


def _mask_shape_for_gsd(hw10m: tuple[int, int], mask_gsd: int) -> tuple[int, int]:
    """Kích thước lưới mask ở độ phân giải mask_gsd (m), suy từ lưới TCI 10m."""
    h, w = hw10m
    return max(1, round(h * 10 / mask_gsd)), max(1, round(w * 10 / mask_gsd))


# =========================================================================== #
# 3. Các bộ tạo mask mây
# =========================================================================== #
def cloud_mask_scl(safe_dir: str, out_hw: tuple[int, int], *,
                   mask_shadow: bool = True, mask_cirrus: bool = True,
                   mask_defect: bool = False) -> np.ndarray:
    """Mask mây từ band SCL của L2A. Trả về bool (H,W) khớp lưới TCI (True = loại)."""
    scl_path = find_band_file(safe_dir, "SCL", res=20) or find_band_file(safe_dir, "SCL", res=60)
    if not scl_path:
        raise FileNotFoundError(
            "Không thấy band SCL — sản phẩm này có phải L2A không? "
            "Với L1C hãy dùng --method omnicloudmask hoặc s2cloudless."
        )
    # SCL là nhãn phân loại -> resample nearest để không sinh giá trị lai.
    scl = read_band(scl_path, out_shape=out_hw, resampling=Resampling.nearest).astype(np.int16)
    remove = set(SCL_CLOUD)
    if mask_cirrus:
        remove |= SCL_CIRRUS
    if mask_shadow:
        remove |= SCL_SHADOW
    if mask_defect:
        remove |= SCL_DEFECT
    return np.isin(scl, list(remove))


def cloud_mask_omni(safe_dir: str, level: str, out_hw: tuple[int, int], *,
                    mask_gsd: int = 20, mask_shadow: bool = True, mask_thin: bool = True,
                    device: Optional[str] = None, patch_size: int = 1000,
                    patch_overlap: int = 300, batch_size: int = 1) -> np.ndarray:
    """
    Mask mây bằng OmniCloudMask (Red/Green/NIR). Trả về bool (H,W) khớp lưới TCI.
    Lớp OMM: 0 clear | 1 thick cloud | 2 thin cloud | 3 cloud shadow.
    """
    try:
        from omnicloudmask import predict_from_array
    except ImportError as e:
        raise ImportError(
            "Cần cài omnicloudmask: pip install omnicloudmask (kéo theo torch)."
        ) from e

    res = 10 if level == "L2A" else None  # L2A: lấy band 10m
    r_path = find_band_file(safe_dir, "B04", res=res)  # Red
    g_path = find_band_file(safe_dir, "B03", res=res)  # Green
    n_path = find_band_file(safe_dir, "B08", res=res)  # NIR
    if not (r_path and g_path and n_path):
        raise FileNotFoundError("Thiếu band B04/B03/B08 cho OmniCloudMask.")

    mh, mw = _mask_shape_for_gsd(out_hw, mask_gsd)
    red = read_band(r_path, out_shape=(mh, mw))
    green = read_band(g_path, out_shape=(mh, mw))
    nir = read_band(n_path, out_shape=(mh, mw))
    arr = np.stack([red, green, nir], axis=0).astype(np.float32)  # (3, mh, mw)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pred = predict_from_array(
        arr, patch_size=patch_size, patch_overlap=patch_overlap,
        batch_size=batch_size, inference_device=device, mosaic_device=device,
    )
    pred = np.squeeze(np.asarray(pred))  # (mh, mw) nhãn 0..3

    remove = {1}  # thick cloud
    if mask_thin:
        remove.add(2)  # thin cloud / cirrus
    if mask_shadow:
        remove.add(3)  # cloud shadow
    mask_small = np.isin(pred, list(remove))
    return _upsample_mask(mask_small, out_hw)


def cloud_mask_s2cloudless(safe_dir: str, level: str, out_hw: tuple[int, int], *,
                           mask_gsd: int = 60, threshold: float = 0.4,
                           average_over: int = 4, dilation_size: int = 2) -> np.ndarray:
    """
    Mask mây bằng s2cloudless (10 band TOA của L1C). Trả về bool (H,W) khớp lưới TCI.
    Lưu ý: s2cloudless KHÔNG phát hiện bóng mây (chỉ mây).
    """
    if level != "L1C":
        raise ValueError(
            "s2cloudless cần band cirrus B10 (chỉ có ở L1C). "
            "Với L2A hãy dùng --method scl hoặc omnicloudmask."
        )
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError as e:
        raise ImportError("Cần cài s2cloudless: pip install s2cloudless.") from e

    mh, mw = _mask_shape_for_gsd(out_hw, mask_gsd)
    bands = []
    for b in S2CLOUDLESS_BANDS:
        p = find_band_file(safe_dir, b)
        if not p:
            raise FileNotFoundError(f"Thiếu band {b} cần cho s2cloudless.")
        bands.append(read_band(p, out_shape=(mh, mw)) / 10000.0)  # DN -> reflectance
    stack = np.stack(bands, axis=-1)[np.newaxis, ...].astype(np.float32)  # (1, mh, mw, 10)

    detector = S2PixelCloudDetector(
        threshold=threshold, average_over=average_over,
        dilation_size=dilation_size, all_bands=False,
    )
    mask_small = detector.get_cloud_masks(stack)[0].astype(bool)  # (mh, mw)
    return _upsample_mask(mask_small, out_hw)


def _upsample_mask(mask_small: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Phóng mask nhị phân về lưới TCI bằng nearest (giữ biên cứng)."""
    if mask_small.shape == tuple(out_hw):
        return mask_small.astype(bool)
    im = Image.fromarray((mask_small.astype(np.uint8) * 255))
    im = im.resize((out_hw[1], out_hw[0]), resample=Image.NEAREST)
    return np.asarray(im) > 127


# =========================================================================== #
# 4. Hình thái học + áp mask
# =========================================================================== #
def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Giãn nở mask (3x3) `iterations` lần để phủ rìa mây. Không cần scipy."""
    if iterations <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=iterations)
    except Exception:
        out = mask.copy()
        for _ in range(iterations):
            p = np.pad(out, 1, mode="edge")
            out = (
                p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] |
                p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] |
                p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
            )
        return out


def apply_mask(rgb: np.ndarray, mask: np.ndarray, fill: int = 0) -> np.ndarray:
    """Đặt pixel mây về giá trị fill (mặc định 0 = đen)."""
    out = rgb.copy()
    out[mask] = fill
    return out


# =========================================================================== #
# 5. Ghi output
# =========================================================================== #
def save_geotiff(path: str, array: np.ndarray, transform, crs) -> None:
    """Ghi GeoTIFF giữ georeference. array: (H,W) hoặc (H,W,C)."""
    if array.ndim == 2:
        array = array[..., np.newaxis]
    h, w, c = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=c,
        dtype=array.dtype, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        for i in range(c):
            dst.write(array[..., i], i + 1)


def save_png(path: str, array: np.ndarray) -> None:
    """Ghi PNG. array uint8 (H,W) grayscale hoặc (H,W,3) RGB."""
    Image.fromarray(array).save(path)


# =========================================================================== #
# 6. Cắt tile cho pipeline YOLO
# =========================================================================== #
def tile_image(rgb: np.ndarray, mask: np.ndarray, out_dir: str, prefix: str, *,
               tile_size: int = 800, overlap: int = 0, skip_empty: bool = True,
               max_cloud_frac: float = 1.0) -> list[dict]:
    """
    Cắt ảnh đã che mây thành các tile vuông cho YOLO.

    skip_empty     : bỏ tile toàn đen (nodata + mây phủ hết).
    max_cloud_frac : bỏ tile có tỉ lệ mây > ngưỡng này (1.0 = không bỏ).
    Trả về danh sách metadata từng tile (để ánh xạ ngược toạ độ).
    """
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    H, W = rgb.shape[:2]
    step = max(1, tile_size - overlap)

    def _starts(size: int) -> list[int]:
        """Vị trí bắt đầu tile theo 1 trục, đảm bảo tile cuối áp sát mép scene."""
        pos = list(range(0, max(1, size - tile_size + 1), step))
        if not pos or pos[-1] + tile_size < size:
            pos.append(max(0, size - tile_size))
        return sorted(set(pos))

    index = []
    for y in _starts(H):
        for x in _starts(W):
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            crop = rgb[y:y2, x:x2]
            cmask = mask[y:y2, x:x2]
            # đệm về đúng tile_size nếu ở rìa
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                pad = np.zeros((tile_size, tile_size, 3), dtype=rgb.dtype)
                pad[: crop.shape[0], : crop.shape[1]] = crop
                crop = pad
                pmask = np.zeros((tile_size, tile_size), dtype=bool)
                pmask[: cmask.shape[0], : cmask.shape[1]] = cmask
                cmask = pmask
            cloud_frac = float(cmask.mean())
            if skip_empty and crop.max() == 0:
                continue
            if cloud_frac > max_cloud_frac:
                continue
            name = f"{prefix}_r{y}_c{x}.png"
            save_png(str(img_dir / name), crop)
            index.append({"file": f"images/{name}", "x": x, "y": y,
                          "w": tile_size, "h": tile_size, "cloud_frac": round(cloud_frac, 4)})
    with open(Path(out_dir) / "tiles_index.json", "w") as f:
        json.dump({"scene": prefix, "tile_size": tile_size, "overlap": overlap,
                   "scene_hw": [H, W], "tiles": index}, f, indent=2)
    return index


# =========================================================================== #
# 7. Orchestrator
# =========================================================================== #
def resolve_method(method: str, level: str) -> str:
    """Chọn phương pháp cụ thể khi method='auto'."""
    if method != "auto":
        return method
    if level == "L2A":
        return "scl"
    # L1C: ưu tiên omnicloudmask, fallback s2cloudless
    try:
        import omnicloudmask  # noqa: F401
        return "omnicloudmask"
    except ImportError:
        return "s2cloudless"


def process_safe(safe_dir: str, out_dir: str, *, method: str = "auto",
                 mask_gsd: Optional[int] = None, fill_value: int = 0,
                 dilation: int = 0, mask_shadow: bool = True, mask_cirrus: bool = True,
                 device: Optional[str] = None, tile_size: Optional[int] = None,
                 tile_overlap: int = 0, skip_empty: bool = True,
                 max_cloud_frac: float = 1.0, save_tif: bool = True) -> dict:
    """
    Xử lý trọn 1 folder .SAFE: đọc TCI -> tạo mask mây -> che mây -> lưu + (cắt tile).
    Trả về dict đường dẫn các output.
    """
    safe_dir = str(safe_dir)
    os.makedirs(out_dir, exist_ok=True)
    level = detect_level(safe_dir)
    method = resolve_method(method, level)
    sid = scene_id(safe_dir)
    print(f"[+] Scene {sid} | mức: {level} | phương pháp: {method}")

    rgb, transform, crs = read_tci(safe_dir, level)
    H, W = rgb.shape[:2]
    print(f"    TCI: {W}x{H}")

    if method == "scl":
        mask = cloud_mask_scl(safe_dir, (H, W), mask_shadow=mask_shadow, mask_cirrus=mask_cirrus)
    elif method == "omnicloudmask":
        mask = cloud_mask_omni(safe_dir, level, (H, W), mask_gsd=mask_gsd or 20,
                               mask_shadow=mask_shadow, mask_thin=mask_cirrus, device=device)
    elif method == "s2cloudless":
        mask = cloud_mask_s2cloudless(safe_dir, level, (H, W), mask_gsd=mask_gsd or 60)
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")

    mask = dilate_mask(mask, dilation)
    frac = float(mask.mean())
    print(f"    Tỉ lệ mây/bóng mây bị loại: {frac * 100:.2f}%")

    masked = apply_mask(rgb, mask, fill=fill_value)
    mask_u8 = (mask.astype(np.uint8) * 255)

    out = {"scene": sid, "level": level, "method": method, "cloud_frac": frac}
    png_masked = os.path.join(out_dir, f"{sid}_TCI_masked.png")
    png_mask = os.path.join(out_dir, f"{sid}_CLOUDMASK.png")
    save_png(png_masked, masked)
    save_png(png_mask, mask_u8)
    out["masked_tci_png"], out["mask_png"] = png_masked, png_mask
    if save_tif:
        tif_masked = os.path.join(out_dir, f"{sid}_TCI_masked.tif")
        tif_mask = os.path.join(out_dir, f"{sid}_CLOUDMASK.tif")
        save_geotiff(tif_masked, masked, transform, crs)
        save_geotiff(tif_mask, mask_u8, transform, crs)
        out["masked_tci_tif"], out["mask_tif"] = tif_masked, tif_mask

    if tile_size:
        tiles_dir = os.path.join(out_dir, "tiles")
        idx = tile_image(masked, mask, tiles_dir, sid, tile_size=tile_size,
                         overlap=tile_overlap, skip_empty=skip_empty,
                         max_cloud_frac=max_cloud_frac)
        out["tiles_dir"], out["n_tiles"] = tiles_dir, len(idx)
        print(f"    Đã cắt {len(idx)} tile {tile_size}x{tile_size} -> {tiles_dir}/images")

    print(f"[✓] Xong: {png_masked}")
    return out


# =========================================================================== #
# 8. CLI
# =========================================================================== #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Che mây cho ảnh Sentinel-2 (.SAFE, L1C/L2A) và xuất TCI đã che mây + mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--safe", required=True,
                   help="Đường dẫn 1 folder .SAFE, hoặc thư mục cha chứa nhiều .SAFE (dùng --batch).")
    p.add_argument("--out", required=True, help="Thư mục output.")
    p.add_argument("--method", default="auto",
                   choices=["auto", "scl", "omnicloudmask", "s2cloudless"],
                   help="Phương pháp tạo mask mây.")
    p.add_argument("--mask-gsd", type=int, default=None,
                   help="Độ phân giải (m) khi chạy model. Mặc định omni=20, s2cloudless=60.")
    p.add_argument("--fill", type=int, default=0, help="Giá trị pixel thay cho vùng mây (0=đen).")
    p.add_argument("--dilation", type=int, default=0, help="Số lần giãn nở mask (phủ rìa mây).")
    p.add_argument("--no-shadow", action="store_true", help="KHÔNG che bóng mây.")
    p.add_argument("--no-cirrus", action="store_true", help="KHÔNG che mây mỏng/cirrus/thin.")
    p.add_argument("--device", default=None, help="'cuda' | 'cpu' (mặc định tự dò).")
    p.add_argument("--tile", type=int, default=None, help="Cắt tile vuông kích thước này (vd 800).")
    p.add_argument("--tile-overlap", type=int, default=0, help="Độ chồng lấn giữa các tile (px).")
    p.add_argument("--keep-empty", action="store_true", help="Giữ cả tile toàn đen.")
    p.add_argument("--max-cloud-frac", type=float, default=1.0,
                   help="Bỏ tile có tỉ lệ mây > ngưỡng (1.0=giữ tất cả).")
    p.add_argument("--no-tif", action="store_true", help="Chỉ xuất PNG, bỏ GeoTIFF.")
    p.add_argument("--batch", action="store_true",
                   help="Coi --safe là thư mục cha, xử lý mọi *.SAFE bên trong.")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    common = dict(
        method=args.method, mask_gsd=args.mask_gsd, fill_value=args.fill,
        dilation=args.dilation, mask_shadow=not args.no_shadow,
        mask_cirrus=not args.no_cirrus, device=args.device, tile_size=args.tile,
        tile_overlap=args.tile_overlap, skip_empty=not args.keep_empty,
        max_cloud_frac=args.max_cloud_frac, save_tif=not args.no_tif,
    )
    if args.batch:
        safes = sorted(glob.glob(os.path.join(args.safe, "*.SAFE")))
        if not safes:
            print(f"[!] Không thấy .SAFE nào trong {args.safe}")
            return 1
        for s in safes:
            process_safe(s, os.path.join(args.out, scene_id(s)), **common)
    else:
        process_safe(args.safe, args.out, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
