"""Biến nhãn tàu vẽ tay trong QGIS thành dữ liệu train/val/test cho fine-tune.

Quy trình (xem labeling/README.md để có hướng dẫn QGIS chi tiết):

    1) grid   - sinh lưới ô 800x800 px (bám đúng lưới pixel của ảnh) để biết
                vùng nào đã được soát nhãn ĐẦY ĐỦ. Mở lưới này trong QGIS,
                sửa cột `status` thành `done` cho từng ô đã soát xong.
    2) export - cắt các ô `done` thành ảnh 800x800 + nhãn (YOLO .txt cho
                YOLO11 — hướng chính; kèm cả COCO cho InternImage) từ layer
                nhãn đã sửa tay.
    3) merge  - trộn các ô mới vào bộ dữ liệu train/valid/test hiện có
                (ghi ra thư mục MỚI, không sửa dữ liệu gốc).

Phụ thuộc: rasterio, geopandas, shapely, numpy, pillow (giống notebook infer).
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
RASTER_EXTS = (".jp2", ".tif", ".tiff", ".png", ".jpg")


# ----------------------------------------------------------------------------
# Đọc ảnh: giữ đúng cách chuyển RGB/uint8 của notebook infer để radiometry của
# tile mới khớp với tile cũ trong bộ train.
# ----------------------------------------------------------------------------
def _band_idx(nbands):
    return [1, 2, 3] if nbands >= 3 else [1, 1, 1]


def _to_uint8(arr, scale=None):
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    denom = float(scale) if scale else (float(a.max()) or 1.0)
    return np.clip(a / denom * 255.0, 0, 255).astype(np.uint8)


def _raster_scale(src):
    """Hệ số chuẩn hoá tính 1 lần trên TOÀN ảnh (ảnh TCI uint8 thì bỏ qua).

    Nếu chuẩn hoá theo từng cửa sổ như lúc infer, mỗi tile sẽ có độ sáng khác
    nhau -> nhiễu cho việc train, nên ở đây dùng chung 1 hệ số cho cả ảnh.
    """
    if src.dtypes[0] == "uint8":
        return None
    ov = src.read(indexes=_band_idx(src.count), out_shape=(3, 512, 512))
    return float(np.percentile(ov, 99.9)) or 1.0


def _read_rgb_window(src, window, scale=None):
    import rasterio  # noqa: F401  (đảm bảo lỗi phụ thuộc hiện sớm)
    arr = src.read(indexes=_band_idx(src.count), window=window,
                   boundless=True, fill_value=0)
    return _to_uint8(np.transpose(arr, (1, 2, 0)), scale)


def _tile_offsets(size, tile, stride, snap_last=True):
    if size <= tile:
        return [0]
    offs = list(range(0, size - tile + 1, stride))
    if snap_last and offs[-1] != size - tile:
        # Ô cuối được kéo về sát mép -> phủ kín ảnh nhưng CHỒNG LÊN ô trước.
        # Dùng --no-snap-last nếu muốn các ô rời hẳn nhau (tránh rò rỉ dữ liệu
        # giữa train và val/test khi chia ngẫu nhiên theo ô).
        offs.append(size - tile)
    return offs


def _list_rasters(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in RASTER_EXTS)
        else:
            out.append(p)
    return out


def _write_vector(gdf, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = "GPKG" if path.suffix.lower() == ".gpkg" else "GeoJSON"
    if driver == "GeoJSON" and gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        # .geojson chuẩn RFC7946 luôn là WGS84; .gpkg giữ nguyên CRS gốc.
        gdf = gdf.to_crs(4326)
    gdf.to_file(path, driver=driver)
    return path


# ----------------------------------------------------------------------------
# 1) grid
# ----------------------------------------------------------------------------
def cmd_grid(args):
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box as shp_box

    near = gpd.read_file(args.near) if args.near else None
    rows, geoms = [], []

    for rp in _list_rasters(args.raster):
        with rasterio.open(rp) as src:
            W, H, T, crs = src.width, src.height, src.transform, src.crs
        stem = Path(rp).stem
        stride = args.stride or args.tile
        near_r = None
        if near is not None:
            near_r = near.to_crs(crs) if (crs and near.crs and str(near.crs) != str(crs)) else near
            if args.buffer:
                near_r = near_r.copy()
                near_r["geometry"] = near_r.geometry.buffer(args.buffer)

        for oy in _tile_offsets(H, args.tile, stride, args.snap_last):
            for ox in _tile_offsets(W, args.tile, stride, args.snap_last):
                tw, th = min(args.tile, W - ox), min(args.tile, H - oy)
                x0, y0 = T * (ox, oy)
                x1, y1 = T * (ox + tw, oy + th)
                poly = shp_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                n_near = 0
                if near_r is not None:
                    n_near = int(near_r.geometry.intersects(poly).sum())
                    if n_near == 0 and not args.keep_empty:
                        continue
                rows.append(dict(tile_id=f"{stem}_x{ox}_y{oy}", raster=stem,
                                 ox=ox, oy=oy, w=tw, h=th, n_pred=n_near,
                                 status="todo", note=""))
                geoms.append(poly)
        print(f"{stem}: {W}x{H}px -> {len(rows)} ô luỹ kế")

    if not rows:
        raise SystemExit("Không sinh được ô nào (thử bỏ --near hoặc thêm --keep-empty).")
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)
    out = _write_vector(gdf, args.out)
    print(f"\n{len(gdf)} ô -> {out}\n"
          f"Mở trong QGIS, soát nhãn từng ô rồi đặt status='done' cho ô đã soát XONG.")


# ----------------------------------------------------------------------------
# 2) export
# ----------------------------------------------------------------------------
def _boxes_in_pixels(gdf, src, point_box_m):
    """Đổi geometry (đa giác/điểm) sang bbox toạ độ pixel của ảnh."""
    from shapely.geometry import Point

    inv = ~src.transform
    out = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Point):
            half = point_box_m / 2.0
            minx, miny, maxx, maxy = (geom.x - half, geom.y - half,
                                      geom.x + half, geom.y + half)
        else:
            minx, miny, maxx, maxy = geom.bounds
        cols, rows = zip(*(inv * (x, y) for x, y in
                           ((minx, miny), (maxx, maxy))))
        out.append([min(cols), min(rows), max(cols), max(rows)])
    return np.asarray(out, np.float64).reshape(-1, 4)


def cmd_export(args):
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    from PIL import Image, ImageDraw

    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out / "qc"
    if args.qc:
        qc_dir.mkdir(parents=True, exist_ok=True)

    labels_all = gpd.read_file(args.labels)
    tiles_all = gpd.read_file(args.tiles)
    if args.status_field in tiles_all.columns:
        keep = tiles_all[args.status_field].astype(str).str.strip().str.lower()
        tiles_all = tiles_all[keep == args.status_value.lower()]
    print(f"{len(tiles_all)} ô đã soát ({args.status_field}={args.status_value}), "
          f"{len(labels_all)} nhãn đầu vào")
    if tiles_all.empty:
        raise SystemExit("Không có ô nào ở trạng thái 'done' — đánh dấu trong QGIS trước.")

    coco = dict(images=[], annotations=[],
                categories=[dict(id=1, name=args.class_name, supercategory="none")])
    index, n_ann, n_empty = [], 0, 0

    for rp in _list_rasters(args.raster):
        stem = Path(rp).stem
        with rasterio.open(rp) as src:
            W, H, crs, T = src.width, src.height, src.crs, src.transform
            scale = _raster_scale(src)

            tiles = tiles_all
            if "raster" in tiles.columns:
                tiles = tiles[tiles["raster"].astype(str) == stem]
            if tiles.empty:
                continue
            if crs and tiles.crs and str(tiles.crs) != str(crs):
                tiles = tiles.to_crs(crs)
            labels = labels_all
            if crs and labels.crs and str(labels.crs) != str(crs):
                labels = labels.to_crs(crs)
            boxes = _boxes_in_pixels(labels, src, args.point_box_m)

            for _, t in tiles.iterrows():
                minx, miny, maxx, maxy = t.geometry.bounds
                inv = ~T
                c0, r0 = inv * (minx, maxy)
                c1, r1 = inv * (maxx, miny)
                ox, oy = int(round(min(c0, c1))), int(round(min(r0, r1)))
                tw = int(round(abs(c1 - c0))) or args.tile
                th = int(round(abs(r1 - r0))) or args.tile
                ox, oy = max(0, min(ox, W - 1)), max(0, min(oy, H - 1))
                tw, th = min(tw, W - ox), min(th, H - oy)
                if tw < args.min_tile or th < args.min_tile:
                    print(f"  bỏ ô {ox},{oy}: {tw}x{th}px nhỏ hơn --min-tile")
                    continue

                sel = np.zeros(len(boxes), bool)
                if len(boxes):
                    sel = ((boxes[:, 0] < ox + tw) & (boxes[:, 2] > ox) &
                           (boxes[:, 1] < oy + th) & (boxes[:, 3] > oy))
                b = boxes[sel].copy()
                b[:, [0, 2]] = np.clip(b[:, [0, 2]] - ox, 0, tw)
                b[:, [1, 3]] = np.clip(b[:, [1, 3]] - oy, 0, th)
                wh = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) if len(b) else np.zeros((0, 2))
                b = b[(wh >= args.min_box_px).all(1)] if len(b) else b
                if len(b) == 0 and not args.keep_empty:
                    continue

                name = f"{stem}_x{ox}_y{oy}"
                crop = _read_rgb_window(src, Window(ox, oy, tw, th), scale)
                Image.fromarray(crop).save(img_dir / f"{name}.png")

                with open(lbl_dir / f"{name}.txt", "w") as f:  # YOLO: cx cy w h chuẩn hoá
                    for x1, y1, x2, y2 in b:
                        f.write(f"0 {(x1 + x2) / 2 / tw:.6f} {(y1 + y2) / 2 / th:.6f} "
                                f"{(x2 - x1) / tw:.6f} {(y2 - y1) / th:.6f}\n")

                img_id = len(coco["images"]) + 1
                coco["images"].append(dict(id=img_id, file_name=f"{name}.png",
                                           width=tw, height=th))
                for x1, y1, x2, y2 in b:  # COCO: [x, y, w, h] theo pixel
                    n_ann += 1
                    coco["annotations"].append(dict(
                        id=n_ann, image_id=img_id, category_id=1, iscrowd=0,
                        bbox=[round(float(x1), 2), round(float(y1), 2),
                              round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                        area=round(float((x2 - x1) * (y2 - y1)), 2)))

                gx, gy = T * (ox, oy)
                index.append(dict(file_name=f"{name}.png", raster=str(rp), ox=ox, oy=oy,
                                  w=tw, h=th, n_box=len(b), crs=str(crs),
                                  origin_x=round(gx, 2), origin_y=round(gy, 2),
                                  res=round(abs(T.a), 4)))
                n_empty += int(len(b) == 0)

                if args.qc:
                    im = Image.fromarray(crop).convert("RGB")
                    d = ImageDraw.Draw(im)
                    for x1, y1, x2, y2 in b:
                        d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(255, 0, 0))
                    im.save(qc_dir / f"{name}.png")

    with open(out / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    with open(out / "tiles_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()) if index else ["file_name"])
        w.writeheader()
        w.writerows(index)

    n_img = len(coco["images"])
    print(f"\n{n_img} tile ({n_empty} tile rỗng = hard-negative) / {n_ann} nhãn -> {out}")
    if n_img:
        sizes = [min(a["bbox"][2], a["bbox"][3]) for a in coco["annotations"]]
        if sizes:
            print("cạnh nhỏ nhất của box (px): "
                  + "  ".join(f"p{p}={np.percentile(sizes, p):.1f}" for p in (0, 50, 95, 100)))
        print(f"Kiểm tra bằng mắt: {qc_dir}" if args.qc else "Thêm --qc để xuất ảnh kiểm tra.")


# ----------------------------------------------------------------------------
# 3) merge
# ----------------------------------------------------------------------------
def _split_names(names, ratios, seed, group_by):
    rng = random.Random(seed)
    if group_by == "scene":
        groups = {}
        for n in names:
            groups.setdefault(n.rsplit("_x", 1)[0], []).append(n)
        units = list(groups.values())
    else:
        units = [[n] for n in names]
    rng.shuffle(units)
    n = len(units)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    n_tr = min(n_tr, n)
    n_va = min(n_va, n - n_tr)
    parts = dict(train=units[:n_tr], valid=units[n_tr:n_tr + n_va], test=units[n_tr + n_va:])
    return {k: [n for u in v for n in u] for k, v in parts.items()}


def _merge_coco(dst_json, new_coco, keep_names, class_name):
    if dst_json.exists():
        with open(dst_json) as f:
            coco = json.load(f)
    else:
        coco = dict(images=[], annotations=[],
                    categories=[dict(id=1, name=class_name, supercategory="none")])
    # Chọn category ĐANG ĐƯỢC DÙNG bởi annotation. Export Roboflow thường có 2
    # category trùng tên "ship" (id=0 là placeholder rỗng, id=1 mới là thật);
    # gán nhầm id=0 sẽ khiến nhãn mới không khớp nhãn cũ -> mAP tụt.
    used = Counter(a["category_id"] for a in coco["annotations"])
    named = [c["id"] for c in coco["categories"] if c["name"] == class_name]
    if used:
        cat_id = max((i for i in used if not named or i in named), key=lambda i: used[i])
    else:
        cat_id = named[0] if named else 1
    next_img = max([im["id"] for im in coco["images"]], default=0) + 1
    next_ann = max([a["id"] for a in coco["annotations"]], default=0) + 1

    by_img = {}
    for a in new_coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    added_i = added_a = 0
    for im in new_coco["images"]:
        if im["file_name"] not in keep_names:
            continue
        new_id = next_img
        next_img += 1
        added_i += 1
        coco["images"].append(dict(im, id=new_id))
        for a in by_img.get(im["id"], []):
            coco["annotations"].append(dict(a, id=next_ann, image_id=new_id,
                                            category_id=cat_id))
            next_ann += 1
            added_a += 1
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(coco, f)
    return added_i, added_a, len(coco["images"]), len(coco["annotations"])


def cmd_merge(args):
    new_dir, out = Path(args.new), Path(args.out)
    with open(new_dir / "_annotations.coco.json") as f:
        new_coco = json.load(f)
    names = [im["file_name"] for im in new_coco["images"]]
    if not names:
        raise SystemExit(f"{new_dir} chưa có tile nào.")

    if args.all_to:
        parts = {s: (names if s == args.all_to else []) for s in ("train", "valid", "test")}
    else:
        parts = _split_names(names, args.ratios, args.seed, args.group_by)

    if args.dataset:
        src = Path(args.dataset)
        if out.exists() and args.force:
            shutil.rmtree(out)
        print(f"Sao chép bộ dữ liệu gốc {src} -> {out} (bản gốc không bị sửa)")
        shutil.copytree(src, out, dirs_exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)

    for split, split_names in parts.items():
        if not split_names:
            continue
        keep = set(split_names)
        # Bám theo cách bố trí sẵn có của bộ dữ liệu: Roboflow-COCO để ảnh ngay
        # trong {split}/, còn bố cục YOLO để trong {split}/images/.
        if (out / split / "images").is_dir():
            img_dst = out / split / "images"
        elif (out / split / "_annotations.coco.json").exists():
            img_dst = out / split
        else:
            img_dst = (out / split / "images") if args.format in ("yolo", "both") else (out / split)
        img_dst.mkdir(parents=True, exist_ok=True)
        for n in split_names:
            shutil.copy(new_dir / "images" / n, img_dst / n)
        if args.format in ("yolo", "both"):
            lbl_dst = out / split / "labels"
            lbl_dst.mkdir(parents=True, exist_ok=True)
            for n in split_names:
                shutil.copy(new_dir / "labels" / (Path(n).stem + ".txt"),
                            lbl_dst / (Path(n).stem + ".txt"))
        if args.format in ("coco", "both"):
            dst_json = img_dst / "_annotations.coco.json"      # json nằm cạnh ảnh
            if not dst_json.exists() and (out / split / "_annotations.coco.json").exists():
                dst_json = out / split / "_annotations.coco.json"  # json cũ ở gốc split
            ai, aa, ti, ta = _merge_coco(dst_json, new_coco, keep, args.class_name)
            print(f"{split:5s}: +{ai} ảnh / +{aa} nhãn -> tổng {ti} ảnh / {ta} nhãn ({dst_json})")
        else:
            print(f"{split:5s}: +{len(split_names)} ảnh -> {img_dst}")
    print(f"\nXong. Trỏ CFG của notebook fine-tune vào: {out}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="sinh lưới ô cần soát nhãn")
    g.add_argument("--raster", nargs="+", required=True, help="ảnh TCI (.jp2/.tif) hoặc thư mục")
    g.add_argument("--out", required=True, help="tiles.gpkg (khuyến nghị) hoặc .geojson")
    g.add_argument("--tile", type=int, default=800, help="cạnh ô, px (mặc định 800 = IMG_SIZE lúc train)")
    g.add_argument("--stride", type=int, default=None, help="bước trượt, px (mặc định = --tile, ô không chồng nhau)")
    g.add_argument("--near", default=None, help="vector (vd *_pred.geojson) — chỉ giữ ô có giao với nó")
    g.add_argument("--buffer", type=float, default=0.0, help="nới --near thêm N mét trước khi lọc")
    g.add_argument("--keep-empty", action="store_true", help="giữ cả ô không giao với --near")
    g.add_argument("--no-snap-last", dest="snap_last", action="store_false",
                   help="không kéo ô cuối sát mép ảnh (các ô rời hẳn nhau, không chồng lấn)")
    g.set_defaults(func=cmd_grid)

    e = sub.add_parser("export", help="cắt ô đã soát thành ảnh + nhãn COCO/YOLO")
    e.add_argument("--raster", nargs="+", required=True)
    e.add_argument("--labels", required=True, help="layer nhãn đã sửa tay trong QGIS (.gpkg/.geojson)")
    e.add_argument("--tiles", required=True, help="lưới ô từ lệnh grid, đã đánh dấu status=done")
    e.add_argument("--out", required=True, help="thư mục xuất tile mới")
    e.add_argument("--tile", type=int, default=800)
    e.add_argument("--min-tile", type=int, default=256, help="bỏ ô rìa nhỏ hơn ngưỡng này")
    e.add_argument("--min-box-px", type=float, default=2.0, help="bỏ box (sau khi cắt) nhỏ hơn ngưỡng này")
    e.add_argument("--point-box-m", type=float, default=60.0,
                   help="nếu nhãn là ĐIỂM: tạo box vuông cạnh N mét (60m = 6px ở GSD 10m)")
    e.add_argument("--class-name", default="vessel",
                   help="ten lop (khop CLASS_NAME cua notebook YOLO); chi dung cho ban COCO")
    e.add_argument("--status-field", default="status")
    e.add_argument("--status-value", default="done")
    e.add_argument("--keep-empty", action="store_true", default=True,
                   help="giữ ô không có tàu làm hard-negative (mặc định bật)")
    e.add_argument("--no-keep-empty", dest="keep_empty", action="store_false")
    e.add_argument("--qc", action="store_true", help="xuất thêm ảnh có vẽ box để kiểm tra bằng mắt")
    e.set_defaults(func=cmd_export)

    m = sub.add_parser("merge", help="trộn tile mới vào bộ train/valid/test")
    m.add_argument("--new", required=True, help="thư mục kết quả của lệnh export")
    m.add_argument("--dataset", default=None, help="bộ dữ liệu hiện có (chỉ ĐỌC, sẽ được sao chép)")
    m.add_argument("--out", required=True, help="thư mục bộ dữ liệu mới")
    m.add_argument("--format", choices=("coco", "yolo", "both"), default="yolo",
                   help="yolo (mac dinh, huong chinh) | coco (InternImage) | both")
    m.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15),
                   metavar=("TRAIN", "VALID", "TEST"))
    m.add_argument("--all-to", choices=("train", "valid", "test"), default=None,
                   help="dồn toàn bộ tile mới vào 1 split (giữ test cũ nguyên vẹn để so sánh)")
    m.add_argument("--group-by", choices=("none", "scene"), default="none",
                   help="'scene': mọi ô của cùng 1 ảnh gốc vào cùng 1 split")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--class-name", default="vessel")
    m.add_argument("--force", action="store_true", help="xoá --out nếu đã tồn tại")
    m.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
