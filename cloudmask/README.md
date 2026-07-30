# Che mây ảnh Sentinel-2 (`cloudmask/`)

Module đọc **folder `.SAFE` gốc từ Copernicus** (cả **L1C** và **L2A**, 12–13 band + ảnh
TCI), tạo **mask mây + bóng mây**, rồi **che mây vào ảnh TCI** và (tuỳ chọn) **cắt tile
800×800** để nạp thẳng vào pipeline phát hiện tàu YOLO11 trong repo này.

> Hướng tiếp cận là **che mây (masking)**, *không* tái tạo pixel dưới mây
> (reconstruction). Với bài toán phát hiện tàu, điều này tránh việc model nhận nhầm mây
> trắng thành tàu và tránh dự đoán trên vùng bị mây che — mà không "bịa" ra tàu giả như
> các phương pháp inpainting/GAN.

## Phương pháp (`--method`)

| Phương pháp | Dùng cho | Cần model | Ghi chú |
|---|---|---|---|
| `scl` | **L2A** | Không | Đọc band `SCL` có sẵn trong sản phẩm L2A. Nhanh nhất. |
| `omnicloudmask` | L1C & L2A | Có (torch) | Deep learning SOTA 2025, chỉ cần Red/Green/NIR. Phát hiện cả bóng mây. |
| `s2cloudless` | **L1C** | Có | ML single-scene (cần band cirrus B10 → không dùng cho L2A). Không có bóng mây. |
| `auto` | tự chọn | — | L2A → `scl`; L1C → `omnicloudmask` (fallback `s2cloudless`). |

Lớp bị loại (mặc định): **mây dày + mây mỏng/cirrus + bóng mây**. Tắt bằng `--no-cirrus`
và/hoặc `--no-shadow`.

## Cài đặt

```bash
pip install -r cloudmask/requirements.txt
# tối thiểu: pip install rasterio numpy pillow
# thêm 1 trong 2 tuỳ phương pháp: pip install omnicloudmask   (hoặc)   pip install s2cloudless
```

## Dùng bằng dòng lệnh (CLI)

```bash
# L2A: dùng SCL, cắt tile 800 cho YOLO, bỏ tile toàn đen
python cloudmask/s2_cloud_mask.py \
    --safe /data/S2B_MSIL2A_20230101T..._T33UXP_....SAFE \
    --out  out/scene1 --method auto --tile 800 --skip-empty

# L1C: dùng OmniCloudMask ở 20 m cho nhanh, giãn nở mask 1 vòng để phủ rìa mây
python cloudmask/s2_cloud_mask.py \
    --safe /data/S2A_MSIL1C_..._.SAFE \
    --out  out/scene2 --method omnicloudmask --mask-gsd 20 --dilation 1 --tile 800

# Xử lý cả thư mục chứa nhiều .SAFE
python cloudmask/s2_cloud_mask.py --safe /data/all_safe --batch --out out/ --tile 800
```

Tham số hay dùng:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--method` | `auto` | `auto` / `scl` / `omnicloudmask` / `s2cloudless` |
| `--mask-gsd` | omni=20, s2cloudless=60 | Độ phân giải (m) chạy model (nhỏ hơn = chính xác hơn nhưng chậm) |
| `--tile` | (tắt) | Kích thước tile vuông xuất ra (vd `800`) |
| `--tile-overlap` | 0 | Chồng lấn giữa các tile (px) |
| `--dilation` | 0 | Số vòng giãn nở mask để phủ rìa mây |
| `--no-shadow` / `--no-cirrus` | tắt | Không che bóng mây / mây mỏng |
| `--skip-empty` | tắt | Bỏ tile toàn đen (nodata + mây phủ hết) |
| `--max-cloud-frac` | 1.0 | Bỏ tile có tỉ lệ mây vượt ngưỡng |
| `--fill` | 0 | Giá trị pixel thay cho vùng mây (0 = đen) |
| `--no-tif` | tắt | Chỉ xuất PNG, bỏ GeoTIFF |
| `--batch` | tắt | Coi `--safe` là thư mục cha, xử lý mọi `*.SAFE` |

## Dùng như thư viện

```python
from cloudmask.s2_cloud_mask import process_safe

res = process_safe(
    "S2A_..._.SAFE", "out_dir",
    method="auto", tile_size=800, tile_overlap=0,
    dilation=1, skip_empty=True, max_cloud_frac=0.98,
)
print(res["masked_tci_png"], res["mask_png"], res["tiles_dir"], res["cloud_frac"])
```

## Đầu ra

```
out_dir/
├── <scene>_TCI_masked.tif      # TCI đã che mây (GeoTIFF, giữ toạ độ)
├── <scene>_TCI_masked.png      # TCI đã che mây (PNG để xem nhanh)
├── <scene>_CLOUDMASK.tif       # mask nhị phân (255 = mây/bóng mây bị loại)
├── <scene>_CLOUDMASK.png
└── tiles/                      # chỉ khi có --tile
    ├── images/<scene>_r{y}_c{x}.png   # tile vuông đã che mây, sẵn cho YOLO
    └── tiles_index.json               # offset pixel + tỉ lệ mây từng tile
```

`tiles_index.json` cho phép **quy toạ độ box của YOLO trên từng tile về pixel của scene
gốc** — xem cách dùng trong Section 7 (deploy) của `test_yolo11_ship_detection.ipynb`.

## Tích hợp với các notebook YOLO11

| Notebook | Phần thêm | Tác dụng |
|---|---|---|
| `test_yolo11_ship_detection.ipynb` | **Section 7** (tuỳ chọn) | Che mây 1 scene `.SAFE` mới → cắt tile → YOLO11 predict → JSON toạ độ tàu toàn scene. |
| `finetune_yolo11_ship_detection.ipynb` | **Section 4b** (tuỳ chọn) | Che mây + cắt tile từ `.SAFE` để bổ sung vào tập train/val. |
| `finetune_yolo11_ship_detection_colab.ipynb` | **Section 4b** (tuỳ chọn) | Như trên, bản Google Colab. |

Các phần này **mặc định tắt** (`RUN_DEPLOY=False` / `PREP_FROM_SAFE=False`) nên không ảnh
hưởng luồng chạy sẵn có; bật cờ và trỏ đường dẫn `.SAFE` để dùng. Notebook tự `pip install`
deps và tải `s2_cloud_mask.py` từ repo.

> Ghi chú: notebook `finetune_internimage_ship_detection.ipynb` **không** được đụng tới,
> đúng theo yêu cầu.

## Về độ phân giải & tốc độ

- `scl` chạy tức thì (chỉ đọc + resample 1 band nhãn).
- `omnicloudmask` mặc định chạy ở **20 m** (giảm ~4× bộ nhớ/thời gian so với 10 m) rồi
  phóng mask về lưới TCI 10 m; đặt `--mask-gsd 10` nếu cần biên mây sắc hơn.
- `s2cloudless` mặc định **60 m** (đúng thiết kế của thuật toán) rồi phóng mask lên 10 m.
