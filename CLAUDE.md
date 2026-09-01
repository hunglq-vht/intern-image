# Context dự án — phát hiện tàu trên ảnh Sentinel-2

## Hướng làm việc chính (QUAN TRỌNG)

**Mục tiêu chính hiện tại: fine-tune tiếp model YOLO11**, khởi tạo từ checkpoint
HuggingFace `mayrajeo/marine-vessel-yolo` → `yolo11s_tci.pt`.

Notebook chuẩn là **`finetune_yolo11_ship_detection.ipynb` (bản Kaggle)** — mọi
định dạng dữ liệu, siêu tham số và quy ước đặt tên đều lấy theo file này.
`finetune_yolo11_ship_detection_colab.ipynb` chỉ là bản port sang Colab.

`finetune_internimage_ship_detection.ipynb` (Faster R-CNN từ InternImage,
định dạng COCO) là **nhánh phụ, không phải hướng chính**. Đừng mặc định đề xuất
COCO/mmdetection; chỉ nhắc tới khi người dùng hỏi thẳng về InternImage.

## Định dạng dữ liệu chuẩn (YOLO)

```
<dataset_root>/
  train/images/*.png   train/labels/*.txt
  valid/images/*.png   valid/labels/*.txt
  test/images/*.png    test/labels/*.txt
```

- Nhãn YOLO: `class cx cy w h` (đã chuẩn hoá 0–1), **`single_cls=True`**, mọi class ép về `0`.
- Tên lớp: **`vessel`** (`CFG["CLASS_NAME"]`).
- Ảnh: tile **800×800 px**, RGB uint8, cắt từ Sentinel-2 **L2A TCI** (GSD 10 m → 1 tile = 8 km).
- Bộ hiện có: ~2000 tile, vốn export từ Roboflow (nên có thể còn lẫn di sản COCO
  `_annotations.coco.json` với 2 category trùng tên — xem mục Cạm bẫy).

## Siêu tham số fine-tune đang dùng

`IMG_SIZE=800`, `EPOCHS=100`, `PATIENCE=20`, `BATCH=-1` (auto ~60% VRAM),
`OPTIMIZER="auto"` (→ AdamW), `LR0=0.001`, `SCALE=0.25`, `FLIPUD=0.5`,
`FLIPLR=0.5`, `FREEZE=0`, `SEED=42` — theo repo `mayrajeo/ship-detection`.
Kết quả: `/kaggle/working/finetune_outputs/finetune/weights/best.pt`.

Khi fine-tune **tiếp** (thêm dữ liệu mới): khởi tạo từ `best.pt` hiện có, KHÔNG
phải từ `yolo11s_tci.pt`; hạ lr xuống ~1/5–1/10; chạy ít epoch.

## Suy luận

`infer_drive_folder_ship_detection.ipynb` — chạy trên ảnh tải từ Google Drive.
Cắt tile `TILE=320`, `OVERLAP=64`, `IMGSZ=320`, `CONF=0.1`. GSD giữ nguyên nên
không lệch scale so với lúc train, nhưng **kích thước tile lúc infer (320) khác
lúc train (800)** — ngữ cảnh quanh mỗi tàu hẹp hơn.

Xuất ra: `{stem}_pred.geojson/.gpkg` (sau lọc mây), `{stem}_pred_raw.*` (trước
lọc mây), crops zip, hard-negative tiles. Có tầng lọc mây nhiều nguồn
(SCL / MSK_CLDPRB / OmniCloudMask / B10 cirrus) với `CLOUD_PROTECT_CONF=0.5`.

## Công cụ gán nhãn tay

`labeling/qgis_labels_to_dataset.py` (`grid` → `export` → `merge`) +
`labeling/README.md`. Xuất **YOLO là chính**, COCO chỉ là tuỳ chọn `--format`.
Nguyên tắc: chỉ xuất các ô đã soát nhãn ĐẦY ĐỦ (`status=done`), vì mọi pixel
ngoài box đều bị coi là background.

## Ảnh & CRS

Tile MGRS **51RUM** (Đông Á) → **EPSG:32651** (WGS 84 / UTM zone 51N, đơn vị mét).
Mức xử lý: **L2A**. File `.gpkg` do notebook xuất giữ CRS gốc; file `.geojson`
luôn bị chuyển về EPSG:4326.

## Cạm bẫy đã biết

- **Tập train gần như không có mây** → model bắt nhầm mây thành tàu; tầng lọc mây
  lúc infer là bản vá cho lỗ hổng dữ liệu này, và nó cũng ăn cả tàu thật.
- **Nhãn gốc do người nhìn TCI tạo ra** → tàu quá mờ chưa bao giờ được gán nhãn,
  tức model đã được dạy bỏ qua chúng. Nhãn thiếu, không phải model yếu.
- **TCI 8-bit gain cố định** (DN ≈ reflectance × 1000, bão hoà ở 0,255): nước L2A
  chỉ còn DN 10–30, tàu mờ chênh vài DN → dễ mất khi lượng tử hoá.
- Export Roboflow có **2 category trùng tên `ship`** (id 0 rỗng / id 1 thật) —
  chỉ ảnh hưởng nhánh COCO, `labeling/` đã xử lý.
- `_to_uint8` trong notebook infer chia cho **max của mảng**, nên ảnh burn ra từ
  nguồn không phải uint8 có thể mất hết tương phản.

## Ngôn ngữ

Người dùng trao đổi bằng **tiếng Việt**; comment và tài liệu trong repo cũng viết
bằng tiếng Việt.
