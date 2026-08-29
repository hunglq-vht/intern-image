# Gán nhãn tay trong QGIS → thêm vào train/val/test → fine-tune tiếp

Tình huống: chạy model trên ảnh Sentinel-2, xuất `{stem}_pred.geojson`, mở trong QGIS và
thấy **tàu bị bỏ sót** (false negative). Mục tiêu: sửa/bổ sung nhãn bằng tay rồi nạp
những vùng đó vào bộ dữ liệu để fine-tune tiếp.

Toàn bộ phần cắt ảnh + sinh nhãn được gói trong `labeling/qgis_labels_to_dataset.py`
(3 lệnh: `grid` → `export` → `merge`).

---

## 0. Nguyên tắc quan trọng nhất — "nhãn đầy đủ trong vùng đã chọn"

Detector học theo kiểu: **mọi pixel không nằm trong box đều là nền (background)**.
Nếu chỉ vẽ thêm mấy con tàu bị sót rồi cắt ảnh xung quanh, thì những con tàu **model
đã bắt đúng** trong cùng ảnh đó — nếu không có nhãn — sẽ thành "nền", và bạn đang dạy
model **thôi đừng phát hiện tàu**. Kết quả tệ hơn trước khi thêm dữ liệu.

Vì vậy quy trình luôn xoay quanh **ô (tile)**:

> Chọn một ô 800×800 px → soát **toàn bộ** tàu trong ô đó (giữ nhãn đúng, sửa nhãn lệch,
> xoá nhãn sai, thêm nhãn thiếu) → đánh dấu ô đó `status = done` → chỉ ô `done` mới được
> cắt vào dataset.

Ô nào chưa soát hết thì để `todo` — nó sẽ không bị xuất, và như vậy không gây hại.

Hệ quả tích cực: một ô đã soát mà **không có tàu nào** (ví dụ vùng mây làm model bắn
nhầm) vẫn rất giá trị — nó là **hard negative**, script vẫn xuất nó kèm nhãn rỗng.

Ô 800×800 px vì `IMG_SIZE = 800` trong `finetune_internimage_ship_detection.ipynb`, và
bộ ~2000 ảnh hiện có cũng là tile 800×800 (TCI 10 m → 8 km/ô). Giữ đúng kích thước để
scale của tàu (4–18 px) trong dữ liệu mới **khớp** dữ liệu cũ.

---

## 1. Chuẩn bị trong QGIS

Cần 3 layer:

| Layer | Nguồn | Vai trò |
|---|---|---|
| Ảnh nền | `*_TCI.jp2` (hoặc GeoTIFF đã burn) | ảnh gốc để nhìn và để cắt tile |
| `ship_labels.gpkg` | copy từ `{stem}_pred.geojson` | layer nhãn **sẽ sửa tay** |
| `tiles.gpkg` | lệnh `grid` bên dưới | lưới ô + cột `status` để đánh dấu đã soát |

**Dùng đúng ảnh gốc** (`*_TCI.jp2`), không dùng ảnh overview/PNG đã thu nhỏ — nếu không
toạ độ pixel khi cắt sẽ lệch.

### 1a. Tạo layer nhãn từ kết quả model

Không sửa trực tiếp file geojson do notebook sinh ra (còn cần để đối chiếu). Trong QGIS:

1. Chuột phải layer `{stem}_pred` → **Export → Save Features As…**
2. Format **GeoPackage**, tên `ship_labels.gpkg`, **CRS: chọn đúng CRS của ảnh** (ví dụ
   EPSG:32651 cho tile 51RUM) — làm việc trong CRS mét thì vẽ hình chữ nhật dễ hơn nhiều.
3. Bỏ tick trường `confidence` nếu muốn cho gọn (script không dùng đến).

> Nếu muốn bắt đầu từ layer trắng: **Layer → Create Layer → New GeoPackage Layer**,
> geometry type **Polygon**, CRS = CRS của ảnh.

Nên gộp nhãn của **nhiều ảnh** vào cùng một file `ship_labels.gpkg` — script tự lọc theo
từng ảnh dựa trên vị trí địa lý.

### 1b. Tạo lưới ô cần soát

```bash
python labeling/qgis_labels_to_dataset.py grid \
  --raster /duong/dan/T51RUM_..._TCI.jp2 \
  --near   /duong/dan/T51RUM_..._pred.geojson \
  --buffer 500 \
  --tile 800 \
  --out    tiles.gpkg
```

- `--near` + `--buffer`: chỉ giữ ô có dự đoán (± 500 m) — bạn không phải soát cả 12 000 ô
  của một ảnh Sentinel-2 đầy đủ. Bỏ `--near` (hoặc thêm `--keep-empty`) nếu muốn cả lưới.
- `--raster` nhận nhiều ảnh hoặc một thư mục; cột `raster` trong kết quả ghi rõ ô thuộc ảnh nào.
- `--no-snap-last`: các ô rời hẳn nhau (mặc định ô cuối mỗi hàng/cột bị kéo sát mép nên
  **chồng lấn** ô kề — xem mục 5 về rò rỉ dữ liệu).

Kết quả `tiles.gpkg` có các cột `tile_id, raster, ox, oy, w, h, n_pred, status, note`.

### 1c. Cấu hình QGIS cho dễ vẽ tàu 4–18 px

- Style ảnh: **Symbology → Contrast enhancement: Stretch to MinMax**, cắt ở phân vị 2–98
  để tàu nổi bật trên nước.
- Style `tiles.gpkg`: fill trong suốt, viền vàng; **Labels** hiển thị `tile_id`; và đặt
  **rule-based / categorized theo `status`** để ô `done` đổi màu — nhìn phát biết đã soát tới đâu.
- Style `ship_labels`: fill trong suốt, viền đỏ, độ rộng 0.3 mm.
- Tắt **Snapping** (biểu tượng nam châm) — bật sẽ khiến box mới bị hút vào box cũ.
- Zoom làm việc: khoảng **1:2 000 – 1:5 000**. Dùng phím tắt `Ctrl + rê chuột giữa` để pan.

---

## 2. Soát nhãn (phần làm tay)

Bật **Toggle Editing** (bút chì) cho layer `ship_labels`, và bật thanh công cụ
**View → Toolbars → Shape Digitizing**.

Với **từng ô** trong `tiles.gpkg`, làm đủ 4 việc:

1. **Xoá false positive** — box không phải tàu (mây, sóng vỡ, đảo nhỏ, phao):
   công cụ *Select Features* → `Delete Selected`.
2. **Sửa box lệch** — *Vertex Tool* kéo góc cho box ôm sát tàu (kể cả vệt nước sau đuôi
   thì **không** tính vào box; box chỉ ôm thân tàu — giữ đúng quy ước của 2000 ảnh cũ).
3. **Thêm tàu bị sót** — công cụ **Add Rectangle → Rectangle from Extent**, kéo một hình
   chữ nhật ôm sát tàu. (`Add Polygon Feature` thông thường cũng được nhưng chậm hơn.)
4. Mở bảng thuộc tính `tiles.gpkg`, bật editing, đặt `status = done` cho ô vừa soát xong.
   (Nhanh hơn: chọn nhiều ô trên bản đồ → **Field Calculator** → *Update existing field*
   `status` = `'done'` → tick **Only update selected features**.)

Mẹo tốc độ:
- Ô nào nhìn hết một lượt không thấy tàu nào → vẫn đặt `done`: nó thành hard-negative.
- Ô mây dày, ô có sương mù dày đặc mà bạn **không chắc** → để `todo`, đừng đoán. Nhãn
  đoán bừa hại hơn là thiếu dữ liệu.
- Muốn nhanh hơn nữa: vẽ **điểm** thay vì hình chữ nhật (layer Point) rồi dùng
  `--point-box-m 60` khi export để tự sinh box 60 m (6 px). **Chỉ nên dùng khi bí thời
  gian** — box cùng một cỡ làm hỏng phần hồi quy kích thước; ưu tiên hình chữ nhật.

---

## 3. Cắt thành ảnh + nhãn

```bash
python labeling/qgis_labels_to_dataset.py export \
  --raster /duong/dan/T51RUM_..._TCI.jp2 \
  --labels ship_labels.gpkg \
  --tiles  tiles.gpkg \
  --out    new_tiles \
  --tile 800 --qc
```

Sinh ra:

```
new_tiles/
  images/  T51RUM_..._TCI_x4000_y2400.png     # tile 800x800, RGB uint8
  labels/  T51RUM_..._TCI_x4000_y2400.txt     # YOLO: class cx cy w h (chuẩn hoá)
  _annotations.coco.json                      # COCO: bbox [x, y, w, h] pixel, 1 category "ship"
  tiles_index.csv                             # ảnh gốc / ox / oy / CRS -> truy ngược về toạ độ thật
  qc/                                         # ảnh đã vẽ box (chỉ khi có --qc)
```

Điểm cần biết:
- Chỉ ô `status=done` được xuất (`--status-field/--status-value` nếu bạn đặt tên khác).
- Box **giao** với ô đều được lấy và cắt theo mép ô, nên không có con tàu nào nằm trong
  ảnh mà thiếu nhãn. Mảnh nhỏ hơn `--min-box-px` (mặc định 2 px) bị bỏ.
- Ô rỗng vẫn được xuất làm hard-negative; tắt bằng `--no-keep-empty`.
- Ảnh không phải `uint8` được chuẩn hoá bằng **một hệ số duy nhất cho cả ảnh** (phân vị
  99.9), không phải theo từng cửa sổ như lúc infer — để độ sáng giữa các tile nhất quán.
  Với TCI (vốn đã uint8) thì bước này là no-op.

**Luôn mở `new_tiles/qc/` xem qua một lượt trước khi merge.** Dòng thống kê "cạnh nhỏ nhất
của box" mà script in ra nên nằm quanh 4–18 px; nếu ra 50–100 px là bạn vẽ box quá rộng.

---

## 4. Trộn vào bộ train/valid/test

Bộ dữ liệu gốc **không bị sửa** — script copy sang thư mục mới.

**Cho InternImage (COCO)** — bố cục Roboflow `{split}/_annotations.coco.json`:

```bash
python labeling/qgis_labels_to_dataset.py merge \
  --new     new_tiles \
  --dataset /kaggle/input/datasets/hunglq/annotated-sentinel \
  --out     /kaggle/working/annotated-sentinel-v2 \
  --format  coco \
  --ratios  0.7 0.15 0.15 --seed 0
```

**Cho YOLO11** (`{split}/images` + `{split}/labels`): đổi `--format yolo`.
Dùng `--format both` nếu muốn cả hai.

Script tự đánh lại `image_id`/`annotation_id` (không đụng ảnh cũ) và gán `category_id`
đúng bằng **category mà annotation cũ đang dùng** — tránh đúng cái bẫy 2 category trùng
tên `ship` (id 0 rỗng / id 1 thật) mà Section 5c của notebook đã xử lý.

Sau đó chỉ việc trỏ `CFG` của notebook fine-tune vào thư mục `--out` mới.

### Chia split thế nào cho đúng

- `--ratios 0.7 0.15 0.15` (mặc định): chia ngẫu nhiên theo ô. Hợp lý khi ô không chồng nhau.
- `--all-to train`: dồn **toàn bộ** ô mới vào train, giữ nguyên valid/test cũ.
  **Đây là lựa chọn tốt nhất nếu bạn muốn so sánh công bằng model mới vs model cũ** —
  test không đổi thì con số mAP mới đối chiếu trực tiếp được với lần train trước.
- `--group-by scene`: mọi ô của cùng một ảnh gốc rơi vào cùng một split (chống rò rỉ khi
  các ô của cùng cảnh chồng lấn hoặc quá giống nhau).

Khuyến nghị thực dụng: **chạy hai lần**.
1. Lần 1 `--all-to train` để đo cải thiện trên test cũ (so sánh táo với táo).
2. Sau khi đã hài lòng, chạy lại với `--ratios` để một phần mẫu khó cũng vào val/test —
   giúp lần fine-tune sau đo được đúng loại lỗi này.

---

## 5. Bẫy hay gặp

| Bẫy | Hậu quả | Cách tránh |
|---|---|---|
| Chỉ vẽ tàu bị sót, bỏ qua tàu model đã bắt đúng | Tàu đúng thành background → model **tệ đi** | Chỉ export ô đã soát **đầy đủ** (`status=done`) |
| Vẽ box gồm cả vệt nước sau đuôi tàu | Box to hơn nhãn cũ → phân bố kích thước lệch, mAP giảm | Box ôm **thân tàu**, kiểm bằng thống kê px mà `export` in ra |
| Ô cuối hàng/cột chồng lấn ô kề, rồi chia ngẫu nhiên | Cùng một vùng nằm ở cả train lẫn test → mAP ảo | `grid --no-snap-last`, hoặc `merge --group-by scene`, hoặc `--all-to train` |
| Layer nhãn ở CRS khác ảnh | Box lệch vị trí | Script tự `to_crs`, nhưng nên đặt layer nhãn ở đúng CRS ảnh ngay từ đầu |
| Vẽ trên ảnh overview/PNG thu nhỏ | Toạ độ pixel sai | Luôn vẽ trên `*_TCI.jp2` gốc |
| Trộn ảnh L1C vào bộ L2A | Màu khác miền → nhiễu | Giữ đúng mức xử lý (bạn đang train trên L2A) |
| Thêm 30 ô mới vào 2000 ô cũ rồi mong đổi đời | Ảnh hưởng bị pha loãng | Xem mục 6 |

---

## 6. Fine-tune lại thế nào cho hiệu quả

Vài chục ô mới bên cạnh ~2000 ô cũ là **~1–2 % dữ liệu** — nếu train y hệt lần trước thì
gần như không thấy khác biệt. Các cách xử lý, theo thứ tự nên thử:

1. **Khởi tạo từ checkpoint đã fine-tune của bạn**, không phải từ checkpoint COCO gốc:
   trong notebook InternImage đặt `load_from` = `best/latest .pth` của lần train trước
   (với YOLO: `model = YOLO('best.pt')`). Rẻ hơn nhiều: vài epoch là đủ.
2. **Learning rate nhỏ** — khoảng 1/5 đến 1/10 lr của lần fine-tune đầu, `warmup` ngắn,
   chạy 3–6 epoch. Lr to sẽ xoá mất những gì model đã học từ 2000 ảnh.
3. **Oversample ô mới**: lặp lại danh sách ảnh mới 3–5 lần trong split train (với mmdet có
   thể bọc dataset bằng `RepeatDataset` / `ClassBalancedDataset`; với YOLO thì copy ảnh +
   nhãn thêm vài bản đổi tên). Đây là cách rẻ nhất để mẫu khó thực sự có tiếng nói.
4. **Đo riêng**: ngoài mAP trên test cũ, giữ một thư mục test riêng chỉ gồm ô mới và eval
   thêm trên đó. Hai con số đọc cùng nhau mới biết mình *sửa được lỗi mới* mà *không làm hỏng
   cái cũ* (catastrophic forgetting).
5. **Lặp lại vòng**: infer bằng model mới → xem geojson trong QGIS → những chỗ vẫn sai lại
   thành lô nhãn tiếp theo. Vòng 2–3 lần thường hiệu quả hơn một lần gán nhãn thật nhiều.

Nếu tàu bị bỏ sót chủ yếu là loại **rất nhỏ / mờ**, kiểm tra thêm phía model: `anchor
scales` (Section 7 của notebook InternImage đã hạ) và ngưỡng `CONF` lúc infer — có khi
model *có* bắt được nhưng bị lọc mất bởi `CONF` hoặc bởi bộ lọc mây
(`CLOUD_PROTECT_CONF`, `CLOUD_FRAC_THR`). Đối chiếu `{stem}_pred_raw.geojson`
(trước lọc mây) với `{stem}_pred.geojson` (sau lọc) trong QGIS: tàu nào chỉ có ở file
`_raw` nghĩa là **bị bộ lọc mây ăn mất**, chỉnh tham số rẻ hơn gán nhãn.

---

## 7. Lựa chọn thay thế: gán nhãn thẳng trên Roboflow

Bộ dữ liệu hiện tại vốn export từ Roboflow. Nếu quen giao diện đó hơn QGIS:

1. Chạy `export` với `--no-keep-empty` bỏ qua (vẫn giữ ô rỗng) để lấy `new_tiles/images/`.
2. Upload thư mục ảnh đó vào đúng project Roboflow, gán nhãn trong trình duyệt.
3. Generate version mới → export **COCO** (cho InternImage) hoặc **YOLOv11** (cho YOLO).

Đánh đổi: mất khả năng nhìn ảnh trong bối cảnh địa lý (rất tiện để phân biệt tàu với đảo
nhỏ hay bãi cạn cố định) và mất liên kết toạ độ, nhưng giao diện gán nhãn nhanh hơn và
việc chia split do Roboflow lo. Vẫn áp dụng **nguyên tắc mục 0**: mỗi ảnh upload lên phải
được gán nhãn **đầy đủ**.
