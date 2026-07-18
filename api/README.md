# YOLO11 Ship Detection API (FastAPI + Ultralytics)

API bọc model YOLO11 đã fine-tune (`best.pt`): **nhận 1 ảnh → trả JSON toạ độ các tàu + ảnh đã vẽ bounding box**.
Chạy được trên CPU (miễn phí), model giữ **riêng tư**, không phụ thuộc Roboflow.

## Endpoints

| Method & path | Chức năng |
|---|---|
| `GET /health` | Kiểm tra tình trạng + cấu hình |
| `POST /predict` | Trả **JSON**: `count`, `detections[]` (toạ độ) **và** ảnh gán nhãn (PNG base64) trong 1 lần gọi |
| `POST /predict/image` | Trả **thẳng ảnh PNG** đã vẽ box (tiện xem trên trình duyệt) |
| `GET /docs` | Giao diện Swagger để thử ngay trên trình duyệt |

Mỗi phần tử trong `detections`:
```json
{
  "class": "vessel",
  "confidence": 0.87,
  "bbox_xyxy_pixel": [x1, y1, x2, y2],
  "bbox_xywh_pixel": [x, y, w, h],
  "yolo_norm_xywh":  [xc, yc, w, h]
}
```
(`bbox_*_pixel` theo pixel; `yolo_norm_xywh` là toạ độ chuẩn hoá 0–1 kiểu YOLO.)

## 1. Lấy `best.pt` từ Kaggle

Sau khi chạy notebook fine-tune, tải file:
```
/kaggle/working/finetune_outputs/finetune/weights/best.pt
```
(trong Kaggle: panel **Output** → tải `best.pt`). Đặt nó cạnh `app.py` trong thư mục này, hoặc trỏ biến
môi trường `MODEL_PATH` tới đường dẫn của nó.

## 2. Chạy local

```bash
cd api
python -m venv .venv && source .venv/bin/activate     # tuỳ chọn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # torch CPU nhẹ
pip install -r requirements.txt

export MODEL_PATH=./best.pt      # đường dẫn checkpoint
uvicorn app:app --host 0.0.0.0 --port 8000
```
Mở http://localhost:8000/docs để thử. Hoặc bằng dòng lệnh:

```bash
# JSON toạ độ + ảnh base64
curl -X POST -F "file=@anh_test.png" http://localhost:8000/predict

# Lưu thẳng ảnh đã gán nhãn
curl -X POST -F "file=@anh_test.png" http://localhost:8000/predict/image --output ket_qua.png
```
Hoặc dùng client kèm sẵn:
```bash
python test_client.py anh_test.png            # in toạ độ + lưu annotated.png
```

## 3. Chạy bằng Docker

```bash
cd api
# đặt best.pt trong thư mục này trước khi build (hoặc mount volume lúc run)
docker build -t ship-api .
docker run -p 8000:8000 -v $(pwd)/best.pt:/app/best.pt ship-api
```

## 4. Deploy miễn phí

### Hugging Face Spaces (Docker) — khuyến nghị
1. Tạo Space mới, chọn **Docker → Blank**.
2. Upload các file trong thư mục này (`app.py`, `requirements.txt`, `Dockerfile`) **và** `best.pt`.
3. Thêm dòng này vào đầu `README.md` của Space (frontmatter) để HF biết cổng:
   ```
   ---
   title: Ship Detection API
   sdk: docker
   app_port: 8000
   ---
   ```
4. Space tự build & chạy; endpoint public dạng `https://<user>-<space>.hf.space/predict`.
   > Lưu ý: nếu `best.pt` > 10 MB, dùng **Git LFS** khi upload (HF hỗ trợ sẵn).

### Render / Railway
- Tạo **Web Service** từ repo, chọn môi trường **Docker** (dùng `Dockerfile` này).
- Không cần đặt start command (Dockerfile lo). Nền tảng tự cấp `$PORT` — Dockerfile đã đọc `${PORT}`.

## 5. Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_PATH` | `best.pt` | Đường dẫn checkpoint YOLO11 |
| `CONF` | `0.25` | Ngưỡng confidence |
| `IOU` | `0.7` | Ngưỡng IoU cho NMS |
| `IMGSZ` | `800` | Kích thước suy luận (khớp lúc train) |
| `DEVICE` | `cpu` | `cpu` hoặc `0` (GPU) |
| `CLASS_NAME` | `vessel` | Tên lớp hiển thị |

Có thể ghi đè `conf/iou/imgsz` theo từng request qua query param, ví dụ:
`POST /predict?conf=0.4&imgsz=1024`.

## 6. (Tuỳ chọn) Tăng tốc CPU bằng ONNX / OpenVINO

Model YOLO11s trên CPU ~0,3–0,5 s/ảnh (PyTorch, 800px). Export ONNX/OpenVINO nhanh hơn ~2–3×:

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx")       # -> best.onnx
# hoặc: .export(format="openvino")
```
Rồi trỏ `MODEL_PATH=best.onnx` — ultralytics tự nạp và chạy qua ONNX Runtime. Không cần đổi code API.

## Ghi chú

- CPU đủ nhanh cho API xử lý **từng ảnh** (dưới ~0,5 s/ảnh). Cần thông lượng cao / video → dùng GPU (`DEVICE=0`).
- `import ultralytics` được nạp **lười** ở request đầu tiên, nên app khởi động nhanh; lần gọi đầu sẽ hơi trễ do nạp model.
