"""
FastAPI + Ultralytics — API phát hiện tàu (YOLO11 đã fine-tune).

Đầu vào : một ảnh (upload multipart).
Đầu ra  : JSON toạ độ các tàu phát hiện được + ảnh đã vẽ bounding box.

Chạy local:
    export MODEL_PATH=./best.pt          # đường dẫn tới best.pt lấy từ Kaggle
    uvicorn app:app --host 0.0.0.0 --port 8000
Mở http://localhost:8000/docs để thử trực tiếp trên trình duyệt.
"""
import io
import os
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

# ------------------------------------------------------------------ Cấu hình (qua biến môi trường)
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")   # đường dẫn checkpoint YOLO11 đã fine-tune
CONF       = float(os.getenv("CONF", "0.25"))     # ngưỡng confidence
IOU        = float(os.getenv("IOU", "0.7"))       # ngưỡng IoU cho NMS
IMGSZ      = int(os.getenv("IMGSZ", "800"))       # kích thước ảnh khi suy luận (khớp lúc train)
DEVICE     = os.getenv("DEVICE", "cpu")           # "cpu" hoặc "0" (GPU)
CLASS_NAME = os.getenv("CLASS_NAME", "vessel")

app = FastAPI(
    title="YOLO11 Ship Detection API",
    description="Phát hiện tàu trên ảnh vệ tinh bằng model YOLO11 đã fine-tune.",
    version="1.0.0",
)

_model = None


def get_model():
    """Nạp model một lần, lười (chỉ khi có request đầu tiên) để app khởi động nhẹ."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Chưa tìm thấy model tại '{MODEL_PATH}'. "
                       f"Đặt biến môi trường MODEL_PATH hoặc copy best.pt vào đây.",
            )
        from ultralytics import YOLO  # import lười để module load không cần torch sẵn
        _model = YOLO(MODEL_PATH)
    return _model


def _read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh (định dạng không hợp lệ).")


def _predict(img: Image.Image, conf: float, iou: float, imgsz: int):
    model = get_model()
    return model.predict(img, imgsz=imgsz, conf=conf, iou=iou,
                         device=DEVICE, verbose=False)[0]


def _extract_detections(res, W: int, H: int):
    """Chuyển kết quả ultralytics thành danh sách toạ độ (pixel + chuẩn hoá YOLO)."""
    dets = []
    if res.boxes is None:
        return dets
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        w, h = x2 - x1, y2 - y1
        dets.append({
            "class": CLASS_NAME,
            "confidence": round(float(b.conf[0]), 4),
            "bbox_xyxy_pixel":  [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_xywh_pixel":  [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
            "yolo_norm_xywh":   [round((x1 + w / 2) / W, 6), round((y1 + h / 2) / H, 6),
                                 round(w / W, 6), round(h / H, 6)],
        })
    return dets


def _annotated_png(res) -> io.BytesIO:
    """Vẽ box lên ảnh (bộ vẽ sẵn của ultralytics) và trả về PNG trong bộ nhớ."""
    bgr = res.plot(line_width=2)          # ndarray BGR
    rgb = bgr[:, :, ::-1]                  # -> RGB
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return buf


# ------------------------------------------------------------------ Endpoints
@app.get("/")
def root():
    return {
        "service": "YOLO11 Ship Detection API",
        "docs": "/docs",
        "endpoints": {
            "GET  /health": "kiểm tra tình trạng + cấu hình",
            "POST /predict": "trả JSON toạ độ + ảnh gán nhãn (base64) trong 1 lần gọi",
            "POST /predict/image": "trả thẳng ảnh PNG đã gán nhãn",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": _model is not None,
        "device": DEVICE, "imgsz": IMGSZ, "conf": CONF, "iou": IOU,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Ảnh đầu vào"),
    conf: float = Query(None, description="Ghi đè ngưỡng confidence"),
    iou: float = Query(None, description="Ghi đè ngưỡng IoU"),
    imgsz: int = Query(None, description="Ghi đè imgsz"),
):
    """Trả JSON gồm danh sách toạ độ tàu VÀ ảnh đã gán nhãn (PNG mã hoá base64)."""
    img = _read_image(await file.read())
    W, H = img.size
    res = _predict(img, conf or CONF, iou or IOU, imgsz or IMGSZ)
    dets = _extract_detections(res, W, H)
    b64 = base64.b64encode(_annotated_png(res).getvalue()).decode("ascii")
    return JSONResponse({
        "count": len(dets),
        "image_size": {"width": W, "height": H},
        "detections": dets,
        "annotated_image_png_base64": b64,
    })


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(..., description="Ảnh đầu vào"),
    conf: float = Query(None),
    iou: float = Query(None),
    imgsz: int = Query(None),
):
    """Trả thẳng ảnh PNG đã vẽ bounding box (tiện xem trên trình duyệt)."""
    img = _read_image(await file.read())
    res = _predict(img, conf or CONF, iou or IOU, imgsz or IMGSZ)
    return StreamingResponse(_annotated_png(res), media_type="image/png")
