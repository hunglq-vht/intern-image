# YOLO11 Ship Detection API — image CPU gọn nhẹ
FROM python:3.11-slim

WORKDIR /app

# libgl1 + libglib2.0-0: opencv (ultralytics dùng) cần các thư viện hệ thống này
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Cài torch bản CPU trước (tránh tải wheel CUDA hàng GB), rồi mới cài phần còn lại
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/best.pt \
    DEVICE=cpu \
    IMGSZ=800 \
    CONF=0.25 \
    IOU=0.7 \
    CLASS_NAME=vessel

# HF Spaces mặc định cổng 7860; Render/khác đặt qua $PORT. Mặc định 8000 khi chạy local.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
