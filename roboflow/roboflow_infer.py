"""
Gọi Hosted Inference API của Roboflow: ảnh -> JSON toạ độ tàu + ảnh đã gán nhãn.

Cài đặt:
    pip install inference-sdk supervision opencv-python

Dùng:
    python roboflow_infer.py anh_test.jpg
"""
import os
import sys
import json
from inference_sdk import InferenceHTTPClient

API_KEY  = os.getenv("ROBOFLOW_API_KEY", "PASTE_PRIVATE_API_KEY")
MODEL_ID = os.getenv("RF_MODEL_ID", "your-project-id/1")   # dạng {project}/{version}
CONF     = float(os.getenv("RF_CONF", "0.25"))             # ngưỡng confidence (0-1)
IMG      = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
result = client.infer(IMG, model_id=MODEL_ID)

# ------------------------------------------------------------------ In toạ độ
# Lưu ý: Roboflow trả x,y = TÂM box; width,height theo pixel.
preds = result.get("predictions", [])
preds = [p for p in preds if p.get("confidence", 0) >= CONF]
print(f"Số tàu phát hiện (conf>={CONF}): {len(preds)}")
for i, p in enumerate(preds, 1):
    print(f"  #{i}: conf={p['confidence']:.3f}  center=({p['x']:.0f},{p['y']:.0f})  "
          f"wh=({p['width']:.0f},{p['height']:.0f})")

with open("predictions.json", "w") as f:
    json.dump(result, f, indent=2)
print("Đã lưu predictions.json")

# ------------------------------------------------------------------ Vẽ ảnh gán nhãn
try:
    import cv2
    import supervision as sv
    image = cv2.imread(IMG)
    det = sv.Detections.from_inference(result)
    annotated = sv.BoxAnnotator().annotate(scene=image.copy(), detections=det)
    cv2.imwrite("annotated.jpg", annotated)
    print("Đã lưu annotated.jpg")
except Exception as e:
    print("Bỏ qua vẽ ảnh (thiếu supervision/opencv?):", e)
