"""
Đẩy model YOLO11 đã fine-tune (best.pt) lên Roboflow -> lấy Hosted Inference API.

Chạy sau khi fine-tune xong, NGAY TRONG KAGGLE (nơi có best.pt), hoặc local nếu đã tải best.pt về.

Cài đặt:
    pip install roboflow

Điền thông tin (hoặc đặt qua biến môi trường) rồi chạy:
    python deploy_to_roboflow.py
"""
import os
from roboflow import Roboflow

# ------------------------------------------------------------------ ĐIỀN THÔNG TIN
API_KEY   = os.getenv("ROBOFLOW_API_KEY", "PASTE_PRIVATE_API_KEY")  # Roboflow > Settings > API Keys
WORKSPACE = os.getenv("RF_WORKSPACE", "your-workspace-id")          # từ URL app.roboflow.com/<workspace>/<project>
PROJECT   = os.getenv("RF_PROJECT",   "your-project-id")
VERSION   = int(os.getenv("RF_VERSION", "1"))                       # số version của dataset

# Thư mục "run" của ultralytics — PHẢI chứa weights/best.pt.
# Từ notebook fine-tune (Kaggle): /kaggle/working/finetune_outputs/finetune
MODEL_DIR = os.getenv("RF_MODEL_DIR", "/kaggle/working/finetune_outputs/finetune")

# ------------------------------------------------------------------ Kiểm tra nhanh
best = os.path.join(MODEL_DIR, "weights", "best.pt")
assert os.path.exists(best), f"Không thấy {best} — kiểm tra lại RF_MODEL_DIR."

# ------------------------------------------------------------------ Upload
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

print(f"Đang upload {best} lên project '{PROJECT}' version {VERSION} ...")
version.deploy(model_type="yolo11", model_path=MODEL_DIR, filename="weights/best.pt")

print("\nXong. Roboflow đang xử lý/host model (vài phút).")
print("Kiểm tra ở tab 'Deploy' của project trên app.roboflow.com.")
print(f"MODEL_ID để inference: {PROJECT}/{VERSION}")
