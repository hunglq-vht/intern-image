"""
Client thử nghiệm nhanh cho API.

Cách dùng:
    python test_client.py path/to/anh.png
    python test_client.py path/to/anh.png http://localhost:8000

Kết quả: in JSON toạ độ ra màn hình + lưu ảnh gán nhãn thành 'annotated.png'.
"""
import sys
import base64
import requests

def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python test_client.py <đường_dẫn_ảnh> [base_url]")
        sys.exit(1)
    img_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    with open(img_path, "rb") as f:
        resp = requests.post(f"{base_url}/predict", files={"file": f})
    resp.raise_for_status()
    data = resp.json()

    print(f"Số tàu phát hiện: {data['count']}")
    for i, d in enumerate(data["detections"], 1):
        print(f"  #{i}: conf={d['confidence']}  xyxy_px={d['bbox_xyxy_pixel']}  yolo_norm={d['yolo_norm_xywh']}")

    # Lưu ảnh gán nhãn
    with open("annotated.png", "wb") as f:
        f.write(base64.b64decode(data["annotated_image_png_base64"]))
    print("Đã lưu ảnh gán nhãn -> annotated.png")

if __name__ == "__main__":
    main()
