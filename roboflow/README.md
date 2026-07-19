# Deploy model YOLO11 fine-tune lên Roboflow (Hosted Inference API)

Đưa `best.pt` (đã fine-tune trên Kaggle) lên Roboflow để có **API đám mây**: ảnh vào → JSON toạ độ tàu + ảnh gán nhãn.
Roboflow lo phần server/scale; bạn chỉ upload weights.

> **Chi phí & riêng tư:** gói **Public (miễn phí)** đủ cho demo (~$60 credits/tháng, 1 credit ≈ 500 giây suy luận),
> **nhưng dữ liệu + model sẽ CÔNG KHAI** trên Roboflow Universe. Muốn riêng tư cần gói trả phí (Core ~$79/tháng).
> Giá thay đổi theo thời gian — xem [roboflow.com/pricing](https://roboflow.com/pricing).

---

## Bước 0 — Điều kiện cần

- Tài khoản Roboflow (miễn phí).
- **Project chứa dataset** của bạn trên Roboflow. Dataset của bạn vốn export từ Roboflow (tên file `...rf...`),
  nên nhiều khả năng đã có project sẵn. Weights upload lên phải gắn với **một version** của project đó.
- File `best.pt` từ Kaggle: `/kaggle/working/finetune_outputs/finetune/weights/best.pt`.

## Bước 1 — Lấy 4 thông tin định danh

Vào project trên `app.roboflow.com`. URL có dạng:
```
https://app.roboflow.com/<WORKSPACE>/<PROJECT>/<VERSION>
```
- **WORKSPACE**, **PROJECT**, **VERSION** lấy từ URL đó (VERSION là số, ví dụ `1`).
- **API key**: Roboflow → **Settings → API Keys** → copy **Private API Key**.

> Nếu project chưa có version nào: vào tab **Versions → Generate** để tạo một version trước.

## Bước 2 — Cài SDK

```bash
pip install roboflow
```

## Bước 3 — Upload weights (chạy nơi có `best.pt`, ví dụ ngay trong Kaggle)

Dùng script kèm theo `deploy_to_roboflow.py` (điền thông tin ở đầu file hoặc đặt biến môi trường):

```bash
export ROBOFLOW_API_KEY="..."      # Private API Key
export RF_WORKSPACE="..."          # workspace id
export RF_PROJECT="..."            # project id
export RF_VERSION="1"              # version number
export RF_MODEL_DIR="/kaggle/working/finetune_outputs/finetune"   # chứa weights/best.pt
python deploy_to_roboflow.py
```

Bản chất script chỉ gọi:
```python
from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
project.version(VERSION).deploy(
    model_type="yolo11",
    model_path="/kaggle/working/finetune_outputs/finetune",  # thư mục chứa weights/best.pt
    filename="weights/best.pt",
)
```

## Bước 4 — Chờ xử lý & kiểm tra

Sau vài phút, vào tab **Deploy** của project — model sẽ hiện và thử được ngay trên web. `MODEL_ID` để gọi API là
`<PROJECT>/<VERSION>` (ví dụ `annotated-sentinel/1`).

## Bước 5 — Gọi API: ảnh → toạ độ + ảnh gán nhãn

```bash
pip install inference-sdk supervision opencv-python
export ROBOFLOW_API_KEY="..."
export RF_MODEL_ID="<PROJECT>/<VERSION>"
python roboflow_infer.py anh_test.jpg
```
Script in ra danh sách toạ độ tàu và lưu `predictions.json` + `annotated.jpg`.

Cốt lõi:
```python
from inference_sdk import InferenceHTTPClient
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
result = client.infer("anh.jpg", model_id="<PROJECT>/<VERSION>")
# result["predictions"]: mỗi tàu có x,y (TÂM box), width, height (pixel), confidence, class
```

### Cách khác — gọi HTTP trực tiếp (không cần SDK)
```bash
BASE64=$(base64 -w0 anh_test.jpg)
curl -X POST "https://detect.roboflow.com/<PROJECT>/<VERSION>?api_key=$ROBOFLOW_API_KEY&confidence=25&overlap=70" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "$BASE64"                              # thêm &format=image để trả THẲNG ảnh gán nhãn
```
- Mặc định trả **JSON**. Thêm `&format=image` để nhận **ảnh đã vẽ box**.

### Cách khác — dùng luôn gói `roboflow` (lưu ảnh gán nhãn 1 dòng)
```python
model = rf.workspace(WS).project(PROJ).version(N).model
model.predict("anh.jpg", confidence=25, overlap=70).save("annotated.jpg")   # ảnh gán nhãn
print(model.predict("anh.jpg").json())                                      # toạ độ
```

---

## Lưu ý về toạ độ

Roboflow trả **`x, y` = TÂM** bounding box (không phải góc trên-trái), `width/height` theo **pixel**. Quy đổi:
```
x1 = x - width/2 ;  y1 = y - height/2      # góc trên-trái
```

## (Tuỳ chọn) Chạy inference MIỄN PHÍ hoàn toàn — self-host

Không muốn tốn credit đám mây: chạy **Roboflow Inference server** open-source ngay trên máy bạn (CPU/GPU),
nó tự kéo weights từ workspace của bạn:
```bash
pip install inference
inference server start           # dựng server local ở cổng 9001
```
Rồi trỏ `InferenceHTTPClient(api_url="http://localhost:9001", api_key=...)`. Không tính phí theo lượt suy luận.

## Lỗi thường gặp

- **`model_path` sai:** phải là **thư mục** chứa `weights/best.pt`, không phải trỏ thẳng file `.pt`.
- **Version chưa tồn tại:** tạo version ở tab Versions trước khi deploy.
- **Dùng Public plan:** dữ liệu/model công khai — đừng upload dữ liệu nhạy cảm nếu cần riêng tư.
- **Sai kiến trúc:** phải để `model_type="yolo11"` đúng với model đã train.
