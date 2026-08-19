# HỆ THỐNG LÕI NHẬN DIỆN KHUÔN MẶT THỜI GIAN THỰC CHO AI CAMERA
## Kiến trúc Hybrid AI Combo, Controlled A/B Benchmarking và Real-time Debug Dashboard (Báo cáo Tuần 5)

Hệ thống Core Backend và REST API chuyên dụng cho AI Camera, hỗ trợ nhận diện khuôn mặt thời gian thực với kiến trúc đa tổ hợp thuật toán (Hybrid AI Combo Strategy), cơ chế lưu vết lỗi nghiêm ngặt (Strict Error Tracking) lưu trữ vào PostgreSQL và pgvector, tích hợp giao diện Debug Dashboard (HTML5, CSS3, JavaScript thuần) hỗ trợ tính năng lật ảnh camera (Mirror Camera) qua WebSocket.

---

## MỤC LỤC
1. Giới thiệu dự án và tổng quan tiến độ
2. So sánh và hướng dẫn chuyển đổi các tổ hợp mô hình AI
3. Nguyên lý hoạt động của kiến trúc Hybrid Cascade
4. Công nghệ sử dụng
5. Cấu trúc thư mục dự án
6. Hướng dẫn triển khai bằng Docker (Phương pháp ưu tiên)
   - 6.1. Lệnh dọn dẹp môi trường trước khi khởi chạy
   - 6.2. Lệnh khởi chạy hệ thống bằng Docker Compose
   - 6.3. Lệnh theo dõi nhật ký hoạt động (Logs)
   - 6.4. Lệnh tắt và dừng hệ thống
7. Hướng dẫn cài đặt và vận hành thủ công (Môi trường Local)
8. Danh sách REST API và WebSocket Endpoints

---

## 1. GIỚI THIỆU DỰ ÁN VÀ TỔNG QUAN TIẾN ĐỘ

Dự án được xây dựng và phát triển qua các giai đoạn theo đúng lộ trình đề cương:

### Phân biệt mục tiêu Tuần 4 và Tuần 5:

- **Mục tiêu Tuần 4 (Đăng ký và Quản lý CSDL Vector)**:
  - Xây dựng API quản lý thông tin nhân viên (`/api/v1/enroll`) và trích xuất vector đặc trưng 512 chiều (Embedding) từ 5-10 ảnh mẫu.
  - Lưu trữ dữ liệu vector chuẩn (Ground Truth) vào CSDL PostgreSQL thông qua extension pgvector.

- **Mục tiêu Tuần 5 (Lõi nhận diện thời gian thực và Dashboard)**:
  - **Matching Engine siêu tốc**: Tải trước (preload) toàn bộ vector từ PostgreSQL vào RAM Cache khi khởi động server, cho phép so sánh Cosine Similarity trực tiếp trên bộ nhớ với thời gian xử lý dưới 5ms mỗi khung hình.
  - **Phân loại 4 trạng thái nhận diện**:
    - RECOGNIZED: Nhận diện thành công (Độ tương đồng lớn hơn hoặc bằng 0.65).
    - UNCERTAIN: Nghi ngờ (Độ tương đồng từ 0.45 đến dưới 0.65).
    - UNKNOWN: Người lạ hoặc chưa đăng ký (Độ tương đồng dưới 0.45).
    - REJECTED: Bị từ chối do không đạt tiêu chuẩn đầu vào của Quality Gate (ảnh mờ, quá nhỏ, độ nghiêng lớn).
  - **Strict Error Tracking**: Tự động lưu vết hình ảnh bị loại hoặc nghi ngờ vào bộ nhớ cục bộ và ghi nhận dữ liệu chẩn đoán vào bảng `SystemBacklog`.
  - **Multi-frame Voting và Anti-Duplicate**: Tích hợp thuật toán theo dõi vết (ByteTrack/YOLO) qua `track_id` để bình chọn kết quả qua nhiều khung hình liên tiếp, triệt tiêu hiện tượng nhấp nháy video và chống ghi trùng lặp dữ liệu điểm danh.
  - **WebSocket Live Stream và Debug Dashboard**: Truyền tải luồng video thời gian thực (15-30 FPS) qua WebSocket endpoint `/ws/stream` cùng giao diện hiển thị Bounding Box phân màu và tính năng lật ảnh camera.

---

## 2. SO SÁNH VÀ HƯỚNG DẪN CHUYỂN ĐỔI CÁC TỔ HỢP MÔ HÌNH AI

Hệ thống hỗ trợ 3 tổ hợp thuật toán (AI Combo) được điều khiển thông qua biến môi trường `ACTIVE_AI_COMBO` trong file `.env`:

| AI Combo Strategy | Thành phần | Đặc điểm và Ứng dụng |
| :--- | :--- | :--- |
| **YUNET_SFACE** | YuNet Detector + SFace Recognizer | Tối ưu tốc độ: Tiêu thụ ít tài nguyên CPU/iGPU, đáp ứng tốc độ 30 FPS mượt mà trên thiết bị cấu hình phổ thông. |
| **RETINAFACE_ARCFACE** | RetinaFace Detector + ArcFace Recognizer | Tối ưu độ chính xác: Xử lý tốt các trường hợp góc mặt nghiêng, khoảng cách xa hoặc ánh sáng phức tạp. |
| **HYBRID_CASCADE** | YuNet -> Quality Gate -> RetinaFace | Kết hợp phân tầng: Sử dụng YuNet làm mặc định để tối ưu tốc độ, tự động chuyển tiếp (fallback) sang RetinaFace khi phát hiện khung hình khó. |

---

## 3. NGUYÊN LÝ HOẠT ĐỘNG CỦA KIẾN TRÚC HYBRID CASCADE

Khái niệm **Cascade** thể hiện cơ chế xử lý phân tầng nối tiếp theo quy trình 3 bước:

1. **Tầng 1 (Ưu tiên tốc độ với YuNet)**: Luồng video đầu vào được xử lý trước tiên bởi YuNet nhằm tối ưu hiệu năng tính toán. YuNet giải quyết thành công khoảng 85% - 90% các khung hình thông thường.
2. **Đánh giá tiêu chuẩn (Quality Gate)**: Khung hình được kiểm tra các chỉ số về độ phân giải, độ mờ và góc xoay. Nếu đạt tiêu chuẩn, hệ thống thực hiện trích xuất vector và nhận diện ngay lập tức.
3. **Tầng 2 (Dự phòng độ chính xác với RetinaFace)**: Trong trường hợp YuNet không phát hiện được khuôn mặt do điều kiện ánh sáng kém, góc nghiêng lớn hoặc khoảng cách xa, hệ thống tự động hạ tầng (fallback) kích hoạt RetinaFace để quét lại khung hình.

**Ý nghĩa kỹ thuật**: Cơ chế Hybrid Cascade giúp hệ thống duy trì được tốc độ xử lý cao (30 FPS) ở điều kiện vận hành bình thường mà vẫn đảm bảo không bỏ sót các trường hợp nhận diện khó.

---

## 4. HƯỚNG DẪN CHUYỂN ĐỔI TỔ HỢP MÔ HÌNH BẰNG LỆNH TERMINAL

Thay đổi cấu hình mô hình trực tiếp qua dòng lệnh và khởi động lại container:

### Trường hợp 1: Chuyển sang YUNET_SFACE (Tối ưu tốc độ)
- **Trên Windows (PowerShell)**:
  ```powershell
  (Get-Content .env) -replace 'ACTIVE_AI_COMBO=.*', 'ACTIVE_AI_COMBO=YUNET_SFACE' | Set-Content .env
  docker-compose up --build -d
  ```
- **Trên Linux / macOS / Git Bash**:
  ```bash
  sed -i 's/ACTIVE_AI_COMBO=.*/ACTIVE_AI_COMBO=YUNET_SFACE/' .env
  docker-compose up --build -d
  ```

### Trường hợp 2: Chuyển sang RETINAFACE_ARCFACE (Tối ưu độ chính xác)
- **Trên Windows (PowerShell)**:
  ```powershell
  (Get-Content .env) -replace 'ACTIVE_AI_COMBO=.*', 'ACTIVE_AI_COMBO=RETINAFACE_ARCFACE' | Set-Content .env
  docker-compose up --build -d
  ```
- **Trên Linux / macOS / Git Bash**:
  ```bash
  sed -i 's/ACTIVE_AI_COMBO=.*/ACTIVE_AI_COMBO=RETINAFACE_ARCFACE/' .env
  docker-compose up --build -d
  ```

### Trường hợp 3: Chuyển sang HYBRID_CASCADE (Phân tầng tự động)
- **Trên Windows (PowerShell)**:
  ```powershell
  (Get-Content .env) -replace 'ACTIVE_AI_COMBO=.*', 'ACTIVE_AI_COMBO=HYBRID_CASCADE' | Set-Content .env
  docker-compose up --build -d
  ```
- **Trên Linux / macOS / Git Bash**:
  ```bash
  sed -i 's/ACTIVE_AI_COMBO=.*/ACTIVE_AI_COMBO=HYBRID_CASCADE/' .env
  docker-compose up --build -d
  ```

---

## 5. CÔNG NGHỆ SỬ DỤNG (TECH STACK)

- Ngôn ngữ lập trình: Python 3.11+
- Backend Framework: FastAPI, Uvicorn, Pydantic v2
- Cơ sở dữ liệu và ORM: PostgreSQL 16, pgvector extension (Vector 512 chiều), SQLAlchemy 2.0, Alembic Migration
- Thị giác máy tính và AI: OpenCV 4.9+, YuNet, SFace, RetinaFace, ArcFace, YOLO, ByteTrack
- Frontend Debug Dashboard: HTML5, CSS3, JavaScript (WebSocket API)
- Đóng gói và Triển khai: Docker, Docker Compose

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
thuctapccvi/
├── .env                              # File cấu hình biến môi trường
├── .env.example                      # File cấu hình mẫu
├── Dockerfile                        # Script cấu hình Docker image
├── docker-compose.yml                # File cấu hình Docker Compose services
├── alembic.ini                       # Cấu hình Alembic database migrations
├── requirements.txt                  # Danh sách các thư viện phụ thuộc
├── README.md                         # Tài liệu hướng dẫn sử dụng dự án
├── alembic/                          # Thư mục chứa các script migration CSDL
│   └── versions/
├── storage/                          # Lưu trữ dữ liệu hình ảnh và trọng số model
│   ├── captures/                     # Lưu ảnh nhận diện thành công
│   ├── failed/                       # Lưu ảnh lỗi quá trình đăng ký và nhận diện
│   └── models/                       # Lưu trữ file trọng số ONNX
├── static/                           # Mã nguồn Frontend Debug Dashboard
│   ├── index.html                    # Giao diện chính Dashboard
│   ├── css/style.css                 # File định dạng giao diện
│   └── js/app.js                     # Xử lý WebSocket và vẽ Bounding Box
└── app/
    ├── main.py                       # Điểm khởi chạy ứng dụng FastAPI
    ├── core/                         # Cấu hình hệ thống và logging
    ├── db/                           # Khởi tạo kết nối CSDL và pgvector
    ├── models/                       # Định nghĩa các bảng CSDL (ORM Models)
    ├── services/                     # Lõi xử lý AI, Camera Agent, Quality Gate
    └── api/                          # Định nghĩa các REST API và WebSocket router
```

---

## 7. HƯỚNG DẪN TRIỂN KHAI BẰNG DOCKER (PHƯƠNG PHÁP ƯU TIÊN)

Triển khai bằng Docker đảm bảo tính nhất quán về môi trường thực thi và tính đóng gói của hệ thống.

### 7.1. Lệnh dọn dẹp môi trường trước khi khởi chạy
Trong trường hợp xuất hiện xung đột container cũ, trùng cổng kết nối (Port 5432 / 8000) hoặc lỗi bộ nhớ đệm, thực hiện các lệnh dọn dẹp sau:

```bash
# 1. Dừng và xóa toàn bộ container, network và volume dữ liệu cũ
docker-compose down -v --remove-orphans

# 2. Xóa các container không còn hoạt động
docker container prune -f

# 3. Xóa bộ nhớ đệm build cũ
docker builder prune -f
```

---

### 7.2. Lệnh khởi chạy hệ thống bằng Docker Compose
Thực hiện xây dựng lại image và khởi chạy các dịch vụ ở chế độ chạy ngầm:

```bash
docker-compose up --build -d
```

Sau khi khởi chạy thành công:
- Truy cập giao diện Debug Dashboard tại: `http://localhost:8000`
- Truy cập tài liệu REST API (Swagger UI) tại: `http://localhost:8000/docs`

---

### 7.3. Lệnh theo dõi nhật ký hoạt động (Logs)
Để theo dõi quá trình thực thi và xử lý sự cố:

```bash
# Theo dõi nhật ký của toàn bộ dịch vụ
docker-compose logs -f

# Theo dõi nhật ký riêng của API server
docker-compose logs -f api
```

---

### 7.4. Lệnh tắt và dừng hệ thống

- **Dừng hệ thống (giữ nguyên dữ liệu CSDL)**:
  ```bash
  docker-compose stop
  ```

- **Dừng và xóa các container (giữ nguyên dữ liệu CSDL)**:
  ```bash
  docker-compose down
  ```

- **Dừng, xóa container và xóa toàn bộ dữ liệu CSDL (Làm sạch hoàn toàn)**:
  ```bash
  docker-compose down -v --remove-orphans
  ```

---

## 8. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH THỦ CÔNG (MÔI TRƯỜNG LOCAL)

Trong trường hợp vận hành trực tiếp không thông qua Docker:

### Bước 1: Khởi tạo CSDL PostgreSQL
Cài đặt PostgreSQL 16 và kích hoạt extension `pgvector` trên cơ sở dữ liệu mục tiêu.

### Bước 2: Khởi tạo môi trường ảo và cài đặt thư viện
```bash
# Kích hoạt môi trường ảo Python
.\venv\Scripts\activate

# Cài đặt danh sách thư viện phụ thuộc
pip install -r requirements.txt
```

### Bước 3: Cấu hình biến môi trường
Tạo file `.env` từ file mẫu `.env.example` và cập nhật thông tin kết nối CSDL:
```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/face_rec_db
ACTIVE_AI_COMBO=YUNET_SFACE
```

### Bước 4: Thực hiện Database Migration
```bash
alembic upgrade head
```

### Bước 5: Khởi chạy FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 9. DANH SÁCH REST API VÀ WEBSOCKET ENDPOINTS

- `GET /health`: Kiểm tra trạng thái hoạt động của hệ thống và số lượng vector đã nạp trong RAM Cache.
- `POST /api/v1/enroll`: Đăng ký nhân viên mới, kiểm tra tiêu chuẩn ảnh và lưu vector đặc trưng 512 chiều vào CSDL.
- `POST /api/v1/recognize`: Thực hiện nhận diện khuôn mặt từ ảnh tĩnh và phân loại theo 4 trạng thái.
- `GET /api/v1/camera/status`: Truy xuất thông số cấu hình và trạng thái luồng camera.
- `GET /api/v1/logs/backlog`: Truy xuất nhật ký lỗi và các trường hợp bị từ chối từ bảng `SystemBacklog`.
- `GET /api/v1/logs/recognition`: Truy xuất lịch sử các sự kiện nhận diện từ bảng `RecognitionEvent`.
- `GET /api/v1/logs/checkin`: Truy xuất dữ liệu điểm danh hợp lệ từ bảng `CheckinEvent`.
- `WS /ws/stream`: WebSocket endpoint truyền luồng video thời gian thực và metadata Bounding Box.
