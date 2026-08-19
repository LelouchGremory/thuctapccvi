import os
import urllib.request

# Định nghĩa danh sách các file model và đường dẫn tải trực tiếp chính thức từ opencv_zoo
MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

def download_missing_models():
  for filename, url in MODELS.items():
    if not os.path.exists(filename):
      print(f"-> Chưa tìm thấy {filename}, đang tiến hành tải về (file khá nặng, vui lòng đợi chút nhé)...")
      try:
        urllib.request.urlretrieve(url, filename)
        print(f"-> Tải thành công {filename}!")
      except Exception as e:
        print(f"-> Lỗi khi tải {filename}: {e}")
    else:
      print(f"-> Model {filename} đã có sẵn trên máy, bỏ qua.")

if __name__ == "__main__":
  print("Đang kiểm tra các file model AI...")
  download_missing_models()
  print("Hoàn tất kiểm tra!")