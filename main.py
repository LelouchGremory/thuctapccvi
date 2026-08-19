import logging
from typing import List, Optional
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
import cv2
import numpy as np

from database import engine, Base, get_db
import models

from detector import FaceDetector
from preprocessing import FacePreprocessor
from embedding import FaceEmbedder
from enrollment import EnrollmentService
from matcher import FaceMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo bảng và extension pgvector nếu chưa có
try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    logger.info("Khởi tạo Database và các bảng dữ liệu thành công.")
except Exception as e:
    logger.error(f"Lỗi khởi tạo cơ sở dữ liệu: {e}")

app = FastAPI(
    title="AI Camera Core API - Milestone Tuần 5",
    description="Core Backend API cho AI Camera: Pipeline AI, Enrollment, Matching Engine & Phân loại 4 trạng thái (Tuần 1 -> Tuần 5)",
    version="1.0.0"
)

# Khởi tạo các module lõi
face_detector = FaceDetector()
face_preprocessor = FacePreprocessor()
face_embedder = FaceEmbedder()
enrollment_service = EnrollmentService()
face_matcher = FaceMatcher()

@app.on_event("startup")
def startup_event():
    """Khi server khởi động, nạp toàn bộ vector từ PostgreSQL vào RAM Cache của Matching Engine."""
    db = next(get_db())
    try:
        count = face_matcher.load_cache_from_db(db)
        logger.info(f"Startup: Đã nạp thành công {count} vector nhân viên vào RAM Cache cho Matching Engine.")
    except Exception as e:
        logger.error(f"Startup: Lỗi khi nạp RAM cache: {e}")
    finally:
        db.close()

# Pydantic Schemas
class ErrorDetail(BaseModel):
    buoc_that_bai: str
    ly_do: str
    duong_dan_file: str

class EnrollResponse(BaseModel):
    thanh_cong: bool
    thong_bao: str
    so_luong_anh_hop_le: int
    so_luong_anh_loi: int
    chi_tiet_loi: Optional[List[ErrorDetail]] = []

class EmployeeInfo(BaseModel):
    profile_id: Optional[int] = None
    employee_id: Optional[int] = None
    ma_nhan_vien: str
    ho_ten: str
    phong_ban: Optional[str] = ""

class MatchResult(BaseModel):
    khuon_mat_thu: int
    toa_do: List[int]
    do_tin_cay_detect: float
    trang_thai: str # RECOGNIZED | UNCERTAIN | UNKNOWN | REJECTED
    do_tuong_dong: float
    nhan_vien: Optional[EmployeeInfo] = None
    ly_do: str
    multi_frame_confirmed: Optional[bool] = False

class RecognizeResponse(BaseModel):
    so_luong_khuon_mat: int
    ket_qua: List[MatchResult]

@app.get("/", summary="Check system health (Week 1)")
def check_health():
    return {
        "status": "AI Camera Core API (Tuần 5 Milestone) is running normally.",
        "ram_cache_vector_count": len(face_matcher.cache_profiles)
    }

@app.post("/api/v1/detect_and_preprocess", summary="Detect and preprocess face (Week 3)")
async def detect_and_preprocess(
    anh_camera: UploadFile = File(..., description="Camera captured image")
):
    try:
        contents = await anh_camera.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail="Could not read image file.")

    faces = face_detector.phat_hien(img)
    if not faces:
        return {"so_luong_khuon_mat": 0, "ket_qua_chi_tiet": []}

    ket_qua = []
    for i, face_info in enumerate(faces):
        is_valid, msg = face_preprocessor.kiem_tra_chat_luong(img, face_info)
        ket_qua.append({
            "khuon_mat_thu": i + 1,
            "toa_do": face_info["bbox"],
            "do_tin_cay": face_info["confidence"],
            "dat_chat_luong": is_valid,
            "ly_do": msg
        })

    return {
        "so_luong_khuon_mat": len(faces),
        "ket_qua_chi_tiet": ket_qua
    }

@app.post("/api/v1/enroll", response_model=EnrollResponse, summary="Enroll new employee face (Week 4)")
async def dang_ky_nhan_vien(
    ma_nhan_vien: str = Form(..., description="Employee Code (e.g., NV001)"),
    ho_ten: str = Form(..., description="Employee Full Name"),
    phong_ban: str = Form("Chưa phân phòng", description="Department"),
    anh_khuon_mat: List[UploadFile] = File(..., description="List of face images (Recommend 3-5 images)"),
    db: Session = Depends(get_db)
):
    danh_sach_cv2 = []
    for file in anh_khuon_mat:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                danh_sach_cv2.append(img)
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            
    if not danh_sach_cv2:
        raise HTTPException(status_code=400, detail="No valid image data provided.")
        
    ket_qua = enrollment_service.dang_ky_nhan_vien(
        db=db, 
        ma_nhan_vien=ma_nhan_vien, 
        ho_ten=ho_ten, 
        phong_ban=phong_ban, 
        danh_sach_anh=danh_sach_cv2
    )
    
    if not ket_qua["thanh_cong"]:
        raise HTTPException(status_code=400, detail=ket_qua)
    
    # Reload lại RAM Cache sau khi đăng ký nhân viên mới thành công
    face_matcher.load_cache_from_db(db)
        
    return ket_qua

@app.get("/api/v1/employees", summary="Get enrolled employees list (Week 4)")
def danh_sach_nhan_vien(db: Session = Depends(get_db)):
    nv = db.query(models.Employee).all()
    return [{"id": n.id, "ma_nhan_vien": n.ma_nhan_vien, "ho_ten": n.ho_ten, "phong_ban": n.phong_ban} for n in nv]

@app.post("/api/v1/recognize", response_model=RecognizeResponse, summary="Face Recognition & 4-State Classification (Week 5)")
async def recognize_face(
    anh_capture: UploadFile = File(..., description="Captured camera image"),
    track_id: Optional[str] = Form(None, description="Optional persistent tracking ID for multi-frame voting")
):
    """
    Endpoint Nhận diện khuôn mặt mốc Tuần 5:
    1. Face Detection (YuNet)
    2. Quality Gate & Mask Check (Preprocessing)
    3. Alignment & Feature Extraction (SFace 128-d vector)
    4. Batch Matching qua RAM Cache & Phân loại 4 trạng thái (RECOGNIZED, UNCERTAIN, UNKNOWN, REJECTED)
    5. Áp dụng Multi-frame Voting nếu có track_id
    """
    try:
        contents = await anh_capture.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise HTTPException(status_code=400, detail="Could not read image file.")

    faces = face_detector.phat_hien(img)
    if not faces:
        return {"so_luong_khuon_mat": 0, "ket_qua": []}

    ket_qua = []
    for i, face_info in enumerate(faces):
        # 1. Quality Gate Check
        is_valid, msg = face_preprocessor.kiem_tra_chat_luong(img, face_info)
        if not is_valid:
            ket_qua.append({
                "khuon_mat_thu": i + 1,
                "toa_do": face_info["bbox"],
                "do_tin_cay_detect": face_info["confidence"],
                "trang_thai": "REJECTED",
                "do_tuong_dong": 0.0,
                "nhan_vien": None,
                "ly_do": f"Không đạt Quality Gate: {msg}",
                "multi_frame_confirmed": False
            })
            continue

        # 2. Alignment & Embedding Extraction
        aligned = face_preprocessor.align_and_crop(img, face_info)
        vector = face_embedder.trich_xuat_feature(aligned)

        # 3. Matching Engine & 4-State Classification
        match_info = face_matcher.match_vector(vector)

        # 4. Multi-frame Voting (nếu được truyền track_id)
        if track_id:
            match_info = face_matcher.multi_frame_voting(track_id=track_id, current_result=match_info)

        ket_qua.append({
            "khuon_mat_thu": i + 1,
            "toa_do": face_info["bbox"],
            "do_tin_cay_detect": face_info["confidence"],
            "trang_thai": match_info["trang_thai"],
            "do_tuong_dong": match_info["do_tuong_dong"],
            "nhan_vien": match_info["nhan_vien"],
            "ly_do": match_info["ly_do"],
            "multi_frame_confirmed": match_info.get("multi_frame_confirmed", False)
        })

    return {
        "so_luong_khuon_mat": len(faces),
        "ket_qua": ket_qua
    }

@app.post("/api/v1/matcher/reload-cache", summary="Reload Matching Engine RAM Cache (Week 5)")
def reload_matcher_cache(db: Session = Depends(get_db)):
    count = face_matcher.load_cache_from_db(db)
    return {
        "thanh_cong": True,
        "so_luong_vector_loaded": count,
        "thong_bao": f"Đã làm mới bộ nhớ RAM Cache thành công với {count} vector nhân viên."
    }
