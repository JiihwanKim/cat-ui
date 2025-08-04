from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os
import cv2
import numpy as np
from PIL import Image
import io
import uuid
from datetime import datetime
from pathlib import Path
import shutil
from ultralytics import YOLO
import asyncio
from typing import List, Dict, Any
import torch

# --- 경로 설정 ---
# 현재 파일의 디렉토리를 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# main.py 파일 상단에 추가
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, ClassificationModel
from ultralytics.nn.modules import C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3x, DFL, RepC3
import torch.nn as nn

# FastAPI 앱 생성
app = FastAPI(title="고양이 영상 처리 API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 제공 (크롭된 이미지들)
cropped_images_dir = BASE_DIR / "cropped-images"
cropped_images_dir.mkdir(exist_ok=True)
app.mount("/cropped-images", StaticFiles(directory=str(cropped_images_dir)), name="cropped-images")

# 업로드 디렉토리 생성
uploads_dir = BASE_DIR / "uploads"
uploads_dir.mkdir(exist_ok=True)

# YOLO 모델 로드 및 최적화된 설정
class YOLO11Processor:
    def __init__(self):
        self.model = None
        self.is_model_loaded = False
        # YOLO 추론 설정 최적화
        self.inference_config = {
            'conf': 0.25,        # 신뢰도 임계값 (기본값: 0.25)
            'iou': 0.7,          # IoU 임계값 (기본값: 0.7)
            'imgsz': 640,        # 입력 이미지 크기
            'device': None,       # 자동 디바이스 선택
            'max_det': 20,       # 최대 감지 수
            'classes': [16],      # 고양이 클래스만 감지 (COCO 데이터셋)
            'verbose': False,     # 상세 출력 비활성화
            'agnostic_nms': False, # 클래스별 NMS
            'amp': True,          # 혼합 정밀도 사용
        }
        
    async def load_model(self):
        """YOLO11 모델 로드 및 최적화"""
        try:
            print("YOLO 모델 로딩 중...")
            
            # PyTorch 2.6+ 호환성을 위한 안전한 글로벌 추가
            safe_globals = [
                DetectionModel, SegmentationModel, PoseModel, ClassificationModel,
                C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3x, DFL, RepC3,
                nn.Module, nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear,
                nn.BatchNorm2d, nn.Upsample, nn.Identity
            ]
            torch.serialization.add_safe_globals(safe_globals)
            
            # YOLO 모델 로드 (yolo11n.pt 부터 시도)
            try:
                self.model = YOLO('yolo11n.pt')
                print("YOLO11n 모델 로드 완료")
            except Exception as e:
                print(f"YOLO11n 모델 로드 실패: {e}, 다른 모델을 시도합니다.")
                try:
                    self.model = YOLO('yolov8n.pt')
                    print("YOLOv8n 모델 로드 완료")
                except Exception as e2:
                    print(f"모든 모델 로드 실패: {e2}")
                    self.is_model_loaded = False
                    return
            
            # 모델 최적화 설정
            if hasattr(self.model, 'fuse'):
                self.model.fuse()  # 모델 퓨전으로 추론 속도 향상
            
            # 혼합 정밀도 설정
            if torch.cuda.is_available():
                self.model.to('cuda')
                torch.backends.cudnn.benchmark = True  # CUDA 최적화
            
            self.is_model_loaded = True
            print("YOLO 모델 로딩 및 최적화 완료")
            
        except Exception as e:
            print(f"YOLO 모델 로딩 중 예상치 못한 오류: {e}")
            self.is_model_loaded = False
    
    async def extract_frames(self, video_path: str) -> tuple:
        """비디오에서 프레임 추출"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return total_frames, fps
        except Exception as e:
            print(f"프레임 추출 오류: {e}")
            return 0, 30
    
    async def detect_cats(self, video_path: str, total_frames: int, fps: float) -> List[Dict]:
        """비디오에서 고양이 감지 (최적화된 설정 사용)"""
        if not self.is_model_loaded:
            await self.load_model()
        
        # 모델이 로드되지 않았으면 빈 리스트 반환
        if not self.is_model_loaded or self.model is None:
            print("YOLO 모델이 로드되지 않아 고양이 감지를 건너뜁니다.")
            return []
        
        cats = []
        cap = cv2.VideoCapture(video_path)
        frame_interval = max(1, int(fps / 3))  # 3초당 1프레임
        
        frame_count = 0
        batch_frames = []  # 배치 처리를 위한 프레임 저장
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                batch_frames.append((frame_count, frame))
                
                # 배치 크기가 4에 도달하거나 마지막 프레임일 때 처리
                if len(batch_frames) >= 4 or frame_count == total_frames - 1:
                    # 배치 추론 수행
                    batch_results = self.model(
                        [frame for _, frame in batch_frames],
                        **self.inference_config
                    )
                    
                    # 결과 처리
                    for i, (batch_frame_count, frame) in enumerate(batch_frames):
                        if i < len(batch_results):
                            result = batch_results[i]
                            boxes = result.boxes
                            
                            if boxes is not None:
                                for box in boxes:
                                    # 고양이 클래스 (COCO 데이터셋에서 cat은 클래스 16)
                                    if int(box.cls[0]) == 16:  # cat class
                                        confidence = float(box.conf[0])
                                        if confidence > self.inference_config['conf']:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                            
                                            # 크롭 이미지 직접 생성
                                            x = max(0, int(x1))
                                            y = max(0, int(y1))
                                            width = min(int(x2 - x1), frame.shape[1] - x)
                                            height = min(int(y2 - y1), frame.shape[0] - y)
                                            
                                            if width > 0 and height > 0:
                                                # 크롭 영역 추출
                                                cropped = frame[y:y+height, x:x+width]
                                                
                                                # 200x200으로 리사이즈
                                                cropped_resized = cv2.resize(cropped, (200, 200))
                                                
                                                # 파일명 생성
                                                filename = f"cat_{datetime.now().timestamp()}-{batch_frame_count}-{len(cats)}_frame_{batch_frame_count}.png"
                                                filepath = cropped_images_dir / filename
                                                
                                                # 파일 저장
                                                success = cv2.imwrite(str(filepath), cropped_resized)
                                                print(f"이미지 저장: {filepath}, 성공: {success}")
                                                
                                                cat_info = {
                                                    "id": f"cat-{datetime.now().timestamp()}-{batch_frame_count}-{len(cats)}",
                                                    "frame": batch_frame_count,
                                                    "timestamp": batch_frame_count / fps,
                                                    "confidence": confidence,
                                                    "x": x,
                                                    "y": y,
                                                    "width": width,
                                                    "height": height,
                                                    "filename": filename,
                                                    "url": f"/cropped-images/{filename}"
                                                }
                                                cats.append(cat_info)
                    
                    print(f"배치 처리 완료: 프레임 {batch_frame_count}/{total_frames}, 고양이 {len([c for c in cats if c['frame'] == batch_frame_count])}마리 감지")
                    batch_frames = []  # 배치 초기화
            
            frame_count += 1
        
        cap.release()
        print(f"총 {len(cats)}마리의 고양이가 감지되었습니다.")
        return cats

# YOLO 프로세서 인스턴스
yolo_processor = YOLO11Processor()

class ImageCropper:
    def __init__(self):
        self.cropped_images_dir = BASE_DIR / "cropped-images"
        self.cropped_images_dir.mkdir(exist_ok=True)
    
    async def create_cat_crop(self, cat: Dict, frame_buffer: bytes = None) -> Dict:
        """고양이 크롭 이미지 생성 (frame_buffer가 이미 처리된 경우)"""
        try:
            # 이미 크롭된 이미지가 있는 경우
            if 'filename' in cat and 'url' in cat:
                return {
                    "id": cat['id'],
                    "filename": cat['filename'],
                    "url": cat['url'],
                    "frame": cat['frame'],
                    "timestamp": cat['timestamp'],
                    "confidence": cat['confidence'],
                    "bbox": {
                        "x": cat['x'],
                        "y": cat['y'],
                        "width": cat['width'],
                        "height": cat['height']
                    }
                }
            
            # frame_buffer가 제공된 경우 (기존 방식)
            if frame_buffer:
                # 바이트를 이미지로 변환
                nparr = np.frombuffer(frame_buffer, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 바운딩 박스 좌표
                x = max(0, cat['x'])
                y = max(0, cat['y'])
                width = min(cat['width'], frame.shape[1] - x)
                height = min(cat['height'], frame.shape[0] - y)
                
                # 크롭 영역 추출
                cropped = frame[y:y+height, x:x+width]
                
                # 200x200으로 리사이즈
                cropped_resized = cv2.resize(cropped, (200, 200))
                
                # 파일명 생성
                filename = f"cat_{cat['id']}_frame_{cat['frame']}.png"
                filepath = self.cropped_images_dir / filename
                
                # 파일 저장
                cv2.imwrite(str(filepath), cropped_resized)
                
                return {
                    "id": cat['id'],
                    "filename": filename,
                    "url": f"/cropped-images/{filename}",
                    "frame": cat['frame'],
                    "timestamp": cat['timestamp'],
                    "confidence": cat['confidence'],
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                }
            
            return None
        except Exception as e:
            print(f"고양이 크롭 생성 중 오류: {e}")
            return None
    
    async def create_cat_crops(self, cats: List[Dict]) -> List[Dict]:
        """여러 고양이 크롭 이미지 생성"""
        cropped_cats = []
        
        for cat in cats:
            # 이미 크롭된 이미지가 있는 경우
            if 'filename' in cat and 'url' in cat:
                cropped_cat = await self.create_cat_crop(cat)
            # frame_buffer가 있는 경우
            elif 'frame_buffer' in cat:
                cropped_cat = await self.create_cat_crop(cat, cat['frame_buffer'])
            else:
                continue
                
            if cropped_cat:
                cropped_cats.append(cropped_cat)
        
        return cropped_cats

# 이미지 크롭퍼 인스턴스
image_cropper = ImageCropper()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 YOLO 모델 로드"""
    await yolo_processor.load_model()

@app.get("/api/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "success": True,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "yoloModelLoaded": yolo_processor.is_model_loaded,
        "inferenceConfig": yolo_processor.inference_config
    }

@app.post("/api/video/upload")
async def upload_video(video: UploadFile = File(...)):
    """영상 업로드 및 처리"""
    try:
        # 파일 확장자 검사
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")
        
        # 파일 저장
        filename = f"{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4()}{Path(video.filename).suffix}"
        filepath = uploads_dir / filename
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        video_info = {
            "originalName": video.filename,
            "filename": filename,
            "size": filepath.stat().st_size,
            "mimetype": video.content_type,
            "path": str(filepath)
        }
        
        print(f"영상 업로드 완료: {video_info}")
        
        # YOLO11 모델로 고양이 감지 및 크롭 이미지 생성
        total_frames, fps = await yolo_processor.extract_frames(str(filepath))
        detected_cats = await yolo_processor.detect_cats(str(filepath), total_frames, fps)
        cropped_cats = await image_cropper.create_cat_crops(detected_cats)
        
        # 안전한 응답 데이터 생성 (바이트 데이터 제거)
        safe_detected_cats = []
        for cat in detected_cats:
            safe_cat = {
                "id": cat.get("id"),
                "frame": cat.get("frame"),
                "timestamp": cat.get("timestamp"),
                "confidence": cat.get("confidence"),
                "x": cat.get("x"),
                "y": cat.get("y"),
                "width": cat.get("width"),
                "height": cat.get("height"),
                "filename": cat.get("filename"),
                "url": cat.get("url")
            }
            safe_detected_cats.append(safe_cat)
        
        return {
            "success": True,
            "videoInfo": video_info,
            "processingResult": {
                "totalFrames": total_frames,
                "detectedCats": safe_detected_cats,
                "croppedCats": cropped_cats,
                "message": f"{len(detected_cats)}마리의 고양이가 감지되었습니다.",
                "inferenceConfig": yolo_processor.inference_config
            }
        }
        
    except Exception as e:
        print(f"업로드 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cats/upload")
async def upload_cats(cats: List[Dict]):
    """고양이 데이터 전송"""
    try:
        print(f"{len(cats)}마리의 고양이 데이터 수신")
        
        # 안전한 고양이 데이터 생성 (바이트 데이터 제거)
        safe_cats = []
        for cat in cats:
            safe_cat = {
                "id": cat.get("id"),
                "frame": cat.get("frame"),
                "timestamp": cat.get("timestamp"),
                "confidence": cat.get("confidence"),
                "x": cat.get("x"),
                "y": cat.get("y"),
                "width": cat.get("width"),
                "height": cat.get("height")
            }
            safe_cats.append(safe_cat)
        
        # 크롭된 이미지 생성
        cropped_cats = await image_cropper.create_cat_crops(safe_cats)
        
        return {
            "success": True,
            "croppedCats": cropped_cats,
            "message": f"{len(cropped_cats)}개의 크롭된 이미지가 생성되었습니다."
        }
        
    except Exception as e:
        print(f"고양이 데이터 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cropped-cats")
async def get_cropped_cats():
    """크롭된 고양이 이미지 목록"""
    try:
        print(f"크롭된 이미지 디렉토리: {cropped_images_dir}")
        print(f"디렉토리 존재 여부: {cropped_images_dir.exists()}")
        
        image_files = list(cropped_images_dir.glob("*.png")) + list(cropped_images_dir.glob("*.jpg"))
        print(f"발견된 이미지 파일 수: {len(image_files)}")
        
        cropped_cats = []
        for file_path in image_files:
            print(f"이미지 파일: {file_path}")
            cropped_cats.append({
                "id": file_path.stem,
                "filename": file_path.name,
                "url": f"/cropped-images/{file_path.name}",
                "timestamp": datetime.now().timestamp()
            })
        
        print(f"반환할 고양이 데이터: {cropped_cats}")
        
        return {
            "success": True,
            "croppedCats": cropped_cats
        }
        
    except Exception as e:
        print(f"크롭된 이미지 목록 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos")
async def get_videos():
    """업로드된 파일 목록"""
    try:
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            video_files.extend(uploads_dir.glob(f"*{ext}"))
        
        videos = []
        for file_path in video_files:
            videos.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "uploadTime": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "path": str(file_path)
            })
        
        return {
            "success": True,
            "videos": videos
        }
        
    except Exception as e:
        print(f"비디오 파일 목록 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/videos/{filename}")
async def delete_video(filename: str):
    """파일 삭제"""
    try:
        file_path = uploads_dir / filename
        
        if file_path.exists():
            file_path.unlink()
            return {
                "success": True,
                "message": "파일이 삭제되었습니다."
            }
        else:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"파일 삭제 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/yolo/config")
async def get_yolo_config():
    """현재 YOLO 추론 설정 조회"""
    return {
        "success": True,
        "inferenceConfig": yolo_processor.inference_config,
        "modelLoaded": yolo_processor.is_model_loaded
    }

@app.post("/api/yolo/config")
async def update_yolo_config(config: Dict[str, Any]):
    """YOLO 추론 설정 업데이트"""
    try:
        # 허용된 설정 키들
        allowed_keys = {
            'conf', 'iou', 'imgsz', 'device', 'max_det', 
            'classes', 'verbose', 'agnostic_nms', 'amp'
        }
        
        # 설정 업데이트
        for key, value in config.items():
            if key in allowed_keys:
                yolo_processor.inference_config[key] = value
                print(f"YOLO 설정 업데이트: {key} = {value}")
        
        return {
            "success": True,
            "message": "YOLO 설정이 업데이트되었습니다.",
            "updatedConfig": yolo_processor.inference_config
        }
        
    except Exception as e:
        print(f"YOLO 설정 업데이트 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/yolo/reset-config")
async def reset_yolo_config():
    """YOLO 추론 설정을 기본값으로 초기화"""
    try:
        # 기본 설정으로 초기화
        yolo_processor.inference_config = {
            'conf': 0.25,        # 신뢰도 임계값
            'iou': 0.7,          # IoU 임계값
            'imgsz': 640,        # 입력 이미지 크기
            'device': None,       # 자동 디바이스 선택
            'max_det': 20,       # 최대 감지 수
            'classes': [16],      # 고양이 클래스만 감지
            'verbose': False,     # 상세 출력 비활성화
            'agnostic_nms': False, # 클래스별 NMS
            'amp': True,          # 혼합 정밀도 사용
        }
        
        return {
            "success": True,
            "message": "YOLO 설정이 기본값으로 초기화되었습니다.",
            "config": yolo_processor.inference_config
        }
        
    except Exception as e:
        print(f"YOLO 설정 초기화 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/yolo/reload")
async def reload_yolo_model():
    """YOLO 모델 재로드"""
    try:
        await yolo_processor.load_model()
        
        return {
            "success": True,
            "message": "YOLO 모델이 재로드되었습니다.",
            "modelLoaded": yolo_processor.is_model_loaded
        }
        
    except Exception as e:
        print(f"YOLO 모델 재로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/yolo/model-status")
async def get_model_status():
    """YOLO 모델 상태 확인"""
    try:
        import os
        from pathlib import Path
        
        # 모델 파일 경로 확인
        model_paths = [
            BASE_DIR / "yolo11n.pt",
            BASE_DIR / "yolo8n.pt", 
            BASE_DIR / "yolov8n.pt"
        ]
        
        model_info = []
        for model_path in model_paths:
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                model_info.append({
                    "name": model_path.name,
                    "exists": True,
                    "size_mb": round(size_mb, 2),
                    "path": str(model_path.absolute())
                })
            else:
                model_info.append({
                    "name": model_path.name,
                    "exists": False,
                    "size_mb": 0,
                    "path": str(model_path.absolute())
                })
        
        return {
            "success": True,
            "modelLoaded": yolo_processor.is_model_loaded,
            "models": model_info,
            "currentModel": "yolo11n.pt" if yolo_processor.is_model_loaded else None
        }
        
    except Exception as e:
        print(f"모델 상태 확인 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/yolo/download-model")
async def download_model(model_name: str = "yolo11n.pt"):
    """특정 YOLO 모델 다운로드"""
    try:
        print(f"{model_name} 모델 다운로드를 시작합니다...")
        
        # 모델 다운로드
        model = YOLO(model_name)
        
        # 모델 로드 상태 업데이트
        yolo_processor.model = model
        yolo_processor.is_model_loaded = True
        
        # 모델 최적화
        if hasattr(model, 'fuse'):
            model.fuse()
        
        import torch
        if torch.cuda.is_available():
            model.to('cuda')
            torch.backends.cudnn.benchmark = True
        
        return {
            "success": True,
            "message": f"{model_name} 모델이 성공적으로 다운로드되었습니다.",
            "modelName": model_name,
            "modelLoaded": True
        }
        
    except Exception as e:
        print(f"모델 다운로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🐱 고양이 영상 처리 백엔드 서버가 포트 5000에서 실행 중입니다.")
    print("📡 API 엔드포인트:")
    print("   - POST /api/video/upload (영상 업로드 및 처리)")
    print("   - POST /api/cats/upload (고양이 데이터 전송)")
    print("   - GET  /api/cropped-cats (크롭된 고양이 이미지 목록)")
    print("   - GET  /api/health (서버 상태 확인)")
    print("   - GET  /api/videos (업로드된 파일 목록)")
    print("   - DELETE /api/videos/{filename} (파일 삭제)")
    print("   - GET  /cropped-images/* (크롭된 이미지 제공)")
    print("   - GET  /api/yolo/config (YOLO 설정 조회)")
    print("   - POST /api/yolo/config (YOLO 설정 업데이트)")
    print("   - POST /api/yolo/reset-config (YOLO 설정 초기화)")
    print("   - POST /api/yolo/reload (YOLO 모델 재로드)")
    print("   - GET  /api/yolo/model-status (모델 상태 확인)")
    print("   - POST /api/yolo/download-model (모델 다운로드)")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)
