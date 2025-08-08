from fastapi import FastAPI, File, UploadFile, HTTPException, Body
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
import json
import platform

# Rich 라이브러리 추가
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

# Rich 콘솔 초기화
console = Console()

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
# 디렉토리가 없으면 생성 (parents=True로 상위 디렉토리도 생성)
cropped_images_dir.mkdir(parents=True, exist_ok=True)
console.print(f"[green]✓[/green] cropped-images 디렉토리 생성/확인: {cropped_images_dir}")

# 디렉토리가 실제로 존재하는지 확인
if not cropped_images_dir.exists():
    console.print(f"[red]⚠[/red] 경고: cropped-images 디렉토리를 생성할 수 없습니다: {cropped_images_dir}")
else:
    console.print(f"[green]✓[/green] cropped-images 디렉토리 확인됨: {cropped_images_dir}")

app.mount("/cropped-images", StaticFiles(directory=str(cropped_images_dir)), name="cropped-images")

# 업로드 디렉토리 생성
uploads_dir = BASE_DIR / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
console.print(f"[green]✓[/green] uploads 디렉토리 생성/확인: {uploads_dir}")

# 관련 API 엔드포인트들도 제거
# @app.get("/api/caregiver/config")
# @app.post("/api/caregiver/config")
# @app.post("/api/caregiver/cleanup-checkpoints")

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
            console.print("[blue]🔄[/blue] YOLO 모델 로딩 중...")
            
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
                console.print("[green]✓[/green] YOLO11n 모델 로드 완료")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] YOLO11n 모델 로드 실패: {e}, 다른 모델을 시도합니다.")
                try:
                    self.model = YOLO('yolov8n.pt')
                    console.print("[green]✓[/green] YOLOv8n 모델 로드 완료")
                except Exception as e2:
                    console.print(f"[red]✗[/red] 모든 모델 로드 실패: {e2}")
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
            console.print("[green]✓[/green] YOLO 모델 로딩 및 최적화 완료")
            
        except Exception as e:
            console.print(f"[red]✗[/red] YOLO 모델 로딩 중 예상치 못한 오류: {e}")
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
            console.print(f"[red]✗[/red] 프레임 추출 오류: {e}")
            return 0, 30
    
    async def detect_cats(self, video_path: str, total_frames: int, fps: float, video_filename: str = None) -> List[Dict]:
        """비디오에서 고양이 감지 (최적화된 설정 사용)"""
        if not self.is_model_loaded:
            await self.load_model()
        
        # 모델이 로드되지 않았으면 빈 리스트 반환
        if not self.is_model_loaded or self.model is None:
            console.print("[red]✗[/red] YOLO 모델이 로드되지 않아 고양이 감지를 건너뜁니다.")
            return []
        
        # cropped-images 디렉토리 확인 및 생성
        cropped_images_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]ℹ[/blue] detect_cats에서 cropped-images 디렉토리 확인: {cropped_images_dir}")
        
        # 영상 파일명에서 확장자 제거
        if video_filename:
            video_name = Path(video_filename).stem
        else:
            video_name = Path(video_path).stem
        
        console.print(f"[blue]ℹ[/blue] 영상 이름: {video_name}, FPS: {fps}, 총 프레임: {total_frames}")
        
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
                                                
                                                # 파일명 생성 (영상 파일명 기반)
                                                filename = f"{video_name}_frame_{batch_frame_count}_{len(cats)}.png"
                                                filepath = cropped_images_dir / filename
                                                
                                                # 파일 저장 전 디렉토리 재확인
                                                filepath.parent.mkdir(parents=True, exist_ok=True)
                                                
                                                # 파일 저장
                                                success = cv2.imwrite(str(filepath), cropped_resized)
                                                console.print(f"[blue]ℹ[/blue] 이미지 저장: {filepath}, 성공: {success}")
                                                
                                                # 정확한 시간 계산
                                                timestamp = batch_frame_count / fps
                                                minutes = int(timestamp // 60)
                                                seconds = int(timestamp % 60)
                                                time_str = f"{minutes:02d}:{seconds:02d}"
                                                
                                                cat_info = {
                                                    "id": f"{video_name}-{batch_frame_count}-{len(cats)}",
                                                    "frame": batch_frame_count,
                                                    "timestamp": timestamp,
                                                    "timeString": time_str,
                                                    "confidence": confidence,
                                                    "x": x,
                                                    "y": y,
                                                    "width": width,
                                                    "height": height,
                                                    "filename": filename,
                                                    "url": f"/cropped-images/{filename}",
                                                    "videoName": video_name,
                                                    "fps": fps,
                                                    "totalFrames": total_frames
                                                }
                                                
                                                console.print(f"[blue]ℹ[/blue] 고양이 정보 생성: {cat_info}")
                                                cats.append(cat_info)
                    
                    console.print(f"[blue]ℹ[/blue] 배치 처리 완료: 프레임 {batch_frame_count}/{total_frames}, 고양이 {len([c for c in cats if c['frame'] == batch_frame_count])}마리 감지")
                    batch_frames = []  # 배치 초기화
            
            frame_count += 1
        
        cap.release()
        console.print(f"[green]✓[/green] 총 {len(cats)}마리의 고양이가 감지되었습니다.")
        console.print(f"[blue]ℹ[/blue] 최종 고양이 데이터: {cats}")
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
            console.print(f"[red]✗[/red] 고양이 크롭 생성 중 오류: {e}")
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

# 그룹 정보 저장 파일 경로
groups_file = BASE_DIR / "cat_groups.json"

# 그룹 정보 관리 함수들
def load_cat_groups() -> Dict[str, Any]:
    """저장된 고양이 그룹 정보를 로드"""
    try:
        console.print(f"[blue]ℹ[/blue] === 그룹 정보 로드 함수 호출됨 ===")
        console.print(f"[blue]ℹ[/blue] 파일 경로: {groups_file}")
        console.print(f"[blue]ℹ[/blue] 파일 존재 여부: {groups_file.exists()}")
        
        if groups_file.exists():
            console.print(f"[blue]ℹ[/blue] 파일 크기: {groups_file.stat().st_size} bytes")
            with open(groups_file, 'r', encoding='utf-8') as f:
                content = f.read()
                console.print(f"[blue]ℹ[/blue] 파일 내용: {content}")
                data = json.loads(content)
                console.print(f"[blue]ℹ[/blue] 로드된 데이터: {data}")
                
                # 기존 형식과 새로운 형식 모두 지원
                if isinstance(data, dict):
                    # 새로운 형식: {"groups": {...}, "profiles": {...}}
                    if "groups" in data and "profiles" in data:
                        return data
                    # 기존 형식: {"cat_id": "group_name"}
                    else:
                        # 기존 형식을 새로운 형식으로 마이그레이션
                        console.print("[yellow]⚠[/yellow] 기존 형식을 새로운 형식으로 마이그레이션합니다.")
                        migrated_data = {
                            "groups": data,
                            "profiles": {}
                        }
                        # 마이그레이션된 데이터를 저장
                        save_cat_groups(migrated_data)
                        return migrated_data
                else:
                    return {"groups": {}, "profiles": {}}
        else:
            console.print("[yellow]⚠[/yellow] 파일이 존재하지 않음")
            return {"groups": {}, "profiles": {}}
    except Exception as e:
        console.print(f"[red]✗[/red] 그룹 정보 로드 실패: {e}")
        return {"groups": {}, "profiles": {}}

def save_cat_groups(groups_data: Dict[str, Any]):
    """고양이 그룹 정보를 저장"""
    try:
        console.print(f"[blue]ℹ[/blue] === 그룹 정보 저장 함수 호출됨 ===")
        console.print(f"[blue]ℹ[/blue] 저장할 데이터: {groups_data}")
        console.print(f"[blue]ℹ[/blue] 파일 경로: {groups_file}")
        console.print(f"[blue]ℹ[/blue] 파일 경로 타입: {type(groups_file)}")
        console.print(f"[blue]ℹ[/blue] 파일 경로 존재 여부: {groups_file.exists()}")
        
        # 디렉토리가 없으면 생성
        groups_file.parent.mkdir(exist_ok=True)
        console.print(f"[blue]ℹ[/blue] 디렉토리 생성 완료: {groups_file.parent}")
        
        # groups와 profiles 데이터 추출
        groups = groups_data.get("groups", {})
        profiles = groups_data.get("profiles", {})
        
        # groups에 존재하는 그룹만 profiles에서 유지
        valid_groups = set(groups.values())
        filtered_profiles = {}
        
        for group_name, profile_filename in profiles.items():
            if group_name in valid_groups:
                filtered_profiles[group_name] = profile_filename
                console.print(f"[blue]ℹ[/blue] 프로필 유지: {group_name} -> {profile_filename}")
            else:
                console.print(f"[yellow]⚠[/yellow] 프로필 제거: {group_name} (groups에 존재하지 않음)")
        
        # 새로운 형식으로 저장
        save_data = {
            "groups": groups,
            "profiles": filtered_profiles
        }
        
        console.print(f"[blue]ℹ[/blue] 필터링된 프로필: {filtered_profiles}")
        console.print(f"[blue]ℹ[/blue] 저장할 최종 데이터: {save_data}")
        
        with open(groups_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✓[/green] 파일 저장 완료: {groups_file}")
        console.print(f"[blue]ℹ[/blue] 파일 크기: {groups_file.stat().st_size} bytes")
        console.print("[green]✓[/green] 그룹 정보 저장 완료")
    except Exception as e:
        console.print(f"[red]✗[/red] 그룹 정보 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="그룹 정보 저장 실패")

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
async def upload_video(videos: List[UploadFile] = File(...)):
    """여러 영상 업로드 및 처리"""
    try:
        all_results = []
        total_videos = len(videos)
        
        for video_index, video in enumerate(videos):
            # 파일 확장자 검사
            if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                console.print(f"[yellow]⚠[/yellow] 지원되지 않는 파일 형식 건너뛰기: {video.filename}")
                continue
            
            console.print(f"[blue]ℹ[/blue] === 영상 {video_index + 1}/{total_videos} 처리 시작: {video.filename} ===")
            
            # 원본 파일명을 그대로 사용하고 기존 파일이 있으면 덮어쓰기
            filename = video.filename
            filepath = uploads_dir / filename
            
            # 기존 파일이 있으면 덮어쓰기
            if filepath.exists():
                console.print(f"[yellow]⚠[/yellow] 기존 파일 덮어쓰기: {filename}")
            
            console.print(f"[blue]ℹ[/blue] 파일 업로드 중: {filename}")
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            video_info = {
                "originalName": video.filename,
                "filename": filename,
                "size": filepath.stat().st_size,
                "mimetype": video.content_type,
                "path": str(filepath)
            }
            
            console.print(f"[green]✓[/green] 영상 업로드 완료: {video_info}")
            console.print(f"[blue]ℹ[/blue] 프레임 추출 시작: {filename}")
            
            # YOLO11 모델로 고양이 감지 및 크롭 이미지 생성
            total_frames, fps = await yolo_processor.extract_frames(str(filepath))
            console.print(f"[green]✓[/green] 프레임 추출 완료: {total_frames} 프레임, {fps} FPS")
            
            console.print(f"[blue]ℹ[/blue] 고양이 감지 시작: {filename}")
            detected_cats = await yolo_processor.detect_cats(str(filepath), total_frames, fps, filename)
            console.print(f"[green]✓[/green] 고양이 감지 완료: {len(detected_cats)}마리 감지")
            
            console.print(f"[blue]ℹ[/blue] 이미지 크롭 시작: {filename}")
            cropped_cats = await image_cropper.create_cat_crops(detected_cats)
            console.print(f"[green]✓[/green] 이미지 크롭 완료: {len(cropped_cats)}개 이미지 생성")
            
            console.print(f"[green]✓[/green] === 영상 {video_index + 1}/{total_videos} 처리 완료: {video.filename} ===")
            
            # 안전한 응답 데이터 생성 (바이트 데이터 제거)
            safe_detected_cats = []
            for cat in detected_cats:
                safe_cat = {
                    "id": cat.get("id"),
                    "frame": cat.get("frame"),
                    "timestamp": cat.get("timestamp"),
                    "timeString": cat.get("timeString"),
                    "confidence": cat.get("confidence"),
                    "x": cat.get("x"),
                    "y": cat.get("y"),
                    "width": cat.get("width"),
                    "height": cat.get("height"),
                    "filename": cat.get("filename"),
                    "url": cat.get("url"),
                    "videoName": cat.get("videoName"),
                    "fps": cat.get("fps"),
                    "totalFrames": cat.get("totalFrames")
                }
                safe_detected_cats.append(safe_cat)
            
            console.print(f"[blue]ℹ[/blue] 안전한 고양이 데이터: {safe_detected_cats}")
            
            result = {
                "videoInfo": video_info,
                "processingResult": {
                    "totalFrames": total_frames,
                    "detectedCats": safe_detected_cats,
                    "croppedCats": cropped_cats,
                    "message": f"{len(detected_cats)}마리의 고양이가 감지되었습니다.",
                    "inferenceConfig": yolo_processor.inference_config
                }
            }
            all_results.append(result)
        
        # 전체 결과 요약
        total_cats = sum(len(result["processingResult"]["detectedCats"]) for result in all_results)
        total_cropped = sum(len(result["processingResult"]["croppedCats"]) for result in all_results)
        
        final_response = {
            "success": True,
            "results": all_results,
            "summary": {
                "totalVideos": len(all_results),
                "totalCats": total_cats,
                "totalCropped": total_cropped,
                "message": f"{len(all_results)}개 영상에서 총 {total_cats}마리의 고양이가 감지되었습니다."
            }
        }
        
        console.print(f"[blue]ℹ[/blue] 최종 응답 데이터: {final_response}")
        
        return final_response
        
    except Exception as e:
        console.print(f"[red]✗[/red] 업로드 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cats/upload")
async def upload_cats(cats: List[Dict]):
    """고양이 데이터 전송"""
    try:
        console.print(f"[blue]ℹ[/blue] {len(cats)}마리의 고양이 데이터 수신")
        
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
        console.print(f"[red]✗[/red] 고양이 데이터 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cropped-cats")
async def get_cropped_cats():
    """크롭된 고양이 이미지 목록"""
    try:
        console.print(f"[blue]ℹ[/blue] 크롭된 이미지 디렉토리: {cropped_images_dir}")
        console.print(f"[blue]ℹ[/blue] 디렉토리 존재 여부: {cropped_images_dir.exists()}")
        
        image_files = list(cropped_images_dir.glob("*.png")) + list(cropped_images_dir.glob("*.jpg"))
        console.print(f"[blue]ℹ[/blue] 발견된 이미지 파일 수: {len(image_files)}")
        
        # 저장된 그룹 정보 로드
        groups_data = load_cat_groups()
        groups = groups_data.get("groups", {})
        profiles = groups_data.get("profiles", {})
        console.print(f"[blue]ℹ[/blue] 로드된 그룹 정보: {groups}")
        console.print(f"[blue]ℹ[/blue] 로드된 프로필 정보: {profiles}")
        
        cropped_cats = []
        for file_path in image_files:
            console.print(f"[blue]ℹ[/blue] 이미지 파일: {file_path}")
            
            # 파일명에서 정보 추출 (예: video_name_frame_52_0.png)
            filename = file_path.stem
            parts = filename.split('_')
            
            # 기본값 설정
            frame = 0
            timestamp = 0
            timeString = '00:00'
            confidence = 0.8
            x, y, width, height = 0, 0, 200, 200
            videoName = 'unknown'
            fps = 30
            totalFrames = 0
            
            # 파일명 패턴 분석
            if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
                try:
                    # video_name_frame_52_0 형태에서 정보 추출
                    frame = int(parts[-2])
                    cat_index = int(parts[-1])
                    
                    # video_name 추출 (frame과 cat_index 제외)
                    video_name_parts = parts[:-2]
                    videoName = '_'.join(video_name_parts)
                    
                    # FPS 추정 (일반적인 값)
                    fps = 14.285714285714286  # 일반적인 FPS
                    timestamp = frame / fps
                    
                    # 시간 문자열을 더 정확하게 계산
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    timeString = f"{minutes:02d}:{seconds:02d}"
                    
                    # 신뢰도는 기본값 사용 (실제로는 파일에서 추출할 수 없음)
                    confidence = 0.8
                    
                    console.print(f"[blue]ℹ[/blue] 파싱된 정보: frame={frame}, timestamp={timestamp}, timeString={timeString}, videoName={videoName}")
                    
                except (ValueError, IndexError) as e:
                    console.print(f"[red]✗[/red] 파일명 파싱 오류: {filename}, 오류: {e}")
                    # 기본값 사용
                    pass
            else:
                console.print(f"[yellow]⚠[/yellow] 파일명 패턴이 맞지 않음: {filename}")
            
            # 고양이 정보 생성
            cat_id = filename
            cat_info = {
                'id': cat_id,
                'filename': file_path.name,
                'url': f'/cropped-images/{file_path.name}',
                'frame': frame,
                'timestamp': timestamp,
                'timeString': timeString,
                'confidence': confidence,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'videoName': videoName,
                'fps': fps,
                'totalFrames': totalFrames
            }
            
            # 그룹 정보 추가
            if cat_id in groups:
                cat_info['group'] = groups[cat_id]
            
            cropped_cats.append(cat_info)
        
        console.print(f"[blue]ℹ[/blue] 반환할 고양이 데이터: {cropped_cats}")
        
        # 프론트엔드가 기대하는 형식으로 응답
        return {
            "success": True,
            "croppedCats": cropped_cats,
            "groups": groups,
            "profiles": profiles,
            "message": f"총 {len(cropped_cats)}마리의 고양이 데이터를 로드했습니다."
        }
        
    except Exception as e:
        console.print(f"[red]✗[/red] 크롭된 고양이 목록 조회 실패: {e}")
        return {
            "success": False,
            "croppedCats": [],
            "groups": {},
            "message": f"데이터 로드 실패: {str(e)}"
        }

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
        console.print(f"[red]✗[/red] 비디오 파일 목록 조회 중 오류: {e}")
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
        console.print(f"[red]✗[/red] 파일 삭제 중 오류: {e}")
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
                console.print(f"[blue]ℹ[/blue] YOLO 설정 업데이트: {key} = {value}")
        
        return {
            "success": True,
            "message": "YOLO 설정이 업데이트되었습니다.",
            "updatedConfig": yolo_processor.inference_config
        }
        
    except Exception as e:
        console.print(f"[red]✗[/red] YOLO 설정 업데이트 중 오류: {e}")
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
        console.print(f"[red]✗[/red] YOLO 설정 초기화 중 오류: {e}")
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
        console.print(f"[red]✗[/red] YOLO 모델 재로드 중 오류: {e}")
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
        console.print(f"[red]✗[/red] 모델 상태 확인 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/yolo/download-model")
async def download_model(model_name: str = "yolo11n.pt"):
    """특정 YOLO 모델 다운로드"""
    try:
        console.print(f"[blue]ℹ[/blue] {model_name} 모델 다운로드를 시작합니다...")
        
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
        console.print(f"[red]✗[/red] 모델 다운로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def prepare_training_data():
    """cat_groups.json을 읽어서 학습용 데이터셋 구조 생성"""
    try:
        console.print("[blue]ℹ[/blue] === 학습 데이터 준비 시작 ===")
        
        # cat_groups.json 로드
        groups_data = load_cat_groups()
        groups = groups_data.get("groups", {})
        
        if not groups:
            raise ValueError("학습할 그룹 데이터가 없습니다.")
        
        # 그룹별로 이미지 분류
        group_images = {}
        for cat_id, group_name in groups.items():
            if group_name not in group_images:
                group_images[group_name] = []
            
            # cropped-images 디렉토리에서 해당 이미지 찾기 (.jpg와 .png 모두 확인)
            image_path_jpg = cropped_images_dir / f"{cat_id}.jpg"
            image_path_png = cropped_images_dir / f"{cat_id}.png"
            
            if image_path_jpg.exists():
                group_images[group_name].append(str(image_path_jpg))
            elif image_path_png.exists():
                group_images[group_name].append(str(image_path_png))
        
        # 각 그룹별 이미지 수 확인
        console.print(f"[blue]ℹ[/blue] 그룹별 이미지 수:")
        for group_name, images in group_images.items():
            console.print(f"[blue]ℹ[/blue]   {group_name}: {len(images)}개")
        
        # 최소 3개 이상의 이미지가 있는 그룹만 사용
        valid_groups = {name: images for name, images in group_images.items() if len(images) >= 3}
        
        if not valid_groups:
            raise ValueError("학습 가능한 그룹이 없습니다. 각 그룹당 최소 3개 이상의 이미지가 필요합니다.")
        
        # datasets 디렉토리 생성
        datasets_dir = BASE_DIR / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        
        # 각 그룹별로 디렉토리 생성 및 이미지 복사
        for i, (group_name, images) in enumerate(valid_groups.items()):
            group_dir = datasets_dir / str(i + 1)  # 1부터 시작하는 클래스 번호
            group_dir.mkdir(exist_ok=True)
            
            for j, image_path in enumerate(images):
                # 원본 파일명을 그대로 사용
                new_filename = Path(image_path).name
                new_path = group_dir / new_filename
                
                # 이미지 복사
                import shutil
                shutil.copy2(image_path, new_path)
                console.print(f"[blue]ℹ[/blue] 복사됨: {image_path} -> {new_path}")
        
        console.print(f"[green]✓[/green] === 학습 데이터 준비 완료 ===")
        console.print(f"[blue]ℹ[/blue] 유효한 그룹 수: {len(valid_groups)}")
        console.print(f"[blue]ℹ[/blue] 총 이미지 수: {sum(len(images) for images in valid_groups.values())}")
        
        return {
            "success": True,
            "num_groups": len(valid_groups),
            "total_images": sum(len(images) for images in valid_groups.values()),
            "group_info": {name: len(images) for name, images in valid_groups.items()}
        }
        
    except Exception as e:
        console.print(f"[red]✗[/red] 학습 데이터 준비 중 오류: {e}")
        raise e

# cleanup_old_checkpoints 함수 제거 또는 단순화
def cleanup_old_checkpoints(save_dir, max_files=5):
    """체크포인트 관리 - 백업 없이 단순히 완료 메시지만 출력"""
    try:
        console.print(f"[green]✓[/green] 체크포인트 관리 완료")
        
    except Exception as e:
        console.print(f"[red]✗[/red] 체크포인트 관리 중 오류: {e}")

@app.get("/api/yolo/download-checkpoint")
async def download_checkpoint():
    """학습 완료된 체크포인트 파일 다운로드"""
    try:
        # 백엔드의 checkpoints 디렉토리에서 파일 찾기
        checkpoints_dir = BASE_DIR / "checkpoints"
        
        # best_checkpoint 파일 찾기 (타임스탬프 없이)
        checkpoint_file = checkpoints_dir / "best_model_resnet50_contrastive.pth"
        
        if not checkpoint_file.exists():
            return {
                "success": False,
                "error": "체크포인트 파일을 찾을 수 없습니다.",
                "message": f"학습이 완료되지 않았거나 체크포인트 파일이 생성되지 않았습니다. 경로: {checkpoint_file}"
            }
        
        # 파일 정보
        file_size = checkpoint_file.stat().st_size
        file_name = checkpoint_file.name
        
        # 파일 내용을 base64로 인코딩하여 반환
        import base64
        with open(checkpoint_file, 'rb') as f:
            file_content = f.read()
            file_base64 = base64.b64encode(file_content).decode('utf-8')
        
        return {
            "success": True,
            "checkpoint_file": file_name,
            "file_size": file_size,
            "file_content": file_base64,  # base64 인코딩된 파일 내용
            "message": "체크포인트 파일 다운로드 준비 완료"
        }
        
    except Exception as e:
        console.print(f"[red]✗[/red] 체크포인트 다운로드 준비 중 오류: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "체크포인트 다운로드 준비에 실패했습니다."
        }

async def train_model():
    """ResNet50 기반 분류 모델 학습 수행"""
    try:
        console.print("[blue]ℹ[/blue] === 실제 모델 학습 시작 ===")

        # 1) 학습 데이터 준비
        console.print("[blue]ℹ[/blue] 1. 학습 데이터 준비 중...")
        data_result = prepare_training_data()
        if not data_result["success"]:
            return data_result

        # 2) train_resnet50.py 실행
        console.print("[blue]ℹ[/blue] 2. ResNet50 모델 학습 시작...")
        import subprocess, sys
        train_script = BASE_DIR / "train_resnet50.py"
        if not train_script.exists():
            return {"success": False, "error": "train_resnet50.py 파일을 찾을 수 없습니다.", "message": "학습 스크립트가 존재하지 않습니다."}

        # 환경변수 설정 (CUDA 컨텍스트 전달)
        env = os.environ.copy()
        # CUDA 관련 환경변수 명시적 설정
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',  # 첫 번째 GPU 사용
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',  # 메모리 할당 최적화
        })

        # GPU 사용 가능 여부 확인 및 설정
        if torch.cuda.is_available():
            env['CUDA_LAUNCH_BLOCKING'] = '1'  # 디버깅용 (선택사항)
            console.print(f"[blue]ℹ[/blue] GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            console.print("[blue]ℹ[/blue] CPU 사용")

        process = subprocess.Popen(
            [sys.executable, "-u", str(train_script)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 라인 버퍼링
            env=env  # 환경변수 전달
        )

        # 실시간 콘솔 출력 + 버퍼에 누적
        collected = []
        for line in process.stdout:
            line = line.rstrip("\n")
            if line:
                console.print(line)
                collected.append(line)

        returncode = process.wait()
        stdout_text = "\n".join(collected)

        if returncode != 0:
            console.print(f"[red]✗[/red] 학습 스크립트 실행 실패 (returncode={returncode})")
            return {
                "success": False,
                "error": f"학습 스크립트 실행 실패 (returncode={returncode})",
                "message": "모델 학습 중 오류가 발생했습니다.",
                "stdout": stdout_text
            }

        console.print("[green]✓[/green] === 모델 학습 완료 ===")
        console.print(f"[green]✓[/green] 학습 출력(요약): 마지막 {min(50, len(collected))}줄 표시")
        for l in collected[-50:]:
            console.print(l)

        cleanup_old_checkpoints(BASE_DIR / "checkpoints", max_files=3)
        return {"success": True, "message": "ResNet50 모델 학습이 완료되었습니다.", "training_output": stdout_text}

    except Exception as e:
        console.print(f"[red]✗[/red] 모델 학습 중 오류: {e}")
        return {"success": False, "error": str(e), "message": "모델 학습 중 오류가 발생했습니다."}

@app.post("/api/yolo/teach-model")
async def teach_model(teaching_data: Dict[str, Any]):
    """AI 모델에게 고양이를 알려주기 (실제 학습)"""
    try:
        console.print("[blue]ℹ[/blue] === 실제 모델 학습 시작 ===")
        console.print(f"[blue]ℹ[/blue] 학습 데이터: {teaching_data}")
        
        # 선택된 고양이 ID들
        selected_cat_ids = teaching_data.get('selected_cat_ids', [])
        cat_names = teaching_data.get('cat_names', {})
        
        console.print(f"[blue]ℹ[/blue] 선택된 고양이 수: {len(selected_cat_ids)}")
        console.print(f"[blue]ℹ[/blue] 고양이 이름 정보: {cat_names}")
        
        if not selected_cat_ids:
            return {
                "success": False,
                "message": "모든 고양이가 미지정 상태입니다."
            }
        
        # 실제 학습 수행
        train_result = await train_model()
        
        if not train_result["success"]:
            return train_result
        
        # 체크포인트 다운로드 정보 추가
        checkpoint_info = await download_checkpoint()
        
        return {
            "success": True,
            "message": f"{len(set(cat_names.values()))}개 그룹의 {len(selected_cat_ids)}마리 고양이로 모델을 학습시켰습니다.",
            "training_result": train_result,
            "checkpoint_info": checkpoint_info
        }
        
    except Exception as e:
        console.print(f"[red]✗[/red] 모델 학습 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cat-groups")
async def get_cat_groups():
    """저장된 고양이 그룹 정보를 반환"""
    console.print("[blue]ℹ[/blue] === 그룹 정보 로드 API 호출됨 ===")
    try:
        groups_data = load_cat_groups()
        groups = groups_data.get("groups", {})
        profiles = groups_data.get("profiles", {})
        console.print(f"[blue]ℹ[/blue] 로드된 그룹: {groups}")
        console.print(f"[blue]ℹ[/blue] 로드된 프로필: {profiles}")
        return {
            "success": True,
            "groups": groups,
            "profiles": profiles,
            "message": "그룹 정보를 성공적으로 로드했습니다."
        }
    except Exception as e:
        console.print(f"[red]✗[/red] === 그룹 정보 로드 API 오류: {e} ===")
        return {
            "success": False,
            "error": str(e),
            "message": "그룹 정보 로드에 실패했습니다."
        }

@app.post("/api/cat-groups")
async def save_cat_groups_api(groups_data: Dict[str, Any]):
    """고양이 그룹 정보를 저장"""
    console.print("[blue]ℹ[/blue] === 그룹 정보 저장 API 호출됨 ===")
    console.print(f"[blue]ℹ[/blue] 요청 데이터: {groups_data}")
    console.print(f"[blue]ℹ[/blue] 데이터 타입: {type(groups_data)}")
    
    try:
        # save_cat_groups는 동기 함수이므로 await 없이 호출
        save_cat_groups(groups_data)
        console.print("[green]✓[/green] === 그룹 정보 저장 완료 ===")
        return {
            "success": True,
            "message": "그룹 정보가 성공적으로 저장되었습니다."
        }
    except Exception as e:
        console.print(f"[red]✗[/red] === 그룹 정보 저장 API 오류: {e} ===")
        return {
            "success": False,
            "error": str(e),
            "message": "그룹 정보 저장에 실패했습니다."
        }

@app.get("/api/statistics")
async def get_statistics():
    """통계 정보 조회"""
    try:
        # 업로드된 영상 수
        video_files = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi")) + list(uploads_dir.glob("*.mov"))
        video_count = len(video_files)
        
        # cropped-images 수
        cropped_images = list(cropped_images_dir.glob("*.jpg")) + list(cropped_images_dir.glob("*.png"))
        cropped_count = len(cropped_images)
        
        # 라벨링된 이미지 수 및 고양이별 이미지 수
        groups_data = load_cat_groups()
        groups = groups_data.get("groups", {})
        profiles = groups_data.get("profiles", {})
        labeled_count = len(groups)
        
        # 고양이별 이미지 수 계산
        cat_image_counts = {}
        for label in groups.values():
            if label in cat_image_counts:
                cat_image_counts[label] += 1
            else:
                cat_image_counts[label] = 1
        
        # 업로드된 영상 리스트
        video_list = []
        for video_file in video_files:
            video_info = {
                "filename": video_file.name,
                "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2),
                "upload_date": datetime.fromtimestamp(video_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
            video_list.append(video_info)
        
        return {
            "success": True,
            "statistics": {
                "video_count": video_count,
                "cropped_count": cropped_count,
                "labeled_count": labeled_count,
                "label_counts": cat_image_counts,
                "profile_count": len(profiles)
            },
            "video_list": video_list,
            "profiles": profiles
        }
    except Exception as e:
        console.print(f"[red]✗[/red] 통계 정보 조회 실패: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/yolo/download-checkpoint-file/{filename}")
async def download_checkpoint_file(filename: str):
    """체크포인트 파일 다운로드"""
    try:
        # 백엔드의 checkpoints 디렉토리에서 파일 찾기
        checkpoints_dir = BASE_DIR / "checkpoints"
        file_path = checkpoints_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        console.print(f"[red]✗[/red] 체크포인트 파일 다운로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/start")
@app.post("/api/yolo/train-model")  # deprecated: 하위 호환 유지
async def start_training():
    """ResNet50 기반 분류 모델 학습 시작"""
    return await train_model()

if __name__ == "__main__":
    console.print(" 고양이 영상 처리 백엔드 서버가 포트 5000에서 실행 중입니다.")
    console.print("📡 API 엔드포인트:")
    console.print("   - POST /api/video/upload (영상 업로드 및 처리)")
    console.print("   - POST /api/cats/upload (고양이 데이터 전송)")
    console.print("   - GET  /api/cropped-cats (크롭된 고양이 이미지 목록)")
    console.print("   - GET  /api/health (서버 상태 확인)")
    console.print("   - GET  /api/videos (업로드된 파일 목록)")
    console.print("   - DELETE /api/videos/{filename} (파일 삭제)")
    console.print("   - GET  /cropped-images/* (크롭된 이미지 제공)")
    console.print("   - GET  /api/yolo/config (YOLO 설정 조회)")
    console.print("   - POST /api/yolo/config (YOLO 설정 업데이트)")
    console.print("   - POST /api/yolo/reset-config (YOLO 설정 초기화)")
    console.print("   - POST /api/yolo/reload (YOLO 모델 재로드)")
    console.print("   - GET  /api/yolo/model-status (모델 상태 확인)")
    console.print("   - POST /api/yolo/download-model (모델 다운로드)")
    console.print("   - POST /api/yolo/teach-model (모델 학습)")
    console.print("   - GET  /api/cat-groups (그룹 정보 조회)")
    console.print("   - POST /api/cat-groups (그룹 정보 저장)")
    console.print("   - GET  /api/train/progress (학습 진행 상태 확인)")
    console.print("   - POST /api/train/start (학습 시작)")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)


