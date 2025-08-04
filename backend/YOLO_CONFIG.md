# YOLO 구성 설정 가이드

이 문서는 Ultralytics YOLO 모델의 구성 설정에 대한 상세한 설명을 제공합니다.

## 개요

YOLO 모델의 성능은 다양한 하이퍼파라미터와 설정에 의해 결정됩니다. 이 가이드는 고양이 감지 애플리케이션에 최적화된 설정을 제공합니다.

## 주요 설정 파라미터

### 1. 추론 설정 (Inference Configuration)

#### 신뢰도 임계값 (Confidence Threshold)
- **파라미터**: `conf`
- **기본값**: `0.25`
- **설명**: 감지 결과의 최소 신뢰도 임계값
- **권장값**: 
  - 높은 정확도: `0.5` - `0.7`
  - 균형잡힌 성능: `0.25` - `0.5`
  - 높은 감지율: `0.1` - `0.25`

#### IoU 임계값 (Intersection over Union)
- **파라미터**: `iou`
- **기본값**: `0.7`
- **설명**: Non-Maximum Suppression (NMS)에서 사용되는 IoU 임계값
- **권장값**:
  - 엄격한 필터링: `0.8` - `0.9`
  - 일반적인 사용: `0.6` - `0.8`
  - 관대한 필터링: `0.4` - `0.6`

#### 이미지 크기 (Image Size)
- **파라미터**: `imgsz`
- **기본값**: `640`
- **설명**: 모델에 입력되는 이미지의 크기
- **권장값**:
  - 빠른 처리: `320` - `480`
  - 균형잡힌 성능: `640`
  - 높은 정확도: `800` - `1280`

#### 최대 감지 수 (Maximum Detections)
- **파라미터**: `max_det`
- **기본값**: `20`
- **설명**: 한 프레임에서 감지할 수 있는 최대 객체 수
- **권장값**: 고양이 감지의 경우 `10` - `30`

### 2. 성능 최적화 설정

#### 혼합 정밀도 (Mixed Precision)
- **파라미터**: `amp`
- **기본값**: `True`
- **설명**: FP16과 FP32를 혼합하여 메모리 사용량 감소 및 속도 향상
- **권장값**: GPU 사용 시 `True`

#### 클래스별 NMS
- **파라미터**: `agnostic_nms`
- **기본값**: `False`
- **설명**: 클래스별로 NMS를 수행할지 여부
- **권장값**: 단일 클래스 감지 시 `False`

#### 상세 출력
- **파라미터**: `verbose`
- **기본값**: `False`
- **설명**: 추론 과정의 상세한 로그 출력
- **권장값**: 디버깅 시에만 `True`

### 3. 클래스 설정

#### 감지 클래스
- **파라미터**: `classes`
- **기본값**: `[16]` (COCO 데이터셋의 고양이 클래스)
- **설명**: 감지할 클래스 ID 목록
- **COCO 클래스 ID**:
  - 고양이: `16`
  - 개: `17`
  - 말: `18`
  - 양: `19`
  - 소: `20`

## 성능 최적화 팁

### 1. 배치 처리
- 여러 프레임을 배치로 처리하여 GPU 활용도 향상
- 배치 크기는 GPU 메모리에 따라 조정

### 2. 모델 퓨전
- `model.fuse()` 사용으로 추론 속도 향상
- 학습 후 추론 시에만 사용

### 3. CUDA 최적화
- `torch.backends.cudnn.benchmark = True` 설정
- 고정된 입력 크기 사용 시 효과적

### 4. 메모리 관리
- 불필요한 중간 결과 제거
- 배치 크기 조정으로 메모리 사용량 제어

## 설정 예시

### 높은 정확도 설정
```python
{
    'conf': 0.5,
    'iou': 0.8,
    'imgsz': 800,
    'max_det': 10,
    'classes': [16],
    'verbose': False,
    'agnostic_nms': False,
    'amp': True
}
```

### 빠른 처리 설정
```python
{
    'conf': 0.25,
    'iou': 0.6,
    'imgsz': 480,
    'max_det': 20,
    'classes': [16],
    'verbose': False,
    'agnostic_nms': False,
    'amp': True
}
```

### 균형잡힌 설정
```python
{
    'conf': 0.35,
    'iou': 0.7,
    'imgsz': 640,
    'max_det': 15,
    'classes': [16],
    'verbose': False,
    'agnostic_nms': False,
    'amp': True
}
```

## API 사용법

### 설정 조회
```bash
GET /api/yolo/config
```

### 설정 업데이트
```bash
POST /api/yolo/config
Content-Type: application/json

{
    "conf": 0.5,
    "iou": 0.8,
    "imgsz": 800
}
```

### 설정 초기화
```bash
POST /api/yolo/reset-config
```

### 모델 재로드
```bash
POST /api/yolo/reload
```

## 모니터링 및 디버깅

### 성능 메트릭
- 처리 시간 (FPS)
- 메모리 사용량
- 감지 정확도
- False Positive/Negative 비율

### 로그 분석
- 추론 시간 측정
- 배치 처리 효율성
- GPU 활용도

## 참고 자료

- [Ultralytics YOLO 공식 문서](https://docs.ultralytics.com/ko/usage/cfg/)
- [COCO 데이터셋 클래스 정보](https://cocodataset.org/#explore)
- [PyTorch 성능 최적화 가이드](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 