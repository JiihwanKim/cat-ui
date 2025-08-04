# 🐱 고양이 영상 처리 애플리케이션

YOLO11 모델을 사용하여 영상에서 고양이를 자동으로 감지하고 크롭하는 웹 애플리케이션입니다.

## ✨ 주요 기능

- **영상 업로드**: MP4, AVI, MOV, MKV, WEBM 형식 지원
- **고양이 감지**: YOLO11 모델을 사용한 실시간 고양이 감지
- **자동 크롭**: 감지된 고양이를 자동으로 크롭하여 갤러리 생성
- **YOLO 설정**: 추론 파라미터를 실시간으로 조정 가능
- **반응형 UI**: 모바일과 데스크톱에서 모두 사용 가능

## 🚀 시작하기

### 백엔드 설정

1. **Python 환경 설정**
```bash
cd backend
pip install -r requirements.txt
```

2. **서버 실행**
```bash
python main.py
```

서버는 `http://localhost:5000`에서 실행됩니다.

**📦 모델 다운로드**: 서버 시작 시 YOLO 모델이 자동으로 다운로드됩니다. 만약 다운로드에 실패하면 웹 인터페이스에서 수동으로 다운로드할 수 있습니다.

### 프론트엔드 설정

1. **의존성 설치**
```bash
npm install
```

2. **개발 서버 실행**
```bash
npm start
```

애플리케이션은 `http://localhost:3000`에서 실행됩니다.

## 📁 프로젝트 구조

```
cat_ui/
├── backend/
│   ├── main.py              # FastAPI 백엔드 서버
│   ├── requirements.txt      # Python 의존성
│   ├── uploads/             # 업로드된 영상 파일
│   ├── cropped-images/      # 크롭된 고양이 이미지
│   └── YOLO_CONFIG.md       # YOLO 설정 가이드
├── src/
│   ├── components/
│   │   ├── VideoUploader.js # 영상 업로드 컴포넌트
│   │   ├── CatCropper.js    # 고양이 크롭 컴포넌트
│   │   ├── CatGallery.js    # 갤러리 컴포넌트
│   │   ├── YOLOConfig.js    # YOLO 설정 컴포넌트
│   │   └── YOLOConfig.css   # YOLO 설정 스타일
│   ├── App.js               # 메인 앱 컴포넌트
│   └── index.js             # 앱 진입점
└── README.md                # 프로젝트 문서
```

## ⚙️ YOLO 설정

### 주요 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `conf` | 0.25 | 신뢰도 임계값 (0.1-0.9) |
| `iou` | 0.7 | IoU 임계값 (0.3-0.9) |
| `imgsz` | 640 | 입력 이미지 크기 |
| `max_det` | 20 | 최대 감지 수 |
| `amp` | true | 혼합 정밀도 사용 |

### 사전 설정

- **높은 정확도**: `conf=0.5`, `iou=0.8`, `imgsz=800`
- **빠른 처리**: `conf=0.25`, `iou=0.6`, `imgsz=480`
- **균형잡힌 설정**: `conf=0.35`, `iou=0.7`, `imgsz=640`

### API 엔드포인트

- `GET /api/yolo/config` - 현재 설정 조회
- `POST /api/yolo/config` - 설정 업데이트
- `POST /api/yolo/reset-config` - 설정 초기화
- `POST /api/yolo/reload` - 모델 재로드
- `GET /api/yolo/model-status` - 모델 상태 확인
- `POST /api/yolo/download-model` - 모델 다운로드

## 🔧 기술 스택

### 백엔드
- **FastAPI**: 고성능 웹 프레임워크
- **Ultralytics YOLO**: 객체 감지 모델
- **OpenCV**: 영상 처리
- **PyTorch**: 딥러닝 프레임워크

### 프론트엔드
- **React**: 사용자 인터페이스
- **Styled Components**: CSS-in-JS 스타일링
- **Fetch API**: HTTP 통신

## 📊 성능 최적화

### 배치 처리
- 여러 프레임을 배치로 처리하여 GPU 활용도 향상
- 배치 크기: 4 프레임

### 모델 최적화
- 모델 퓨전 (`model.fuse()`)으로 추론 속도 향상
- 혼합 정밀도 (AMP) 사용으로 메모리 효율성 증대
- CUDA 최적화 설정

### 메모리 관리
- 불필요한 중간 결과 제거
- 배치 크기 조정으로 메모리 사용량 제어

## 🎯 사용법

1. **영상 업로드**: "📹 영상 업로드" 탭에서 영상 파일 선택
2. **YOLO 설정**: "⚙️ YOLO 설정" 탭에서 추론 파라미터 조정
3. **결과 확인**: "🖼️ 갤러리" 탭에서 크롭된 고양이 이미지 확인

## 🔍 모니터링

### 성능 메트릭
- 처리 시간 (FPS)
- 메모리 사용량
- 감지 정확도
- False Positive/Negative 비율

### 로그 분석
- 추론 시간 측정
- 배치 처리 효율성
- GPU 활용도

## 📚 참고 자료

- [Ultralytics YOLO 공식 문서](https://docs.ultralytics.com/ko/usage/cfg/)
- [COCO 데이터셋 클래스 정보](https://cocodataset.org/#explore)
- [PyTorch 성능 최적화 가이드](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🐛 문제 보고

버그를 발견하거나 기능 요청이 있으시면 [Issues](https://github.com/your-username/cat_ui/issues) 페이지에서 알려주세요. 