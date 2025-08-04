const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const sharp = require('sharp');

const app = express();
const PORT = process.env.PORT || 5000;

// 미들웨어 설정
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 정적 파일 제공 (크롭된 이미지들)
app.use('/cropped-images', express.static(path.join(__dirname, 'cropped-images')));

// 업로드된 파일 저장을 위한 multer 설정
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueName = `${Date.now()}-${uuidv4()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB 제한
  },
  fileFilter: (req, file, cb) => {
    // 비디오 파일 형식 체크
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('지원되지 않는 파일 형식입니다.'), false);
    }
  }
});

// 이미지 크롭 클래스
class ImageCropper {
  constructor() {
    this.croppedImagesDir = path.join(__dirname, 'cropped-images');
    if (!fs.existsSync(this.croppedImagesDir)) {
      fs.mkdirSync(this.croppedImagesDir, { recursive: true });
    }
  }

  // 실제 프레임에서 고양이 크롭 이미지 생성
  async createCatCrop(cat, frameBuffer) {
    try {
      // Sharp를 사용하여 이미지 처리
      const frame = sharp(frameBuffer);
      
      // 바운딩 박스 좌표 (정수로 변환)
      const x = Math.max(0, Math.floor(cat.x));
      const y = Math.max(0, Math.floor(cat.y));
      const width = Math.min(Math.floor(cat.width), 640 - x); // 기본 비디오 크기 가정
      const height = Math.min(Math.floor(cat.height), 480 - y);
      
      // 크롭 영역 추출
      const croppedBuffer = await frame
        .extract({
          left: x,
          top: y,
          width: width,
          height: height
        })
        .resize(200, 200, { fit: 'cover' })
        .png()
        .toBuffer();
      
      // 파일명 생성
      const filename = `cat_${cat.id}_frame_${cat.frame}.png`;
      const filepath = path.join(this.croppedImagesDir, filename);
      
      // 파일 저장
      fs.writeFileSync(filepath, croppedBuffer);
      
      return {
        id: cat.id,
        filename: filename,
        url: `/cropped-images/${filename}`,
        frame: cat.frame,
        timestamp: cat.timestamp,
        confidence: cat.confidence,
        bbox: {
          x: x,
          y: y,
          width: width,
          height: height
        }
      };
    } catch (error) {
      console.error('고양이 크롭 생성 중 오류:', error);
      return null;
    }
  }

  // 여러 고양이 크롭 이미지 생성
  async createCatCrops(cats) {
    const croppedCats = [];
    
    for (const cat of cats) {
      if (cat.frameBuffer) {
        const croppedCat = await this.createCatCrop(cat, cat.frameBuffer);
        if (croppedCat) {
          croppedCats.push(croppedCat);
        }
      }
    }
    
    return croppedCats;
  }

  // 시뮬레이션용 프레임 이미지 생성
  async generateFrameImage(frameNumber, videoWidth = 640, videoHeight = 480) {
    try {
      // 간단한 그라데이션 이미지 생성
      const svg = `
        <svg width="${videoWidth}" height="${videoHeight}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#87CEEB;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#98FB98;stop-opacity:1" />
            </linearGradient>
          </defs>
          <rect width="100%" height="100%" fill="url(#grad)"/>
          <text x="20" y="40" font-family="Arial" font-size="24" fill="black">Frame: ${frameNumber}</text>
          <circle cx="${Math.random() * videoWidth}" cy="${Math.random() * videoHeight}" r="${Math.random() * 20 + 10}" fill="hsl(${Math.random() * 360}, 70%, 60%)"/>
          <circle cx="${Math.random() * videoWidth}" cy="${Math.random() * videoHeight}" r="${Math.random() * 20 + 10}" fill="hsl(${Math.random() * 360}, 70%, 60%)"/>
          <circle cx="${Math.random() * videoWidth}" cy="${Math.random() * videoHeight}" r="${Math.random() * 20 + 10}" fill="hsl(${Math.random() * 360}, 70%, 60%)"/>
        </svg>
      `;

      return await sharp(Buffer.from(svg))
        .png()
        .toBuffer();
    } catch (error) {
      console.error('프레임 이미지 생성 중 오류:', error);
      // 기본 색상 이미지 생성
      return await sharp({
        create: {
          width: videoWidth,
          height: videoHeight,
          channels: 3,
          background: { r: 135, g: 206, b: 235 }
        }
      })
      .png()
      .toBuffer();
    }
  }
}

// YOLO11 시뮬레이터 (실제 모델이 없을 때 사용)
class YOLO11Simulator {
  constructor() {
    this.isModelLoaded = false;
  }

  async loadModel() {
    console.log('YOLO11 모델 로딩 중...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    this.isModelLoaded = true;
    console.log('YOLO11 모델 로딩 완료');
  }

  async extractFrames(videoPath) {
    console.log('프레임 추출 중...');
    // 실제로는 ffmpeg 등을 사용하여 프레임 추출
    await new Promise(resolve => setTimeout(resolve, 3000));
    return { totalFrames: Math.floor(Math.random() * 300) + 50, fps: 30 }; // 50-350 프레임
  }

  async detectCats(videoPath, totalFrames, fps) {
    console.log('고양이 감지 중...');
    const cats = [];
    const frameInterval = 10; // 10프레임 간격
    const processFrames = Math.floor(totalFrames / frameInterval);

    for (let i = 0; i < processFrames; i++) {
      const frameNumber = i * frameInterval;
      const timestamp = frameNumber / fps;

      // 랜덤하게 고양이 감지 (실제로는 YOLO 모델이 감지)
      const numCatsInFrame = Math.floor(Math.random() * 3); // 0-2마리
      
      for (let j = 0; j < numCatsInFrame; j++) {
        const confidence = 0.7 + Math.random() * 0.3; // 70-100%
        
        if (confidence > 0.8) { // 80% 이상일 때만 감지된 것으로 간주
          // 프레임 이미지 생성
          const frameBuffer = await imageCropper.generateFrameImage(frameNumber);
          
          cats.push({
            id: `cat-${Date.now()}-${i}-${j}`,
            frame: frameNumber,
            timestamp: timestamp,
            confidence: confidence,
            width: 80 + Math.random() * 120,
            height: 80 + Math.random() * 120,
            x: Math.random() * 400,
            y: Math.random() * 300,
            videoPath: videoPath,
            frameBuffer: frameBuffer
          });
        }
      }

      // 진행률 업데이트를 위한 지연
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    return cats;
  }

  async processVideo(videoPath) {
    try {
      // 1. 모델 로딩
      if (!this.isModelLoaded) {
        await this.loadModel();
      }

      // 2. 프레임 정보 추출
      const { totalFrames, fps } = await this.extractFrames(videoPath);

      // 3. 고양이 감지
      const detectedCats = await this.detectCats(videoPath, totalFrames, fps);

      // 4. 크롭된 이미지 생성
      console.log('크롭된 이미지 생성 중...');
      const croppedCats = await imageCropper.createCatCrops(detectedCats);

      return {
        success: true,
        totalFrames: totalFrames,
        detectedCats: detectedCats,
        croppedCats: croppedCats,
        message: `${detectedCats.length}마리의 고양이가 감지되었습니다.`
      };
    } catch (error) {
      console.error('비디오 처리 중 오류:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

// 이미지 크롭퍼 인스턴스 생성
const imageCropper = new ImageCropper();

// YOLO11 시뮬레이터 인스턴스 생성
const yoloProcessor = new YOLO11Simulator();

// API 라우트

// 1. 영상 업로드 및 처리 API
app.post('/api/video/upload', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: '영상 파일이 업로드되지 않았습니다.'
      });
    }

    const videoPath = req.file.path;
    const videoInfo = {
      originalName: req.file.originalname,
      filename: req.file.filename,
      size: req.file.size,
      mimetype: req.file.mimetype,
      path: videoPath
    };

    console.log('영상 업로드 완료:', videoInfo);

    // YOLO11 모델로 고양이 감지 및 크롭 이미지 생성
    const result = await yoloProcessor.processVideo(videoPath);

    if (result.success) {
      res.json({
        success: true,
        videoInfo: videoInfo,
        processingResult: result
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error
      });
    }

  } catch (error) {
    console.error('업로드 처리 중 오류:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 2. 고양이 데이터 전송 API
app.post('/api/cats/upload', async (req, res) => {
  try {
    const { cats } = req.body;
    
    if (!cats || !Array.isArray(cats)) {
      return res.status(400).json({
        success: false,
        error: '고양이 데이터가 올바르지 않습니다.'
      });
    }

    console.log(`${cats.length}마리의 고양이 데이터 수신`);

    // 크롭된 이미지 생성
    const croppedCats = await imageCropper.createCatCrops(cats);

    res.json({
      success: true,
      croppedCats: croppedCats,
      message: `${croppedCats.length}개의 크롭된 이미지가 생성되었습니다.`
    });

  } catch (error) {
    console.error('고양이 데이터 처리 중 오류:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 3. 크롭된 고양이 이미지 목록 API
app.get('/api/cropped-cats', (req, res) => {
  try {
    const croppedImagesDir = path.join(__dirname, 'cropped-images');
    
    if (!fs.existsSync(croppedImagesDir)) {
      return res.json({
        success: true,
        croppedCats: []
      });
    }

    const files = fs.readdirSync(croppedImagesDir);
    const imageFiles = files.filter(file => 
      file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg')
    );

    const croppedCats = imageFiles.map(filename => ({
      id: filename.replace(/\.[^/.]+$/, ''),
      filename: filename,
      url: `/cropped-images/${filename}`,
      timestamp: Date.now()
    }));

    res.json({
      success: true,
      croppedCats: croppedCats
    });

  } catch (error) {
    console.error('크롭된 이미지 목록 조회 중 오류:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 4. 서버 상태 확인 API
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    status: 'healthy',
    timestamp: new Date().toISOString(),
    yoloModelLoaded: yoloProcessor.isModelLoaded
  });
});

// 5. 업로드된 파일 목록 API
app.get('/api/videos', (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, 'uploads');
    
    if (!fs.existsSync(uploadsDir)) {
      return res.json({
        success: true,
        videos: []
      });
    }

    const files = fs.readdirSync(uploadsDir);
    const videoFiles = files.filter(file => 
      file.endsWith('.mp4') || file.endsWith('.avi') || file.endsWith('.mov') || 
      file.endsWith('.mkv') || file.endsWith('.webm')
    );

    const videos = videoFiles.map(filename => {
      const filePath = path.join(uploadsDir, filename);
      const stats = fs.statSync(filePath);
      return {
        filename: filename,
        size: stats.size,
        uploadTime: stats.mtime,
        path: filePath
      };
    });

    res.json({
      success: true,
      videos: videos
    });

  } catch (error) {
    console.error('비디오 파일 목록 조회 중 오류:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 6. 파일 삭제 API
app.delete('/api/videos/:filename', (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(__dirname, 'uploads', filename);
    
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      res.json({
        success: true,
        message: '파일이 삭제되었습니다.'
      });
    } else {
      res.status(404).json({
        success: false,
        error: '파일을 찾을 수 없습니다.'
      });
    }

  } catch (error) {
    console.error('파일 삭제 중 오류:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 에러 핸들링 미들웨어
app.use((error, req, res, next) => {
  console.error('서버 오류:', error);
  res.status(500).json({
    success: false,
    error: error.message || '서버 내부 오류가 발생했습니다.'
  });
});

// 404 핸들러
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: '요청한 API 엔드포인트를 찾을 수 없습니다.'
  });
});

// 서버 시작
app.listen(PORT, () => {
  console.log(`🐱 고양이 영상 처리 백엔드 서버가 포트 ${PORT}에서 실행 중입니다.`);
  console.log('📡 API 엔드포인트:');
  console.log('   - POST /api/video/upload (영상 업로드 및 처리)');
  console.log('   - POST /api/cats/upload (고양이 데이터 전송)');
  console.log('   - GET  /api/cropped-cats (크롭된 고양이 이미지 목록)');
  console.log('   - GET  /api/health (서버 상태 확인)');
  console.log('   - GET  /api/videos (업로드된 파일 목록)');
  console.log('   - DELETE /api/videos/:filename (파일 삭제)');
  console.log('   - GET  /cropped-images/* (크롭된 이미지 제공)');
});

module.exports = app; 