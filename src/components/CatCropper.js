import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const CropperContainer = styled.div`
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 20px;
`;

const VideoInfo = styled.div`
  text-align: center;
  margin-bottom: 30px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #dee2e6;
`;

const VideoName = styled.h3`
  color: #333;
  margin-bottom: 10px;
`;

const VideoSize = styled.p`
  color: #666;
  margin: 5px 0;
`;

const ProcessingStatus = styled.div`
  text-align: center;
  margin: 30px 0;
  padding: 30px;
  background: #e3f2fd;
  border-radius: 12px;
  border: 2px solid #2196f3;
`;

const StatusIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  color: #2196f3;
`;

const StatusTitle = styled.h2`
  color: #1976d2;
  margin-bottom: 15px;
`;

const StatusDescription = styled.p`
  color: #666;
  font-size: 1.1rem;
  margin-bottom: 20px;
`;

const ProgressContainer = styled.div`
  margin: 20px 0;
`;

const ProgressText = styled.p`
  color: #666;
  margin-bottom: 10px;
  font-weight: 500;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 12px;
  background: #e9ecef;
  border-radius: 6px;
  overflow: hidden;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #2196f3, #1976d2);
  transition: width 0.5s ease;
  width: ${props => props.progress}%;
  border-radius: 6px;
`;

const Controls = styled.div`
  display: flex;
  justify-content: center;
  gap: 15px;
  margin: 20px 0;
  flex-wrap: wrap;
`;

const Button = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;

  &.primary {
    background: #007bff;
    color: white;
    
    &:hover {
      background: #0056b3;
    }
  }

  &.secondary {
    background: #6c757d;
    color: white;
    
    &:hover {
      background: #545b62;
    }
  }

  &.success {
    background: #28a745;
    color: white;
    
    &:hover {
      background: #1e7e34;
    }
  }

  &.danger {
    background: #dc3545;
    color: white;
    
    &:hover {
      background: #c82333;
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ResultContainer = styled.div`
  text-align: center;
  margin: 30px 0;
  padding: 30px;
  background: #d4edda;
  border-radius: 12px;
  border: 2px solid #c3e6cb;
`;

const ResultIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  color: #28a745;
`;

const ResultTitle = styled.h2`
  color: #155724;
  margin-bottom: 15px;
`;

const ResultStats = styled.div`
  display: flex;
  justify-content: center;
  gap: 30px;
  margin: 20px 0;
  flex-wrap: wrap;
`;

const StatItem = styled.div`
  text-align: center;
  padding: 15px;
  background: white;
  border-radius: 8px;
  min-width: 120px;
`;

const StatNumber = styled.div`
  font-size: 2rem;
  font-weight: bold;
  color: #28a745;
  margin-bottom: 5px;
`;

const StatLabel = styled.div`
  color: #666;
  font-size: 0.9rem;
`;

const ErrorContainer = styled.div`
  text-align: center;
  margin: 30px 0;
  padding: 30px;
  background: #f8d7da;
  border-radius: 12px;
  border: 2px solid #f5c6cb;
`;

const ErrorIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  color: #dc3545;
`;

const ErrorTitle = styled.h2`
  color: #721c24;
  margin-bottom: 15px;
`;

const ErrorMessage = styled.p`
  color: #721c24;
  font-size: 1.1rem;
`;

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

function CatCropper({ videoFile, onCatsCropped, onBack }) {
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const startProcessing = async () => {
    setProcessingStatus('processing');
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      // 1. 파일 업로드
      setCurrentStep('영상 파일을 업로드하는 중...');
      setProgress(10);
      
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch('http://localhost:5000/api/video/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '업로드 중 오류가 발생했습니다.');
      }

      const uploadResult = await response.json();
      setProgress(100);

      // 2. 결과 처리
      if (uploadResult.success) {
        console.log('업로드 결과:', uploadResult);
        console.log('크롭된 고양이:', uploadResult.processingResult.croppedCats);
        
        setResult({
          totalFrames: uploadResult.processingResult.totalFrames,
          detectedCats: uploadResult.processingResult.detectedCats,
          croppedCats: uploadResult.processingResult.croppedCats,
          message: uploadResult.processingResult.message
        });
        setProcessingStatus('completed');
      } else {
        throw new Error(uploadResult.error || '처리 중 오류가 발생했습니다.');
      }

    } catch (error) {
      console.error('Processing error:', error);
      setError(error.message);
      setProcessingStatus('error');
    }
  };

  const handleComplete = () => {
    if (result && result.croppedCats) {
      onCatsCropped(result.croppedCats);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    if (videoFile) {
      startProcessing();
    }
  }, [videoFile]);

  // 처리 완료 시 자동으로 갤러리로 이동
  useEffect(() => {
    if (processingStatus === 'completed' && result && result.croppedCats) {
      // 약간의 지연 후 갤러리로 이동
      setTimeout(() => {
        onCatsCropped(result.croppedCats);
      }, 1000);
    }
  }, [processingStatus, result, onCatsCropped]);

  if (processingStatus === 'idle') {
    return (
      <CropperContainer>
        <VideoInfo>
          <VideoName>{videoFile.name}</VideoName>
          <VideoSize>크기: {formatFileSize(videoFile.size)}</VideoSize>
          <VideoSize>형식: {videoFile.type}</VideoSize>
        </VideoInfo>
        
        <Controls>
          <Button className="primary" onClick={startProcessing}>
            🚀 처리 시작
          </Button>
          <Button className="secondary" onClick={onBack}>
            ← 뒤로 가기
          </Button>
        </Controls>
      </CropperContainer>
    );
  }

  if (processingStatus === 'processing') {
    return (
      <CropperContainer>
        <VideoInfo>
          <VideoName>{videoFile.name}</VideoName>
          <VideoSize>크기: {formatFileSize(videoFile.size)}</VideoSize>
        </VideoInfo>

        <ProcessingStatus>
          <StatusIcon>🔍</StatusIcon>
          <StatusTitle>고양이 감지 중...</StatusTitle>
          <StatusDescription>
            YOLO 모델을 사용하여 영상에서 고양이를 감지하고 크롭 이미지를 생성하고 있습니다.
          </StatusDescription>
          
          <ProgressContainer>
            <ProgressText>{currentStep}</ProgressText>
            <ProgressBar>
              <ProgressFill progress={progress} />
            </ProgressBar>
            <ProgressText>{Math.round(progress)}% 완료</ProgressText>
          </ProgressContainer>
        </ProcessingStatus>
      </CropperContainer>
    );
  }

  if (processingStatus === 'error') {
    return (
      <CropperContainer>
        <VideoInfo>
          <VideoName>{videoFile.name}</VideoName>
          <VideoSize>크기: {formatFileSize(videoFile.size)}</VideoSize>
        </VideoInfo>

        <ErrorContainer>
          <ErrorIcon>❌</ErrorIcon>
          <ErrorTitle>처리 중 오류 발생</ErrorTitle>
          <ErrorMessage>{error}</ErrorMessage>
        </ErrorContainer>

        <Controls>
          <Button className="primary" onClick={startProcessing}>
            🔄 다시 시도
          </Button>
          <Button className="secondary" onClick={onBack}>
            ← 뒤로 가기
          </Button>
        </Controls>
      </CropperContainer>
    );
  }

  if (processingStatus === 'completed' && result) {
    return (
      <CropperContainer>
        <VideoInfo>
          <VideoName>{videoFile.name}</VideoName>
          <VideoSize>크기: {formatFileSize(videoFile.size)}</VideoSize>
        </VideoInfo>

        <ResultContainer>
          <ResultIcon>✅</ResultIcon>
          <ResultTitle>처리 완료!</ResultTitle>
          <StatusDescription>{result.message}</StatusDescription>
          
          <ResultStats>
            <StatItem>
              <StatNumber>{result.totalFrames}</StatNumber>
              <StatLabel>총 프레임</StatLabel>
            </StatItem>
            <StatItem>
              <StatNumber>{result.detectedCats.length}</StatNumber>
              <StatLabel>감지된 고양이</StatLabel>
            </StatItem>
            <StatItem>
              <StatNumber>{result.croppedCats.length}</StatNumber>
              <StatLabel>크롭된 이미지</StatLabel>
            </StatItem>
          </ResultStats>
        </ResultContainer>

        <Controls>
          <Button className="success" onClick={handleComplete}>
            🖼️ 갤러리 보기
          </Button>
          <Button className="secondary" onClick={onBack}>
            ← 뒤로 가기
          </Button>
        </Controls>
      </CropperContainer>
    );
  }

  return null;
}

export default CatCropper; 