import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const CropperContainer = styled.div`
  background: ${props => props.darkMode ? '#2d3748' : 'white'};
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 20px;
  transition: all 0.3s ease;
  animation: fadeIn 0.4s ease-out;
`;

const VideoInfo = styled.div`
  text-align: center;
  margin-bottom: 30px;
  padding: 20px;
  background: ${props => props.darkMode ? '#4a5568' : '#f8f9fa'};
  border-radius: 8px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#dee2e6'};
  transition: all 0.3s ease;
`;

const VideoName = styled.h3`
  color: ${props => props.darkMode ? '#e2e8f0' : '#333'};
  margin-bottom: 10px;
  transition: color 0.3s ease;
`;

const VideoSize = styled.p`
  color: ${props => props.darkMode ? '#a0aec0' : '#666'};
  margin: 5px 0;
  transition: color 0.3s ease;
`;

const ProcessingStatus = styled.div`
  text-align: center;
  margin: 30px 0;
  padding: 30px;
  background: ${props => props.darkMode ? '#2c5282' : '#e3f2fd'};
  border-radius: 12px;
  border: 2px solid #2196f3;
  transition: all 0.3s ease;
  animation: fadeIn 0.4s ease-out;
`;

const StatusIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  color: #2196f3;
  animation: pulse 1.5s infinite;
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
  }
`;

const StatusTitle = styled.h2`
  color: ${props => props.darkMode ? '#e2e8f0' : '#1976d2'};
  margin-bottom: 15px;
  transition: color 0.3s ease;
`;

const StatusDescription = styled.p`
  color: ${props => props.darkMode ? '#a0aec0' : '#666'};
  font-size: 1.1rem;
  margin-bottom: 20px;
  transition: color 0.3s ease;
`;

const ProgressContainer = styled.div`
  margin: 20px 0;
`;

const ProgressText = styled.p`
  color: ${props => props.darkMode ? '#a0aec0' : '#666'};
  margin-bottom: 10px;
  font-weight: 500;
  transition: color 0.3s ease;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 12px;
  background: ${props => props.darkMode ? '#4a5568' : '#e9ecef'};
  border-radius: 6px;
  overflow: hidden;
  transition: background 0.3s ease;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #2196f3, #1976d2);
  transition: width 0.3s ease;
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
  transition: all 0.2s ease;
  font-weight: 500;

  &.primary {
    background: #007bff;
    color: white;
    
    &:hover {
      background: #0056b3;
      transform: translateY(-1px);
    }
  }

  &.secondary {
    background: ${props => props.darkMode ? '#4a5568' : '#6c757d'};
    color: ${props => props.darkMode ? '#e2e8f0' : 'white'};
    transition: all 0.2s ease;
    
    &:hover {
      background: ${props => props.darkMode ? '#718096' : '#545b62'};
      transform: translateY(-1px);
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none !important;
  }
`;

const ErrorMessage = styled.div`
  background: #f8d7da;
  color: #721c24;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #f5c6cb;
  margin: 20px 0;
  text-align: center;
`;

const SuccessMessage = styled.div`
  background: #d4edda;
  color: #155724;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #c3e6cb;
  margin: 20px 0;
  text-align: center;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 20px 0;
`;

const StatCard = styled.div`
  background: ${props => props.darkMode ? '#4a5568' : '#f8f9fa'};
  padding: 20px;
  border-radius: 8px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#dee2e6'};
  text-align: center;
  transition: all 0.3s ease;
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.darkMode ? '#e2e8f0' : '#333'};
  margin-bottom: 5px;
  transition: color 0.3s ease;
`;

const StatLabel = styled.div`
  color: ${props => props.darkMode ? '#a0aec0' : '#666'};
  font-size: 0.9rem;
  transition: color 0.3s ease;
`;

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

function CatCropper({ videoFile, onCatsCropped, onBack, darkMode = false }) {
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const startProcessing = async () => {
    setProcessing(true);
    setProgress(0);
    setError('');
    setStatus('영상을 분석하고 있습니다...');

    try {
      // 진행률 시뮬레이션
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);

      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch('http://localhost:5000/api/process-video', {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setStatus('처리가 완료되었습니다!');
          setResult(data);
          setTimeout(() => {
            onCatsCropped(data.cats);
          }, 1000);
        } else {
          setError(data.error || '처리 중 오류가 발생했습니다.');
        }
      } else {
        setError('서버 오류가 발생했습니다.');
      }
    } catch (error) {
      console.error('처리 오류:', error);
      setError('네트워크 오류가 발생했습니다.');
    } finally {
      setProcessing(false);
    }
  };

  const handleComplete = () => {
    if (result) {
      onCatsCropped(result.cats);
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <CropperContainer darkMode={darkMode}>
      <VideoInfo darkMode={darkMode}>
        <VideoName darkMode={darkMode}>{videoFile.name}</VideoName>
        <VideoSize darkMode={darkMode}>크기: {formatFileSize(videoFile.size)}</VideoSize>
      </VideoInfo>

      {!processing && !result && (
        <ProcessingStatus darkMode={darkMode}>
          <StatusIcon>🎬</StatusIcon>
          <StatusTitle darkMode={darkMode}>영상 처리 준비</StatusTitle>
          <StatusDescription darkMode={darkMode}>
            업로드된 영상에서 고양이를 감지하고 개별 이미지로 추출합니다.
          </StatusDescription>
          <Controls>
            <Button 
              className="primary" 
              onClick={startProcessing}
              darkMode={darkMode}
            >
              처리 시작
            </Button>
            <Button 
              className="secondary" 
              onClick={onBack}
              darkMode={darkMode}
            >
              뒤로 가기
            </Button>
          </Controls>
        </ProcessingStatus>
      )}

      {processing && (
        <ProcessingStatus darkMode={darkMode}>
          <StatusIcon>⚙️</StatusIcon>
          <StatusTitle darkMode={darkMode}>처리 중...</StatusTitle>
          <StatusDescription darkMode={darkMode}>{status}</StatusDescription>
          
          <ProgressContainer>
            <ProgressText darkMode={darkMode}>진행률: {progress}%</ProgressText>
            <ProgressBar darkMode={darkMode}>
              <ProgressFill progress={progress} />
            </ProgressBar>
          </ProgressContainer>
        </ProcessingStatus>
      )}

      {result && (
        <ProcessingStatus darkMode={darkMode}>
          <StatusIcon>✅</StatusIcon>
          <StatusTitle darkMode={darkMode}>처리 완료!</StatusTitle>
          <StatusDescription darkMode={darkMode}>
            {result.cats.length}마리의 고양이를 감지했습니다.
          </StatusDescription>
          
          <StatsGrid>
            <StatCard darkMode={darkMode}>
              <StatValue darkMode={darkMode}>{result.cats.length}</StatValue>
              <StatLabel darkMode={darkMode}>감지된 고양이</StatLabel>
            </StatCard>
            <StatCard darkMode={darkMode}>
              <StatValue darkMode={darkMode}>{result.summary?.total_frames || 0}</StatValue>
              <StatLabel darkMode={darkMode}>총 프레임</StatLabel>
            </StatCard>
            <StatCard darkMode={darkMode}>
              <StatValue darkMode={darkMode}>{formatTime(result.summary?.duration || 0)}</StatValue>
              <StatLabel darkMode={darkMode}>영상 길이</StatLabel>
            </StatCard>
          </StatsGrid>

          <Controls>
            <Button 
              className="primary" 
              onClick={handleComplete}
              darkMode={darkMode}
            >
              갤러리로 이동
            </Button>
            <Button 
              className="secondary" 
              onClick={onBack}
              darkMode={darkMode}
            >
              다시 처리
            </Button>
          </Controls>
        </ProcessingStatus>
      )}

      {error && (
        <ErrorMessage>
          <strong>오류:</strong> {error}
        </ErrorMessage>
      )}
    </CropperContainer>
  );
}

export default CatCropper; 