import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 40px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  border: 2px dashed rgba(255, 255, 255, 0.3);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;

  &:hover {
    border-color: rgba(255, 255, 255, 0.6);
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-4px);
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.6s;
  }

  &:hover::before {
    left: 100%;
  }
`;

const Dropzone = styled.div`
  width: 100%;
  max-width: 600px;
  text-align: center;
  cursor: pointer;
  position: relative;
  z-index: 1;
`;

const UploadIcon = styled.div`
  font-size: 5rem;
  margin-bottom: 24px;
  color: rgba(255, 255, 255, 0.9);
  animation: pulse 2s infinite;
`;

const UploadText = styled.h2`
  color: white;
  margin-bottom: 16px;
  font-size: 2rem;
  font-weight: 600;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
`;

const UploadDescription = styled.p`
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 32px;
  font-size: 1.1rem;
  line-height: 1.6;
  max-width: 500px;
  margin-left: auto;
  margin-right: auto;
`;

const FileList = styled.div`
  margin-top: 32px;
  width: 100%;
  max-width: 600px;
  animation: fadeIn 0.5s ease-out;
`;

const FileItem = styled.div`
  padding: 16px 20px;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: all 0.3s ease;
  animation: slideIn 0.3s ease-out;

  &:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateX(4px);
  }
`;

const FileInfo = styled.div`
  flex: 1;
`;

const FileName = styled.p`
  font-weight: 600;
  color: white;
  margin: 0;
  font-size: 1rem;
`;

const FileSize = styled.p`
  color: rgba(255, 255, 255, 0.7);
  margin: 4px 0 0 0;
  font-size: 0.9rem;
`;

const RemoveButton = styled.button`
  background: rgba(244, 67, 54, 0.8);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  margin-left: 16px;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(244, 67, 54, 1);
    transform: scale(1.05);
  }
`;

const UploadButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 16px 32px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  margin-top: 24px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
  }

  &:disabled {
    background: rgba(108, 117, 125, 0.6);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const LoadingOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  animation: fadeIn 0.3s ease-out;
`;

const LoadingContent = styled.div`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  padding: 48px;
  border-radius: 24px;
  text-align: center;
  max-width: 450px;
  width: 90%;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
`;

const LoadingIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 24px;
  animation: spin 2s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const LoadingTitle = styled.h3`
  color: white;
  margin-bottom: 16px;
  font-size: 1.5rem;
  font-weight: 600;
`;

const LoadingText = styled.p`
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 16px;
  font-size: 1rem;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 12px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  overflow: hidden;
  margin: 20px 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
  border-radius: 6px;
`;

const ErrorMessage = styled.div`
  color: #ff6b6b;
  background: rgba(255, 107, 107, 0.1);
  padding: 16px;
  border-radius: 12px;
  margin-top: 16px;
  border: 1px solid rgba(255, 107, 107, 0.3);
  backdrop-filter: blur(10px);
`;

const SummaryText = styled.div`
  margin-top: 20px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.9);
  text-align: center;
`;

function VideoUploader({ onVideoUpload }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [uploadError, setUploadError] = useState('');

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      console.error('파일 업로드 실패:', rejectedFiles);
      return;
    }

    // 비디오 파일만 필터링
    const videoFiles = acceptedFiles.filter(file => file.type.startsWith('video/'));
    
    if (videoFiles.length > 0) {
      setSelectedFiles(prev => [...prev, ...videoFiles]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    multiple: true,
    maxSize: 100 * 1024 * 1024 // 100MB
  });

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadError('');
    setUploadStatus('업로드 준비 중...');

    try {
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('videos', file);
      });

      setUploadStatus('파일 업로드 중...');
      setUploadProgress(20);

      const response = await fetch('http://localhost:5000/api/video/upload', {
        method: 'POST',
        body: formData,
      });

      setUploadProgress(60);
      setUploadStatus('고양이 감지 중...');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setUploadProgress(80);
      setUploadStatus('처리 완료 중...');

      if (data.success) {
        setUploadProgress(100);
        setUploadStatus('완료!');
        
        // 모든 영상의 고양이 데이터를 하나로 합치기
        const allCats = [];
        data.results.forEach(result => {
          allCats.push(...result.processingResult.croppedCats);
        });
        
        onVideoUpload(allCats, data.summary);
        setSelectedFiles([]);
        
        // 잠시 후 로딩 화면 닫기
        setTimeout(() => {
          setIsUploading(false);
          setUploadProgress(0);
          setUploadStatus('');
        }, 1000);
      } else {
        throw new Error(data.message || '업로드 실패');
      }
    } catch (error) {
      console.error('업로드 오류:', error);
      setUploadError(`업로드 실패: ${error.message}`);
      setIsUploading(false);
      setUploadProgress(0);
      setUploadStatus('');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const totalSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);

  return (
    <>
      <UploadContainer {...getRootProps()}>
        <Dropzone>
          <input {...getInputProps()} />
          <UploadIcon>📹</UploadIcon>
          <UploadText>
            {isDragActive ? '파일을 여기에 놓으세요' : '영상 파일들을 업로드하세요'}
          </UploadText>
          <UploadDescription>
            MP4, AVI, MOV, MKV, WEBM 형식을 지원합니다 (최대 100MB)
            <br />
            여러 파일을 동시에 선택할 수 있습니다
          </UploadDescription>
          
          {selectedFiles.length > 0 && (
            <FileList>
              {selectedFiles.map((file, index) => (
                <FileItem key={index}>
                  <FileInfo>
                    <FileName>{file.name}</FileName>
                    <FileSize>크기: {formatFileSize(file.size)}</FileSize>
                  </FileInfo>
                  <RemoveButton onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}>
                    삭제
                  </RemoveButton>
                </FileItem>
              ))}
              
              <SummaryText>
                선택된 파일: {selectedFiles.length}개
                <br />
                총 크기: {formatFileSize(totalSize)}
              </SummaryText>
              
              <UploadButton 
                onClick={(e) => {
                  e.stopPropagation();
                  handleUpload();
                }}
                disabled={isUploading}
              >
                {isUploading ? '업로드 중...' : '업로드 시작'}
              </UploadButton>
            </FileList>
          )}

          {fileRejections.length > 0 && (
            <ErrorMessage>
              일부 파일 업로드에 실패했습니다. 지원되는 형식인지 확인해주세요.
            </ErrorMessage>
          )}
        </Dropzone>
      </UploadContainer>

      {isUploading && (
        <LoadingOverlay>
          <LoadingContent>
            <LoadingIcon>🔄</LoadingIcon>
            <LoadingTitle>영상 처리 중...</LoadingTitle>
            <LoadingText>{uploadStatus}</LoadingText>
            <ProgressBar>
              <ProgressFill progress={uploadProgress} />
            </ProgressBar>
            <LoadingText>{uploadProgress}% 완료</LoadingText>
            {uploadError && (
              <ErrorMessage>
                {uploadError}
              </ErrorMessage>
            )}
          </LoadingContent>
        </LoadingOverlay>
      )}
    </>
  );
}

export default VideoUploader; 