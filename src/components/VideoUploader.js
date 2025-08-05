import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 2px dashed #ddd;
  transition: all 0.3s ease;

  &:hover {
    border-color: #007bff;
    background-color: #f8f9fa;
  }
`;

const Dropzone = styled.div`
  width: 100%;
  max-width: 500px;
  text-align: center;
  cursor: pointer;
`;

const UploadIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  color: #007bff;
`;

const UploadText = styled.h2`
  color: #333;
  margin-bottom: 10px;
  font-size: 1.5rem;
`;

const UploadDescription = styled.p`
  color: #666;
  margin-bottom: 20px;
  font-size: 1rem;
`;

const FileList = styled.div`
  margin-top: 20px;
  width: 100%;
  max-width: 500px;
`;

const FileItem = styled.div`
  padding: 10px 15px;
  background: #e3f2fd;
  border-radius: 8px;
  border: 1px solid #2196f3;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const FileInfo = styled.div`
  flex: 1;
`;

const FileName = styled.p`
  font-weight: bold;
  color: #1976d2;
  margin: 0;
`;

const FileSize = styled.p`
  color: #666;
  margin: 5px 0 0 0;
  font-size: 0.9rem;
`;

const RemoveButton = styled.button`
  background: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 5px 10px;
  cursor: pointer;
  font-size: 0.8rem;
  margin-left: 10px;

  &:hover {
    background: #d32f2f;
  }
`;

const UploadButton = styled.button`
  background: #28a745;
  color: white;
  border: none;
  border-radius: 6px;
  padding: 12px 24px;
  font-size: 1rem;
  cursor: pointer;
  margin-top: 20px;
  font-weight: 600;

  &:hover {
    background: #1e7e34;
  }

  &:disabled {
    background: #6c757d;
    cursor: not-allowed;
  }
`;

const LoadingOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

const LoadingContent = styled.div`
  background: white;
  padding: 40px;
  border-radius: 12px;
  text-align: center;
  max-width: 400px;
  width: 90%;
`;

const LoadingIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 20px;
  animation: spin 2s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const LoadingTitle = styled.h3`
  color: #333;
  margin-bottom: 15px;
`;

const LoadingText = styled.p`
  color: #666;
  margin-bottom: 10px;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  margin: 15px 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: #007bff;
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
`;

const ErrorMessage = styled.div`
  color: #d32f2f;
  background: #ffebee;
  padding: 10px;
  border-radius: 4px;
  margin-top: 10px;
  border: 1px solid #f44336;
`;

const SummaryText = styled.div`
  margin-top: 15px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #dee2e6;
  font-size: 0.9rem;
  color: #495057;
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