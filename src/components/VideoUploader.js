import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 40px;
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  border-radius: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 2px dashed ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;

  &:hover {
    border-color: #3182ce;
    background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(49, 130, 206, 0.05), transparent);
    transition: left 0.3s ease;
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
  color: #3182ce;
  animation: pulse 3s ease-in-out infinite;
  
  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }
`;

const UploadText = styled.h2`
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  margin-bottom: 16px;
  font-size: 2rem;
  font-weight: 600;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: color 0.3s ease;
`;

const UploadDescription = styled.p`
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  margin-bottom: 32px;
  font-size: 1.1rem;
  line-height: 1.6;
  max-width: 500px;
  margin-left: auto;
  margin-right: auto;
  transition: color 0.3s ease;
`;

const FileList = styled.div`
  margin-top: 32px;
  width: 100%;
  max-width: 600px;
  animation: fadeIn 0.5s ease-out;
`;

const FileItem = styled.div`
  padding: 16px 20px;
  background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  border-radius: 16px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: all 0.2s ease;
  animation: slideIn 0.2s ease;

  &:hover {
    background: ${props => props.darkMode ? '#718096' : '#edf2f7'};
    transform: translateX(2px);
  }
`;

const FileInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
`;

const FileIcon = styled.div`
  font-size: 2rem;
  color: #3182ce;
`;

const FileDetails = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const FileName = styled.span`
  font-weight: 600;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 1rem;
  transition: color 0.3s ease;
`;

const FileSize = styled.span`
  color: ${props => props.darkMode ? '#a0aec0' : '#718096'};
  font-size: 0.9rem;
  transition: color 0.3s ease;
`;

const RemoveButton = styled.button`
  background: #fed7d7;
  border: none;
  color: #c53030;
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s ease;

  &:hover {
    background: #feb2b2;
    transform: scale(1.02);
  }
`;

const UploadButton = styled.button`
  background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
  color: white;
  border: none;
  padding: 16px 32px;
  border-radius: 16px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-top: 24px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);

  &:hover:not(:disabled) {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
  }

  &:disabled {
    background: #cbd5e0;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const ProgressContainer = styled.div`
  width: 100%;
  max-width: 600px;
  margin-top: 24px;
  animation: fadeIn 0.5s ease-out;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #667eea, #4facfe);
  border-radius: 4px;
  transition: width 0.3s ease;
  width: ${props => props.progress}%;
`;

const ProgressText = styled.div`
  text-align: center;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 0.9rem;
  font-weight: 500;
  transition: color 0.3s ease;
`;

const Message = styled.div`
  padding: 16px 24px;
  border-radius: 12px;
  margin-top: 20px;
  font-weight: 600;
  text-align: center;
  animation: slideIn 0.3s ease-out;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;

  &.success {
    background: rgba(86, 171, 47, 0.2);
    color: #56ab2f;
    border: 1px solid rgba(86, 171, 47, 0.3);
  }

  &.error {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid rgba(255, 107, 107, 0.3);
  }

  &.info {
    background: rgba(79, 172, 254, 0.2);
    color: #4facfe;
    border: 1px solid rgba(79, 172, 254, 0.3);
  }
`;

const TipBox = styled.div`
  margin: 24px 0;
  padding: 16px 20px;
  background: ${props => props.darkMode ? 'rgba(214, 158, 46, 0.1)' : 'rgba(214, 158, 46, 0.05)'};
  border: 1px solid ${props => props.darkMode ? 'rgba(214, 158, 46, 0.3)' : 'rgba(214, 158, 46, 0.2)'};
  border-radius: 12px;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 0.95rem;
  line-height: 1.5;
  transition: all 0.3s ease;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: #ffffff;
  animation: spin 1s ease-in-out infinite;
  margin-right: 8px;

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const VideoUploader = ({ onVideoUpload, darkMode = false }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');

  const onDrop = useCallback((acceptedFiles) => {
    const videoFiles = acceptedFiles.filter(file => 
      file.type.startsWith('video/') || 
      file.name.toLowerCase().endsWith('.mp4') ||
      file.name.toLowerCase().endsWith('.avi') ||
      file.name.toLowerCase().endsWith('.mov') ||
      file.name.toLowerCase().endsWith('.mkv')
    );

    if (videoFiles.length === 0) {
      setMessage({ text: '비디오 파일만 업로드 가능합니다.', type: 'error' });
      return;
    }

    setFiles(prev => [...prev, ...videoFiles]);
    setMessage({ text: `${videoFiles.length}개의 비디오 파일이 추가되었습니다.`, type: 'success' });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    },
    multiple: true
  });

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setMessage({ text: '업로드할 파일을 선택해주세요.', type: 'error' });
      return;
    }

    setUploading(true);
    setProgress(0);
    setMessage('');

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('videos', file);
      });

      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          setProgress(percentComplete);
        }
      });

      xhr.addEventListener('load', async () => {
        if (xhr.status === 200) {
          try {
            const response = JSON.parse(xhr.responseText);
            if (response.success) {
              setMessage({ text: '업로드가 완료되었습니다! 고양이를 감지하는 중...', type: 'success' });
              
              // 고양이 감지 결과를 기다림
              const detectionResponse = await fetch('http://localhost:5000/api/detect-cats', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  video_files: response.video_files
                })
              });

              const detectionData = await detectionResponse.json();
              
              if (detectionData.success) {
                setMessage({ text: '고양이 감지가 완료되었습니다!', type: 'success' });
                onVideoUpload(detectionData.cats, detectionData.summary);
              } else {
                setMessage({ text: detectionData.error || '고양이 감지 중 오류가 발생했습니다.', type: 'error' });
              }
            } else {
              setMessage({ text: response.error || '업로드 중 오류가 발생했습니다.', type: 'error' });
            }
          } catch (error) {
            setMessage({ text: '서버 응답을 처리하는 중 오류가 발생했습니다.', type: 'error' });
          }
        } else {
          setMessage({ text: '업로드 중 오류가 발생했습니다.', type: 'error' });
        }
        setUploading(false);
        setProgress(0);
      });

      xhr.addEventListener('error', () => {
        setMessage({ text: '네트워크 오류가 발생했습니다.', type: 'error' });
        setUploading(false);
        setProgress(0);
      });

      xhr.open('POST', 'http://localhost:5000/api/upload-videos');
      xhr.send(formData);

    } catch (error) {
      console.error('업로드 오류:', error);
      setMessage({ text: '업로드 중 오류가 발생했습니다.', type: 'error' });
      setUploading(false);
      setProgress(0);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <UploadContainer {...getRootProps()} darkMode={darkMode}>
      <input {...getInputProps()} />
      <Dropzone>
        <UploadIcon>📹</UploadIcon>
        <UploadText darkMode={darkMode}>
          {isDragActive ? '파일을 여기에 놓으세요!' : '비디오 파일을 업로드하세요'}
        </UploadText>
        <UploadDescription darkMode={darkMode}>
          MP4, AVI, MOV, MKV 형식의 비디오 파일을 드래그 앤 드롭하거나 클릭하여 선택하세요.
          <br />
          AI가 자동으로 고양이를 감지하고 개별 이미지로 추출합니다.
        </UploadDescription>
      </Dropzone>

      <TipBox darkMode={darkMode}>
        💡 <strong>영상 선택 팁:</strong> 한 고양이가 등장하는 영상들을 업로드하면 나중에 고양이를 알려줄 때 선택하기 쉬워집니다.
        <br />
        여러 고양이가 동시에 나오는 영상보다는 한 마리씩 나오는 영상을 업로드하면 
        각 고양이의 특징을 더 명확하게 구분할 수 있어서 AI 학습에도 도움이 됩니다.
      </TipBox>

      {files.length > 0 && (
        <FileList>
          {files.map((file, index) => (
            <FileItem key={index} darkMode={darkMode}>
              <FileInfo>
                <FileIcon>🎬</FileIcon>
                <FileDetails>
                  <FileName darkMode={darkMode}>{file.name}</FileName>
                  <FileSize darkMode={darkMode}>{formatFileSize(file.size)}</FileSize>
                </FileDetails>
              </FileInfo>
              <RemoveButton onClick={() => removeFile(index)}>
                삭제
              </RemoveButton>
            </FileItem>
          ))}
        </FileList>
      )}

      {files.length > 0 && (
        <UploadButton 
          onClick={handleUpload} 
          disabled={uploading}
        >
          {uploading ? (
            <>
              <LoadingSpinner />
              업로드 중...
            </>
          ) : (
            '업로드 시작'
          )}
        </UploadButton>
      )}

      {uploading && (
        <ProgressContainer>
          <ProgressBar darkMode={darkMode}>
            <ProgressFill progress={progress} />
          </ProgressBar>
          <ProgressText darkMode={darkMode}>
            업로드 진행률: {Math.round(progress)}%
          </ProgressText>
        </ProgressContainer>
      )}

      {message && (
        <Message className={message.type}>
          {message.text}
        </Message>
      )}
    </UploadContainer>
  );
};

export default VideoUploader; 