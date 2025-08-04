import React, { useCallback } from 'react';
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

const FileInfo = styled.div`
  margin-top: 20px;
  padding: 15px;
  background: #e3f2fd;
  border-radius: 8px;
  border: 1px solid #2196f3;
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

const ErrorMessage = styled.div`
  color: #d32f2f;
  background: #ffebee;
  padding: 10px;
  border-radius: 4px;
  margin-top: 10px;
  border: 1px solid #f44336;
`;

function VideoUploader({ onVideoUpload }) {
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      console.error('파일 업로드 실패:', rejectedFiles);
      return;
    }

    const file = acceptedFiles[0];
    if (file && file.type.startsWith('video/')) {
      onVideoUpload(file);
    }
  }, [onVideoUpload]);

  const { getRootProps, getInputProps, isDragActive, acceptedFiles, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    multiple: false,
    maxSize: 100 * 1024 * 1024 // 100MB
  });

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <UploadContainer {...getRootProps()}>
      <Dropzone>
        <input {...getInputProps()} />
        <UploadIcon>📹</UploadIcon>
        <UploadText>
          {isDragActive ? '파일을 여기에 놓으세요' : '영상 파일을 업로드하세요'}
        </UploadText>
        <UploadDescription>
          MP4, AVI, MOV, MKV, WEBM 형식을 지원합니다 (최대 100MB)
        </UploadDescription>
        
        {acceptedFiles.length > 0 && (
          <FileInfo>
            <FileName>{acceptedFiles[0].name}</FileName>
            <FileSize>크기: {formatFileSize(acceptedFiles[0].size)}</FileSize>
          </FileInfo>
        )}

        {fileRejections.length > 0 && (
          <ErrorMessage>
            파일 업로드에 실패했습니다. 지원되는 형식인지 확인해주세요.
          </ErrorMessage>
        )}
      </Dropzone>
    </UploadContainer>
  );
}

export default VideoUploader; 