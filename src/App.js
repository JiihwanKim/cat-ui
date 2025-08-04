import React, { useState } from 'react';
import styled from 'styled-components';
import VideoUploader from './components/VideoUploader';
import CatCropper from './components/CatCropper';
import CatGallery from './components/CatGallery';
import YOLOConfig from './components/YOLOConfig';

const AppContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 30px;
`;

const Title = styled.h1`
  color: #333;
  font-size: 2.5rem;
  margin-bottom: 10px;
`;

const Subtitle = styled.p`
  color: #666;
  font-size: 1.1rem;
`;

const TabContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 30px;
  border-bottom: 2px solid #e9ecef;
`;

const Tab = styled.button`
  padding: 12px 24px;
  margin: 0 5px;
  border: none;
  background: ${props => props.active ? '#007bff' : 'transparent'};
  color: ${props => props.active ? 'white' : '#666'};
  border-radius: 8px 8px 0 0;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.active ? '#0056b3' : '#f8f9fa'};
  }
`;

const MainContent = styled.main`
  display: flex;
  flex-direction: column;
  gap: 30px;
`;

const TabContent = styled.div`
  display: ${props => props.active ? 'block' : 'none'};
`;

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [croppedCats, setCroppedCats] = useState([]);
  const [currentStep, setCurrentStep] = useState('upload'); // upload, processing, gallery
  const [activeTab, setActiveTab] = useState('upload'); // upload, config, gallery

  const handleVideoUpload = (file) => {
    setVideoFile(file);
    setCurrentStep('processing');
    setActiveTab('upload');
  };

  const handleCatsCropped = (cats) => {
    setCroppedCats(cats);
    setCurrentStep('gallery');
    setActiveTab('gallery');
  };

  const handleBackToUpload = () => {
    setVideoFile(null);
    setCroppedCats([]);
    setCurrentStep('upload');
    setActiveTab('upload');
  };

  const handleBackToProcessing = () => {
    setCurrentStep('processing');
    setActiveTab('upload');
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  return (
    <AppContainer>
      <Header>
        <Title>🐱 고양이 영상 처리</Title>
        <Subtitle>영상을 업로드하면 YOLO 모델이 자동으로 고양이를 감지합니다</Subtitle>
      </Header>

      <TabContainer>
        <Tab 
          active={activeTab === 'upload'} 
          onClick={() => handleTabChange('upload')}
        >
          📹 영상 업로드
        </Tab>
        <Tab 
          active={activeTab === 'config'} 
          onClick={() => handleTabChange('config')}
        >
          ⚙️ YOLO 설정
        </Tab>
        <Tab 
          active={activeTab === 'gallery'} 
          onClick={() => handleTabChange('gallery')}
        >
          🖼️ 갤러리
        </Tab>
      </TabContainer>

      <MainContent>
        <TabContent active={activeTab === 'upload'}>
          {currentStep === 'upload' && (
            <VideoUploader onVideoUpload={handleVideoUpload} />
          )}

          {currentStep === 'processing' && videoFile && (
            <CatCropper 
              videoFile={videoFile} 
              onCatsCropped={handleCatsCropped}
              onBack={handleBackToUpload}
            />
          )}
        </TabContent>

        <TabContent active={activeTab === 'config'}>
          <YOLOConfig />
        </TabContent>

        <TabContent active={activeTab === 'gallery'}>
          <CatGallery 
            croppedCats={croppedCats}
            onBack={handleBackToProcessing}
            onReset={handleBackToUpload}
          />
        </TabContent>
      </MainContent>
    </AppContainer>
  );
}

export default App; 