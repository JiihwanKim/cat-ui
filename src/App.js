import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import VideoUploader from './components/VideoUploader';
import CatCropper from './components/CatCropper';
import CatGallery from './components/CatGallery';
import FloatingMenu from './components/FloatingMenu';

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
  const [activeTab, setActiveTab] = useState('upload'); // upload, gallery
  const [uploadSummary, setUploadSummary] = useState(null);
  const [isLoadingGallery, setIsLoadingGallery] = useState(false);
  const [savedGroups, setSavedGroups] = useState({});

  // 갤러리 탭이 활성화될 때 저장된 고양이 데이터 로드
  useEffect(() => {
    if (activeTab === 'gallery') {
      loadSavedCats();
    }
  }, [activeTab]);

  const loadSavedCats = async () => {
    try {
      setIsLoadingGallery(true);
      console.log('=== 갤러리 데이터 로드 시작 ===');
      
      const response = await fetch('http://localhost:5000/api/cropped-cats');
      const data = await response.json();
      
      console.log('백엔드 응답:', data);
      
      if (data.success) {
        console.log('저장된 고양이 데이터 로드:', data.croppedCats);
        console.log('저장된 그룹 정보 로드:', data.groups);
        
        setCroppedCats(data.croppedCats || []);
        setSavedGroups(data.groups || {});
        
        if (data.croppedCats && data.croppedCats.length > 0) {
          console.log(`총 ${data.croppedCats.length}마리의 고양이 데이터를 로드했습니다.`);
        }
      } else {
        console.error('백엔드 응답 실패:', data);
        setCroppedCats([]);
        setSavedGroups({});
      }
    } catch (error) {
      console.error('저장된 고양이 데이터 로드 실패:', error);
      setCroppedCats([]);
      setSavedGroups({});
    } finally {
      setIsLoadingGallery(false);
    }
  };

  const handleVideoUpload = (cats, summary) => {
    console.log('=== 업로드된 원본 데이터 ===');
    console.log('cats:', cats);
    console.log('summary:', summary);
    
    // cats 데이터에 필요한 정보가 있는지 확인하고 보완
    const processedCats = cats.map((cat, index) => {
      console.log(`고양이 ${index + 1} 원본 데이터:`, cat);
      
      // 기본값 설정
      const processedCat = {
        ...cat,
        frame: cat.frame || 0,
        timestamp: cat.timestamp || 0,
        timeString: cat.timeString || '00:00',
        confidence: cat.confidence || 0.8,
        videoName: cat.videoName || 'unknown',
        fps: cat.fps || 30,
        totalFrames: cat.totalFrames || 0
      };
      
      // timeString이 없으면 timestamp에서 계산
      if (!processedCat.timeString && processedCat.timestamp > 0) {
        const minutes = Math.floor(processedCat.timestamp / 60);
        const seconds = Math.floor(processedCat.timestamp % 60);
        processedCat.timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
      }
      
      console.log(`고양이 ${index + 1} 처리된 데이터:`, processedCat);
      return processedCat;
    });
    
    console.log('=== 최종 처리된 데이터 ===');
    console.log('processedCats:', processedCats);
    
    setCroppedCats(processedCats);
    setUploadSummary(summary);
    setCurrentStep('gallery');
    setActiveTab('gallery');
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
    setUploadSummary(null);
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
        <Title>🐱 다둥이 매니저</Title>
        <Subtitle>영상을 업로드하면 자동으로 고양이를 감지하고 관리합니다</Subtitle>
      </Header>

      <TabContainer>
        <Tab 
          active={activeTab === 'upload'} 
          onClick={() => handleTabChange('upload')}
        >
          📹 영상 업로드
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

        <TabContent active={activeTab === 'gallery'}>
          <CatGallery 
            croppedCats={croppedCats}
            onBack={handleBackToProcessing}
            onReset={handleBackToUpload}
            uploadSummary={uploadSummary}
            isLoading={isLoadingGallery}
            onRefresh={loadSavedCats}
            savedGroups={savedGroups}
          />
        </TabContent>
      </MainContent>

      <FloatingMenu />
    </AppContainer>
  );
}

export default App; 