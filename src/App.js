import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import VideoUploader from './components/VideoUploader';
import CatCropper from './components/CatCropper';
import CatGallery from './components/CatGallery';
import FloatingMenu from './components/FloatingMenu';

const AppContainer = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 20px;
  min-height: 100vh;
  animation: fadeIn 0.3s ease;
  background: ${props => props.$darkMode ? '#1a202c' : '#ffffff'};
  color: ${props => props.$darkMode ? '#e2e8f0' : '#2d3748'};
  transition: all 0.2s ease;
`;

const GlobalMessage = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  padding: 16px 24px;
  border-radius: 12px;
  font-weight: 600;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  animation: slideInRight 0.2s ease;
  max-width: 400px;
  
  @keyframes slideInRight {
    from {
      opacity: 0;
      transform: translateX(50px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
  
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

const Header = styled.header`
  text-align: center;
  margin-bottom: 50px;
  animation: fadeIn 0.4s ease;
`;

const Title = styled.h1`
  color: ${props => props.$darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 3.5rem;
  font-weight: 700;
  margin-bottom: 16px;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  letter-spacing: -0.02em;
  transition: color 0.3s ease;
  
  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const Subtitle = styled.p`
  color: ${props => props.$darkMode ? '#a0aec0' : '#2d3748'};
  font-size: 1.25rem;
  font-weight: 500;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
  transition: color 0.3s ease;
  
  @media (max-width: 768px) {
    font-size: 1.1rem;
  }
`;

const TabContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 40px;
  background: ${props => props.$darkMode ? '#2d3748' : '#f7fafc'};
  border-radius: 20px;
  padding: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid ${props => props.$darkMode ? '#4a5568' : '#e2e8f0'};
  transition: all 0.3s ease;
`;

const Tab = styled.button`
  padding: 16px 32px;
  margin: 0 4px;
  border: none;
  background: ${props => props.$active ? '#3182ce' : 'transparent'};
  color: ${props => props.$active ? 'white' : props.$darkMode ? '#a0aec0' : '#4a5568'};
  border-radius: 16px;
  cursor: pointer;
  font-weight: 600;
  font-size: 1rem;
  transition: all 0.15s ease;
  position: relative;
  overflow: hidden;
  
  &:hover {
    background: ${props => props.$active ? '#2c5aa0' : props.$darkMode ? '#4a5568' : '#edf2f7'};
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  }
  
  &:active {
    transform: translateY(0);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
    transition: left 0.2s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
`;

const MainContent = styled.main`
  display: flex;
  flex-direction: column;
  gap: 40px;
  animation: fadeIn 0.3s ease;
`;

const TabContent = styled.div`
  display: ${props => props.active ? 'block' : 'none'};
  animation: ${props => props.active ? 'slideIn 0.2s ease' : 'none'};
  
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(5px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [croppedCats, setCroppedCats] = useState([]);
  const [currentStep, setCurrentStep] = useState('upload'); // upload, processing, gallery
  const [activeTab, setActiveTab] = useState('upload'); // upload, gallery
  const [uploadSummary, setUploadSummary] = useState(null);
  const [isLoadingGallery, setIsLoadingGallery] = useState(false);
  const [savedGroups, setSavedGroups] = useState({});
  const [globalMessage, setGlobalMessage] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);

  // 갤러리 탭이 활성화될 때 저장된 고양이 데이터 로드
  useEffect(() => {
    if (activeTab === 'gallery') {
      loadSavedCats();
    }
  }, [activeTab]);

  const showGlobalMessage = (message, type = 'info') => {
    setGlobalMessage({ text: message, type });
    setTimeout(() => setGlobalMessage(null), 3000);
  };

  const handleDarkModeToggle = () => {
    setDarkMode(!darkMode);
  };

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
    setUploadComplete(true); // 업로드 완료 상태 설정
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
    setUploadComplete(false); // 업로드 상태 초기화
  };

  const handleBackToProcessing = () => {
    setCurrentStep('upload');
    setActiveTab('upload');
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    // 업로드 탭을 선택할 때 currentStep을 upload로 설정
    if (tab === 'upload') {
      setCurrentStep('upload');
    }
  };

  return (
    <AppContainer $darkMode={darkMode}>
      {globalMessage && typeof globalMessage === 'object' && globalMessage.text && (
        <GlobalMessage className={globalMessage.type}>
          {globalMessage.text}
        </GlobalMessage>
      )}

      <Header>
        <Title $darkMode={darkMode}>🐱 다둥이 매니저</Title>
        <Subtitle $darkMode={darkMode}>AI가 영상에서 고양이를 자동으로 감지하고 관리해드립니다</Subtitle>
      </Header>

      <TabContainer $darkMode={darkMode}>
        <Tab 
          $active={activeTab === 'upload'} 
          onClick={() => handleTabChange('upload')}
          $darkMode={darkMode}
        >
          📹 영상 업로드
        </Tab>
        <Tab 
          $active={activeTab === 'gallery'} 
          onClick={() => handleTabChange('gallery')}
          $darkMode={darkMode}
        >
          🐱 우리집 고양이 알려주기
        </Tab>
      </TabContainer>

      <MainContent>
        <TabContent active={activeTab === 'upload'}>
          {currentStep === 'upload' && (
            <VideoUploader 
              onVideoUpload={handleVideoUpload} 
              $darkMode={darkMode}
              uploadComplete={uploadComplete}
              onResetUpload={() => setUploadComplete(false)}
              onShowGlobalMessage={showGlobalMessage}
            />
          )}

          {currentStep === 'processing' && videoFile && (
            <CatCropper 
              videoFile={videoFile} 
              onCatsCropped={handleCatsCropped}
              onBack={handleBackToUpload}
              $darkMode={darkMode}
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
            onShowGlobalMessage={showGlobalMessage}
            $darkMode={darkMode}
            activeTab={activeTab}
          />
        </TabContent>
      </MainContent>

      <FloatingMenu $darkMode={darkMode} onDarkModeToggle={handleDarkModeToggle} />
    </AppContainer>
  );
}

export default App; 