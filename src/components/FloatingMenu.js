import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const FloatingMenuContainer = styled.div`
  position: fixed;
  right: 30px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 1000;
`;

const MenuButton = styled.button`
  width: 70px;
  height: 70px;
  border-radius: 50%;
  border: none;
  background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
  color: white;
  font-size: 28px;
  cursor: pointer;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
  transition: all 0.2s ease;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);

  &:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
  }

  &.active {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.2);
  }
`;

const MenuPanel = styled.div`
  position: fixed;
  right: 110px;
  top: 50%;
  transform: translateY(-50%);
  width: 350px;
  max-height: 80vh;
  background: #ffffff;
  border-radius: 24px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
  z-index: 999;
  animation: slideIn 0.2s ease;
  border: 1px solid #e2e8f0;

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-50%) translateX(15px);
    }
    to {
      opacity: 1;
      transform: translateY(-50%) translateX(0);
    }
  }
`;

const PanelHeader = styled.div`
  padding: 24px;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const PanelTitle = styled.h3`
  margin: 0;
  color: #2d3748;
  font-size: 1.3rem;
  font-weight: 600;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const CloseButton = styled.button`
  background: #f7fafc;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #2d3748;
  padding: 8px;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;

  &:hover {
    background: #edf2f7;
    transform: scale(1.05);
  }
`;

const PanelContent = styled.div`
  padding: 24px;
`;

const DarkModeToggle = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px;
  background: #f7fafc;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  margin-bottom: 20px;
`;

const ToggleLabel = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const ToggleText = styled.span`
  font-weight: 600;
  color: #2d3748;
  font-size: 1.1rem;
`;

const ToggleSwitch = styled.label`
  position: relative;
  display: inline-block;
  width: 60px;
  height: 34px;
`;

const ToggleInput = styled.input`
  opacity: 0;
  width: 0;
  height: 0;

  &:checked + span {
    background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
  }

  &:checked + span:before {
    transform: translateX(26px);
  }
`;

const ToggleSlider = styled.span`
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: #cbd5e0;
  transition: 0.3s;
  border-radius: 34px;

  &:before {
    position: absolute;
    content: "";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background: white;
    transition: 0.3s;
    border-radius: 50%;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
`;

const StatsPanel = styled.div`
  padding: 24px;
`;

const StatItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  border-bottom: 1px solid #e2e8f0;

  &:last-child {
    border-bottom: none;
  }
`;

const StatLabel = styled.span`
  font-weight: 600;
  color: #4a5568;
  font-size: 1rem;
`;

const StatValue = styled.span`
  color: #3182ce;
  font-weight: 700;
  font-size: 1.1rem;
  text-shadow: 0 2px 10px rgba(49, 130, 206, 0.2);
`;

const LoadingContainer = styled.div`
  text-align: center;
  padding: 40px 20px;
  color: #718096;
`;

const LoadingIcon = styled.div`
  font-size: 2rem;
  margin-bottom: 16px;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  text-align: center;
  padding: 20px;
  color: #e53e3e;
  background: rgba(229, 62, 62, 0.1);
  border-radius: 12px;
  margin: 20px 0;
`;

const RefreshButton = styled.button`
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 8px 16px;
  font-size: 0.9rem;
  cursor: pointer;
  color: #4a5568;
  transition: all 0.2s ease;
  margin-left: 12px;

  &:hover {
    background: #edf2f7;
    transform: scale(1.02);
  }
`;

const InfoPanel = styled.div`
  padding: 24px;
`;

const InfoSection = styled.div`
  margin-bottom: 24px;
`;

const InfoTitle = styled.h4`
  color: #2d3748;
  margin-bottom: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const InfoText = styled.p`
  color: #4a5568;
  line-height: 1.6;
  margin: 8px 0;
  font-size: 0.95rem;
`;

const FloatingMenu = ({ darkMode, onDarkModeToggle }) => {
  const [activeMenu, setActiveMenu] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [videoList, setVideoList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchStatistics = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch('http://localhost:5000/api/statistics');
      const data = await response.json();
      
      if (data.success) {
        setStatistics(data.statistics);
        setVideoList(data.video_list || []);
      } else {
        setError(data.error || '통계 정보를 불러오는데 실패했습니다.');
      }
    } catch (error) {
      console.error('통계 정보 조회 실패:', error);
      setError('서버에 연결할 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleMenuClick = (menu) => {
    if (activeMenu === menu) {
      setActiveMenu(null);
    } else {
      setActiveMenu(menu);
      // 통계 메뉴가 열릴 때 데이터 새로고침
      if (menu === 'stats') {
        fetchStatistics();
      }
    }
  };

  const closeMenu = () => {
    setActiveMenu(null);
  };

  const renderMenuContent = () => {
    switch (activeMenu) {
      case 'settings':
        return (
          <>
            <PanelHeader>
              <PanelTitle>⚙️ 설정</PanelTitle>
              <CloseButton onClick={closeMenu}>×</CloseButton>
            </PanelHeader>
            <PanelContent>
              <DarkModeToggle>
                <ToggleLabel>
                  <span>🌙</span>
                  <ToggleText>다크 모드</ToggleText>
                </ToggleLabel>
                <ToggleSwitch>
                  <ToggleInput
                    type="checkbox"
                    checked={darkMode}
                    onChange={onDarkModeToggle}
                  />
                  <ToggleSlider />
                </ToggleSwitch>
              </DarkModeToggle>
            </PanelContent>
          </>
        );
      
      case 'stats':
        return (
          <>
            <PanelHeader>
              <PanelTitle>📊 통계</PanelTitle>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <RefreshButton onClick={fetchStatistics} disabled={loading}>
                  {loading ? '🔄' : '🔄'}
                </RefreshButton>
                <CloseButton onClick={closeMenu}>×</CloseButton>
              </div>
            </PanelHeader>
            <StatsPanel>
              {loading ? (
                <LoadingContainer>
                  <LoadingIcon>⏳</LoadingIcon>
                  <p>통계 정보를 불러오는 중...</p>
                </LoadingContainer>
              ) : error ? (
                <ErrorMessage>
                  <div>❌ {error}</div>
                </ErrorMessage>
              ) : (
                <>
                  <div style={{ marginBottom: '16px' }}>
                    <h4 style={{ color: '#2d3748', margin: '0 0 12px 0', fontSize: '1rem' }}>
                      📊 소스데이터
                    </h4>
                  </div>
                  <StatItem>
                    <StatLabel>영상</StatLabel>
                    <StatValue>{statistics?.video_count || 0}개</StatValue>
                  </StatItem>
                  <StatItem>
                    <StatLabel>이미지 수</StatLabel>
                    <StatValue>{statistics?.cropped_count || 0}개</StatValue>
                  </StatItem>
                  <StatItem>
                    <StatLabel>라벨링된 이미지</StatLabel>
                    <StatValue>{statistics?.labeled_count || 0}개</StatValue>
                  </StatItem>
                  
                  {statistics?.label_counts && Object.keys(statistics.label_counts).length > 0 && (
                    <>
                      <div style={{ marginTop: '24px', marginBottom: '16px' }}>
                        <h4 style={{ color: '#2d3748', margin: '0 0 12px 0', fontSize: '1rem' }}>
                          🐱 고양이별 이미지 수
                        </h4>
                      </div>
                      {Object.entries(statistics.label_counts)
                        .sort(([,a], [,b]) => b - a) // 개수 기준 내림차순 정렬
                        .map(([catName, count]) => (
                          <StatItem key={catName}>
                            <StatLabel style={{ fontSize: '0.9rem' }}>
                              {catName}
                            </StatLabel>
                            <StatValue style={{ fontSize: '0.9rem' }}>
                              {count}개
                            </StatValue>
                          </StatItem>
                        ))}
                    </>
                  )}
                  
                  {videoList.length > 0 && (
                    <>
                      <div style={{ marginTop: '24px', marginBottom: '16px' }}>
                        <h4 style={{ color: '#2d3748', margin: '0 0 12px 0', fontSize: '1rem' }}>
                          📹 최근 업로드된 영상
                        </h4>
                      </div>
                      {videoList.slice(0, 3).map((video, index) => (
                        <StatItem key={index}>
                          <StatLabel style={{ fontSize: '0.9rem', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {video.filename}
                          </StatLabel>
                          <StatValue style={{ fontSize: '0.9rem' }}>
                            {video.size_mb}MB
                          </StatValue>
                        </StatItem>
                      ))}
                    </>
                  )}
                </>
              )}
            </StatsPanel>
          </>
        );
      
      case 'info':
        return (
          <>
            <PanelHeader>
              <PanelTitle>ℹ️ 정보</PanelTitle>
              <CloseButton onClick={closeMenu}>×</CloseButton>
            </PanelHeader>
            <InfoPanel>
              <InfoSection>
                <InfoTitle>다둥이 매니저 v1.0</InfoTitle>
                <InfoText>AI 기반 고양이 감지 및 관리 시스템</InfoText>
              </InfoSection>
              
              <InfoSection>
                <InfoTitle>주요 기능</InfoTitle>
                <InfoText>• 영상에서 고양이 자동 감지</InfoText>
                <InfoText>• 개별 고양이 이미지 추출</InfoText>
                <InfoText>• 갤러리 형태로 관리</InfoText>
                <InfoText>• YOLO 모델 기반 분석</InfoText>
              </InfoSection>
              
              <InfoSection>
                <InfoTitle>기술 스택</InfoTitle>
                <InfoText>• Frontend: React.js</InfoText>
                <InfoText>• Backend: Python Flask</InfoText>
                <InfoText>• AI: YOLOv11</InfoText>
                <InfoText>• Styling: Styled Components</InfoText>
              </InfoSection>
              
              <InfoSection>
                <InfoTitle>개발자</InfoTitle>
                <InfoText>Cat UI Team</InfoText>
                <InfoText>© 2024 All rights reserved</InfoText>
              </InfoSection>
            </InfoPanel>
          </>
        );
      
      default:
        return null;
    }
  };

  return (
    <>
      <FloatingMenuContainer>
        <MenuButton
          onClick={() => handleMenuClick('settings')}
          className={activeMenu === 'settings' ? 'active' : ''}
          title="설정"
        >
          ⚙️
        </MenuButton>
        <MenuButton
          onClick={() => handleMenuClick('stats')}
          className={activeMenu === 'stats' ? 'active' : ''}
          title="통계"
        >
          📊
        </MenuButton>
        <MenuButton
          onClick={() => handleMenuClick('info')}
          className={activeMenu === 'info' ? 'active' : ''}
          title="정보"
        >
          ℹ️
        </MenuButton>
      </FloatingMenuContainer>

      {activeMenu && (
        <MenuPanel>
          {renderMenuContent()}
        </MenuPanel>
      )}
    </>
  );
};

export default FloatingMenu; 