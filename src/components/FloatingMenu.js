import React, { useState } from 'react';
import styled from 'styled-components';
import YOLOConfig from './YOLOConfig';

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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 28px;
  cursor: pointer;
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);

  &:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    transform: scale(1.1) translateY(-2px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
  }

  &.active {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    box-shadow: 0 8px 25px rgba(86, 171, 47, 0.3);
  }
`;

const MenuPanel = styled.div`
  position: fixed;
  right: 110px;
  top: 50%;
  transform: translateY(-50%);
  width: 450px;
  max-height: 80vh;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
  overflow-y: auto;
  z-index: 999;
  animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid rgba(255, 255, 255, 0.2);

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-50%) translateX(30px);
    }
    to {
      opacity: 1;
      transform: translateY(-50%) translateX(0);
    }
  }
`;

const PanelHeader = styled.div`
  padding: 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const PanelTitle = styled.h3`
  margin: 0;
  color: white;
  font-size: 1.3rem;
  font-weight: 600;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
`;

const CloseButton = styled.button`
  background: rgba(255, 255, 255, 0.2);
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: white;
  padding: 8px;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);

  &:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: scale(1.1);
  }
`;

const PanelContent = styled.div`
  padding: 24px;
`;

const StatsPanel = styled.div`
  padding: 24px;
`;

const StatItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);

  &:last-child {
    border-bottom: none;
  }
`;

const StatLabel = styled.span`
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
  font-size: 1rem;
`;

const StatValue = styled.span`
  color: rgba(102, 126, 234, 1);
  font-weight: 700;
  font-size: 1.1rem;
  text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
`;

const InfoPanel = styled.div`
  padding: 24px;
`;

const InfoSection = styled.div`
  margin-bottom: 24px;
`;

const InfoTitle = styled.h4`
  color: white;
  margin-bottom: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
`;

const InfoText = styled.p`
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  margin: 8px 0;
  font-size: 0.95rem;
`;

const FloatingMenu = () => {
  const [activeMenu, setActiveMenu] = useState(null);

  const handleMenuClick = (menu) => {
    if (activeMenu === menu) {
      setActiveMenu(null);
    } else {
      setActiveMenu(menu);
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
              <YOLOConfig />
            </PanelContent>
          </>
        );
      
      case 'stats':
        return (
          <>
            <PanelHeader>
              <PanelTitle>📊 통계</PanelTitle>
              <CloseButton onClick={closeMenu}>×</CloseButton>
            </PanelHeader>
            <StatsPanel>
              <StatItem>
                <StatLabel>처리된 영상</StatLabel>
                <StatValue>12개</StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>감지된 고양이</StatLabel>
                <StatValue>47마리</StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>총 처리 시간</StatLabel>
                <StatValue>2시간 34분</StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>평균 정확도</StatLabel>
                <StatValue>94.2%</StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>저장된 이미지</StatLabel>
                <StatValue>156개</StatValue>
              </StatItem>
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