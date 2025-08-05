import React, { useState } from 'react';
import styled from 'styled-components';
import YOLOConfig from './YOLOConfig';

const FloatingMenuContainer = styled.div`
  position: fixed;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 1000;
`;

const MenuButton = styled.button`
  width: 60px;
  height: 60px;
  border-radius: 50%;
  border: none;
  background: #007bff;
  color: white;
  font-size: 24px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
  transition: all 0.3s ease;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background: #0056b3;
    transform: scale(1.1);
    box-shadow: 0 6px 16px rgba(0, 123, 255, 0.4);
  }

  &.active {
    background: #28a745;
  }
`;

const MenuPanel = styled.div`
  position: fixed;
  right: 90px;
  top: 50%;
  transform: translateY(-50%);
  width: 400px;
  max-height: 80vh;
  background: white;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  overflow-y: auto;
  z-index: 999;
  animation: slideIn 0.3s ease;

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-50%) translateX(20px);
    }
    to {
      opacity: 1;
      transform: translateY(-50%) translateX(0);
    }
  }
`;

const PanelHeader = styled.div`
  padding: 20px;
  border-bottom: 1px solid #e9ecef;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const PanelTitle = styled.h3`
  margin: 0;
  color: #333;
  font-size: 18px;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #666;
  padding: 5px;

  &:hover {
    color: #333;
  }
`;

const PanelContent = styled.div`
  padding: 20px;
`;

const StatsPanel = styled.div`
  padding: 20px;
`;

const StatItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;

  &:last-child {
    border-bottom: none;
  }
`;

const StatLabel = styled.span`
  font-weight: 600;
  color: #495057;
`;

const StatValue = styled.span`
  color: #007bff;
  font-weight: 600;
`;

const InfoPanel = styled.div`
  padding: 20px;
`;

const InfoSection = styled.div`
  margin-bottom: 20px;
`;

const InfoTitle = styled.h4`
  color: #333;
  margin-bottom: 10px;
  font-size: 16px;
`;

const InfoText = styled.p`
  color: #666;
  line-height: 1.6;
  margin: 5px 0;
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