import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const GalleryContainer = styled.div`
  background: #ffffff;
  border-radius: 24px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  padding: 32px;
  border: 1px solid #e2e8f0;
  animation: fadeIn 0.6s ease-out;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 2px solid #e2e8f0;
`;

const Title = styled.h2`
  color: #2d3748;
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const Stats = styled.div`
  text-align: right;
  color: #4a5568;
  font-weight: 500;
`;

const Controls = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  flex-wrap: wrap;
`;

const Button = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 16px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-weight: 600;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
  }

  &:hover::before {
    left: 100%;
  }

  &.primary {
    background: #3182ce;
    color: white;
    box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2);
    
    &:hover {
      background: #2c5aa0;
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(49, 130, 206, 0.3);
    }
  }

  &.secondary {
    background: #f7fafc;
    color: #2d3748;
    border: 1px solid #e2e8f0;
    
    &:hover {
      background: #edf2f7;
      transform: translateY(-1px);
    }
  }

  &.success {
    background: #38a169;
    color: white;
    box-shadow: 0 4px 12px rgba(56, 161, 105, 0.2);
    
    &:hover {
      background: #2f855a;
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(56, 161, 105, 0.3);
    }
  }

  &.danger {
    background: #e53e3e;
    color: white;
    box-shadow: 0 4px 12px rgba(229, 62, 62, 0.2);
    
    &:hover {
      background: #c53030;
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(229, 62, 62, 0.3);
    }
  }

  &.info {
    background: #4299e1;
    color: white;
    box-shadow: 0 4px 12px rgba(66, 153, 225, 0.2);
    
    &:hover {
      background: #3182ce;
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(66, 153, 225, 0.3);
    }
  }

  &.warning {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    
    &:hover {
      background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const GroupSelector = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  flex-wrap: wrap;
  align-items: center;
`;

const GroupButton = styled.button`
  padding: 12px 20px;
  border: 2px solid ${props => props.selected ? '#3182ce' : props.borderColor || '#e2e8f0'};
  background: ${props => props.selected ? '#3182ce' : props.backgroundColor || '#ffffff'};
  color: ${props => props.selected ? 'white' : '#2d3748'};
  border-radius: 20px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 600;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  gap: 8px;
  
  &:hover {
    border-color: #3182ce;
    background: ${props => props.selected ? '#2c5aa0' : '#f7fafc'};
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  }
`;

const GroupCount = styled.span`
  background: ${props => props.selected ? 'white' : '#3182ce'};
  color: ${props => props.selected ? '#3182ce' : 'white'};
  border-radius: 50%;
  padding: 4px 8px;
  font-size: 0.8rem;
  font-weight: 700;
  margin-left: 8px;
`;

const GroupActions = styled.div`
  display: flex;
  gap: 4px;
  margin-left: 8px;
`;

const ActionButton = styled.button`
  padding: 6px 10px;
  border: none;
  border-radius: 12px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${props => {
    if (props.action === 'edit') return 'rgba(255, 193, 7, 0.8)';
    if (props.action === 'delete') return 'rgba(255, 107, 107, 0.8)';
    return 'rgba(108, 117, 125, 0.8)';
  }};
  color: ${props => props.action === 'edit' ? '#2d3748' : 'white'};
  
  &:hover {
    background: ${props => {
      if (props.action === 'edit') return 'rgba(255, 193, 7, 1)';
      if (props.action === 'delete') return 'rgba(255, 107, 107, 1)';
      return 'rgba(108, 117, 125, 1)';
    }};
    transform: scale(1.05);
  }
`;

const Modal = styled.div`
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

const ModalContent = styled.div`
  background: #ffffff;
  padding: 40px;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  min-width: 450px;
  max-width: 550px;
  border: 1px solid #e2e8f0;
`;

const ModalTitle = styled.h3`
  margin: 0 0 24px 0;
  color: #2d3748;
  font-size: 1.5rem;
  font-weight: 600;
`;

const ModalInput = styled.input`
  width: 100%;
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  font-size: 1rem;
  margin-bottom: 24px;
  background: #ffffff;
  color: #2d3748;
  
  &::placeholder {
    color: #a0aec0;
  }
  
  &:focus {
    outline: none;
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
  }
`;

const ModalButtons = styled.div`
  display: flex;
  gap: 12px;
  justify-content: flex-end;
`;

const BulkActionSection = styled.div`
  margin-bottom: 24px;
  padding: 20px;
  background: #f7fafc;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  display: ${props => props.show ? 'block' : 'none'};
  animation: fadeIn 0.5s ease-out;
`;

const BulkActionTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  color: #2d3748;
  font-weight: 600;
`;

const BulkActionDescription = styled.p`
  margin: 0 0 16px 0;
  font-size: 1rem;
  color: #4a5568;
  line-height: 1.5;
`;

const BulkActionControls = styled.div`
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
`;

const BulkNameInput = styled.input`
  padding: 12px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  font-size: 1rem;
  min-width: 250px;
  background: #ffffff;
  color: #2d3748;
  
  &::placeholder {
    color: #a0aec0;
  }
  
  &:focus {
    outline: none;
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
  }
`;

const GalleryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 24px;
  margin-bottom: 24px;
`;

const CatCard = styled.div`
  border: 2px solid ${props => {
    if (props.selected) return '#3182ce';
    if (props.highlighted) return '#38a169';
    return '#e2e8f0';
  }};
  border-radius: 16px;
  overflow: hidden;
  background: #ffffff;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    border-color: #3182ce;
  }

  &.selected {
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.2);
  }

  &.highlighted {
    border-color: #38a169;
    box-shadow: 0 0 0 3px rgba(56, 161, 105, 0.2);
  }
`;

const CatImage = styled.div`
  position: relative;
  width: 100%;
  height: 220px;
  background: #f7fafc;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .no-image {
    color: #a0aec0;
    font-size: 1rem;
    text-align: center;
  }
`;

const CatInfo = styled.div`
  padding: 20px;
`;

const CatTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  color: #2d3748;
  font-weight: 600;
  padding: 8px 12px;
  background: ${props => props.groupColor || 'rgba(102, 126, 234, 0.1)'};
  border-radius: 8px;
  border-left: 4px solid ${props => props.groupColor || 'rgba(102, 126, 234, 0.8)'};
  display: inline-block;
`;

const CatDetails = styled.div`
  font-size: 0.9rem;
  color: #4a5568;
  line-height: 1.5;
`;

const ConfidenceBadge = styled.span`
  display: inline-block;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-top: 8px;

  &.high {
    background: rgba(86, 171, 47, 0.2);
    color: #56ab2f;
    border: 1px solid rgba(86, 171, 47, 0.3);
  }

  &.medium {
    background: rgba(255, 193, 7, 0.2);
    color: #ffc107;
    border: 1px solid rgba(255, 193, 7, 0.3);
  }

  &.low {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid rgba(255, 107, 107, 0.3);
  }
`;

const CatName = styled.div`
  font-weight: 600;
  color: rgba(102, 126, 234, 1);
  margin-top: 12px;
  font-size: 1rem;
  padding: 8px 12px;
  background: rgba(102, 126, 234, 0.1);
  border-radius: 8px;
  border-left: 4px solid rgba(102, 126, 234, 0.8);
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 60px 20px;
  color: #718096;
`;

const EmptyIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  opacity: 0.6;
`;

const EmptyText = styled.p`
  margin: 0;
  font-size: 1.2rem;
  font-weight: 500;
`;

const TeachingOverlay = styled.div`
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
  z-index: 2000;
  animation: fadeIn 0.5s ease-out;
`;

const TeachingContent = styled.div`
  background: #ffffff;
  padding: 60px 40px;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  text-align: center;
  border: 1px solid #e2e8f0;
  max-width: 500px;
  width: 90%;
`;

const TeachingIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  animation: pulse 2s infinite;
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }
`;

const TeachingTitle = styled.h2`
  color: #2d3748;
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 16px;
`;

const TeachingDescription = styled.p`
  color: #4a5568;
  font-size: 1.1rem;
  line-height: 1.6;
  margin-bottom: 24px;
`;

const TeachingProgress = styled.div`
  width: 100%;
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 20px;
`;

const TeachingProgressBar = styled.div`
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
  border-radius: 4px;
  animation: progress 3s ease-in-out infinite;
  
  @keyframes progress {
    0% { width: 0%; }
    50% { width: 70%; }
    100% { width: 100%; }
  }
`;

const TeachingStatus = styled.div`
  color: #4a5568;
  font-size: 1rem;
  font-weight: 500;
`;

const GallerySection = styled.div`
  margin-top: 24px;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  overflow: hidden;
  background: #ffffff;
`;

const GalleryHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: #f7fafc;
  border-bottom: 1px solid #e2e8f0;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: #edf2f7;
  }
`;

const GalleryTitle = styled.h3`
  margin: 0;
  font-size: 1.1rem;
  color: #2d3748;
  font-weight: 600;
`;

const GalleryToggle = styled.button`
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #4a5568;
  transition: transform 0.3s ease;
  
  &:hover {
    color: #2d3748;
  }
`;

const GalleryContent = styled.div`
  padding: 24px;
  display: ${props => props.collapsed ? 'none' : 'block'};
  animation: ${props => props.collapsed ? 'none' : 'slideDown 0.3s ease-out'};
  
  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

const StatusMessage = styled.div`
  text-align: center;
  padding: 16px;
  margin: 16px 0;
  border-radius: 12px;
  font-weight: 600;
  backdrop-filter: blur(10px);

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

const FilterSection = styled.div`
  margin-bottom: 24px;
  padding: 20px;
  background: #f7fafc;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
`;

const FilterTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  color: #2d3748;
  font-weight: 600;
`;

function CatGallery({ croppedCats, onBack, onReset, uploadSummary, isLoading: isGalleryLoading, onRefresh, savedGroups, onShowGlobalMessage }) {
  const [selectedCats, setSelectedCats] = useState(new Set());
  const [catNames, setCatNames] = useState({});
  const [statusMessage, setStatusMessage] = useState('');
  const [selectedGroup, setSelectedGroup] = useState('all');
  const [highlightedGroup, setHighlightedGroup] = useState('');
  const [bulkNameInput, setBulkNameInput] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [modalAction, setModalAction] = useState('');
  const [modalGroupName, setModalGroupName] = useState('');
  const [modalInputValue, setModalInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTeaching, setIsTeaching] = useState(false);
  const [teachingStep, setTeachingStep] = useState('');
  const [isGalleryCollapsed, setIsGalleryCollapsed] = useState(false);
  
  // Shift 키 연결 선택을 위한 상태 추가
  const [lastSelectedCat, setLastSelectedCat] = useState(null);
  const [isShiftPressed, setIsShiftPressed] = useState(false);

  // 동적 색상 생성 함수들
  const generateColorFromString = (str) => {
    // 문자열을 해시로 변환하여 일관된 색상 생성
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // 32비트 정수로 변환
    }
    
    // 해시를 기반으로 HSL 색상 생성
    const hue = Math.abs(hash) % 360;
    const saturation = 60 + (Math.abs(hash) % 20); // 60-80%
    const lightness = 45 + (Math.abs(hash) % 15); // 45-60%
    
    return { hue, saturation, lightness };
  };

  const generateBackgroundColor = (str) => {
    const { hue, saturation, lightness } = generateColorFromString(str);
    return `hsla(${hue}, ${saturation}%, ${lightness}%, 0.1)`;
  };

  const generateBorderColor = (str) => {
    const { hue, saturation, lightness } = generateColorFromString(str);
    return `hsla(${hue}, ${saturation}%, ${lightness}%, 0.8)`;
  };



  // 저장된 그룹 정보가 변경될 때마다 catNames 업데이트
  useEffect(() => {
    if (savedGroups && Object.keys(savedGroups).length > 0) {
      console.log('저장된 그룹 정보 적용:', savedGroups);
      setCatNames(savedGroups);
    }
  }, [savedGroups]);

  // 컴포넌트 마운트 시 백엔드 연결 테스트
  useEffect(() => {
    const testBackendConnection = async () => {
      try {
        console.log('=== 백엔드 연결 테스트 ===');
        const response = await fetch('http://localhost:5000/api/health');
        const data = await response.json();
        console.log('백엔드 연결 성공:', data);
      } catch (error) {
        console.error('백엔드 연결 실패:', error);
        setStatusMessage({ 
          type: 'error', 
          text: '백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.' 
        });
      }
    };
    
    testBackendConnection();
  }, []);

  // 디버깅을 위한 로그
  useEffect(() => {
    console.log('CatGallery - croppedCats:', croppedCats);
    console.log('CatGallery - catNames:', catNames);
    if (croppedCats && croppedCats.length > 0) {
      console.log('크롭된 고양이 수:', croppedCats.length);
      croppedCats.forEach((cat, index) => {
        console.log(`고양이 ${index + 1}:`, cat);
      });
    }
  }, [croppedCats, catNames]);

  // 저장된 그룹 정보 로드
  const loadCatGroups = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:5000/api/cat-groups');
      const data = await response.json();
      
      if (data.success && data.groups) {
        setCatNames(data.groups);
        const message = `저장된 그룹 정보를 로드했습니다. (${Object.keys(data.groups).length}개의 고양이 이름)`;
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'success');
        } else {
          setStatusMessage({ type: 'success', text: message });
          setTimeout(() => setStatusMessage(''), 3000);
        }
      }
    } catch (error) {
      console.error('그룹 정보 로드 실패:', error);
      const message = '저장된 그룹 정보 로드에 실패했습니다.';
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'error');
      } else {
        setStatusMessage({ type: 'error', text: message });
        setTimeout(() => setStatusMessage(''), 3000);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // 그룹 정보 저장
  const saveCatGroups = async () => {
    try {
      setIsLoading(true);
      
      // catNames가 올바른 형식인지 확인
      console.log('=== 그룹 정보 저장 시작 ===');
      console.log('저장할 그룹 정보:', catNames);
      console.log('그룹 정보 타입:', typeof catNames);
      console.log('그룹 정보 키:', Object.keys(catNames));
      
      // 빈 객체가 아닌지 확인
      if (Object.keys(catNames).length === 0) {
        const message = '저장할 그룹 정보가 없습니다.';
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'info');
        } else {
          setStatusMessage({ type: 'info', text: message });
          setTimeout(() => setStatusMessage(''), 3000);
        }
        return;
      }
      
      const requestBody = JSON.stringify(catNames);
      console.log('요청 URL:', 'http://localhost:5000/api/cat-groups');
      console.log('요청 메서드:', 'POST');
      console.log('요청 본문:', requestBody);
      
      const response = await fetch('http://localhost:5000/api/cat-groups', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: requestBody,
      });
      
      console.log('=== 응답 정보 ===');
      console.log('응답 상태:', response.status);
      console.log('응답 상태 텍스트:', response.statusText);
      console.log('응답 URL:', response.url);
      console.log('응답 헤더:', response.headers);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('저장 응답 데이터:', data);
      
      if (data.success) {
        console.log('=== 저장 성공 ===');
        const message = `그룹 정보가 성공적으로 저장되었습니다! (${Object.keys(catNames).length}개의 고양이 이름)`;
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'success');
        } else {
          setStatusMessage({ type: 'success', text: message });
          setTimeout(() => setStatusMessage(''), 3000);
        }
      } else {
        throw new Error(data.message || data.error || '저장 실패');
      }
    } catch (error) {
      console.error('=== 그룹 정보 저장 실패 ===');
      console.error('에러:', error);
      const message = `그룹 정보 저장에 실패했습니다: ${error.message}`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'error');
      } else {
        setStatusMessage({ type: 'error', text: message });
        setTimeout(() => setStatusMessage(''), 5000);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // 고양이 그룹 생성
  const getCatGroups = () => {
    const groups = {
      'all': { name: '전체', count: 0 },
      'unnamed': { name: '미지정', count: 0 }
    };

    if (!croppedCats || croppedCats.length === 0) {
      return groups;
    }

    // 전체 고양이 수 계산
    groups['all'].count = croppedCats.length;

    croppedCats.forEach(cat => {
      if (!cat || !cat.id) return;
      
      const name = catNames[cat.id];
      if (name && name.trim()) {
        if (!groups[name]) {
          groups[name] = { name: name.trim(), count: 0 };
        }
        groups[name].count++;
      } else {
        groups['unnamed'].count++;
      }
    });

    return groups;
  };

  const catGroups = getCatGroups();

  // 현재 선택된 그룹의 고양이들 필터링
  const getFilteredCats = () => {
    if (!croppedCats || croppedCats.length === 0) {
      return [];
    }
    
    if (selectedGroup === 'all') {
      return croppedCats;
    }
    if (selectedGroup === 'unnamed') {
      return croppedCats.filter(cat => !cat || !catNames[cat.id]);
    }
    return croppedCats.filter(cat => cat && catNames[cat.id] === selectedGroup);
  };

  const filteredCats = getFilteredCats();

  // 키보드 이벤트 리스너 추가
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Shift') {
        setIsShiftPressed(true);
      }
    };

    const handleKeyUp = (e) => {
      if (e.key === 'Shift') {
        setIsShiftPressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  const handleCatSelect = (catId) => {
    const newSelected = new Set(selectedCats);
    
    if (isShiftPressed && lastSelectedCat) {
      // Shift 키가 눌린 상태에서 연결 선택
      const currentIndex = filteredCats.findIndex(cat => cat.id === catId);
      const lastIndex = filteredCats.findIndex(cat => cat.id === lastSelectedCat);
      
      if (currentIndex !== -1 && lastIndex !== -1) {
        const startIndex = Math.min(currentIndex, lastIndex);
        const endIndex = Math.max(currentIndex, lastIndex);
        
        // 범위 내의 모든 고양이 선택
        for (let i = startIndex; i <= endIndex; i++) {
          newSelected.add(filteredCats[i].id);
        }
        
        setSelectedCats(newSelected);
        setLastSelectedCat(catId);
        return;
      }
    }
    
    // 일반 선택 (Shift 키가 눌리지 않은 경우)
    if (newSelected.has(catId)) {
      newSelected.delete(catId);
    } else {
      newSelected.add(catId);
    }
    
    setSelectedCats(newSelected);
    setLastSelectedCat(catId);
  };

  const handleBulkNameSubmit = () => {
    const name = bulkNameInput.trim();
    if (name && selectedCats.size > 0) {
      const newCatNames = { ...catNames };
      selectedCats.forEach(catId => {
        newCatNames[catId] = name;
      });
      
      setCatNames(newCatNames);
      setBulkNameInput('');
      setSelectedCats(new Set());
      
      const message = `${selectedCats.size}마리의 고양이에게 "${name}" 이름이 등록되었습니다!`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'success');
      } else {
        setStatusMessage({ type: 'success', text: message });
        setTimeout(() => {
          setStatusMessage('');
        }, 3000);
      }
    }
  };

  // 그룹에서 제거하는 함수 추가
  const handleRemoveFromGroup = () => {
    if (selectedCats.size > 0) {
      const newCatNames = { ...catNames };
      let removedCount = 0;
      
      selectedCats.forEach(catId => {
        if (newCatNames[catId]) {
          delete newCatNames[catId];
          removedCount++;
        }
      });
      
      setCatNames(newCatNames);
      setSelectedCats(new Set());
      
      const message = `${removedCount}마리의 고양이가 그룹에서 제거되어 "미지정" 그룹으로 이동했습니다!`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'success');
      } else {
        setStatusMessage({ type: 'success', text: message });
        setTimeout(() => {
          setStatusMessage('');
        }, 3000);
      }
    }
  };

  const handleGroupSelect = (groupName) => {
    setSelectedGroup(groupName);
    setSelectedCats(new Set()); // 그룹 변경 시 선택 해제
  };

  const handleGroupHighlight = (groupName) => {
    setHighlightedGroup(groupName);
    setTimeout(() => setHighlightedGroup(''), 2000);
  };

  const handleSelectGroup = (groupName) => {
    let catIds = [];
    let groupDisplayName = '';
    
    if (groupName === 'all') {
      catIds = croppedCats.map(cat => cat.id);
      groupDisplayName = '전체';
    } else if (groupName === 'unnamed') {
      catIds = croppedCats.filter(cat => !catNames[cat.id]).map(cat => cat.id);
      groupDisplayName = '미지정';
    } else {
      catIds = croppedCats.filter(cat => catNames[cat.id] === groupName).map(cat => cat.id);
      groupDisplayName = groupName;
    }
    
    setSelectedCats(new Set(catIds));
    setSelectedGroup(groupName);
    const message = `${groupDisplayName} 그룹의 ${catIds.length}마리를 선택했습니다.`;
    if (onShowGlobalMessage) {
      onShowGlobalMessage(message, 'info');
    } else {
      setStatusMessage({ type: 'info', text: message });
      setTimeout(() => {
        setStatusMessage('');
      }, 2000);
    }
  };

  const handleGroupAction = (action, groupName) => {
    setModalAction(action);
    setModalGroupName(groupName);
    setModalInputValue(groupName);
    setShowModal(true);
  };

  const handleModalSubmit = () => {
    const newName = modalInputValue.trim();
    
    if (modalAction === 'edit' && newName && newName !== modalGroupName) {
      // 그룹 이름 변경
      const newCatNames = { ...catNames };
      Object.keys(newCatNames).forEach(catId => {
        if (newCatNames[catId] === modalGroupName) {
          newCatNames[catId] = newName;
        }
      });
      setCatNames(newCatNames);
      
      const message = `그룹 이름이 "${modalGroupName}"에서 "${newName}"으로 변경되었습니다!`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'success');
      } else {
        setStatusMessage({ type: 'success', text: message });
      }
    } else if (modalAction === 'delete') {
      // 그룹 삭제 (이름 제거)
      const newCatNames = { ...catNames };
      Object.keys(newCatNames).forEach(catId => {
        if (newCatNames[catId] === modalGroupName) {
          delete newCatNames[catId];
        }
      });
      setCatNames(newCatNames);
      
      const message = `"${modalGroupName}" 그룹이 삭제되었습니다!`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'success');
      } else {
        setStatusMessage({ type: 'success', text: message });
      }
    }
    
    setShowModal(false);
    setModalInputValue('');
    
    if (!onShowGlobalMessage) {
      setTimeout(() => {
        setStatusMessage('');
      }, 3000);
    }
  };

  // 모델 학습 함수
  const handleTeachModel = async () => {
    // 모든 고양이가 미지정인지 확인
    const hasNamedCats = croppedCats.some(cat => catNames[cat.id] && catNames[cat.id].trim());
    
    if (!hasNamedCats) {
      const message = '모든 고양이가 미지정 상태입니다. 먼저 고양이들에게 이름을 지어주세요.';
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'error');
      } else {
        setStatusMessage({ type: 'error', text: message });
        setTimeout(() => setStatusMessage(''), 3000);
      }
      return;
    }

    try {
      setIsTeaching(true);
      setTeachingStep('데이터 준비 중...');
      
      // 학습 데이터 준비 - 이름이 지정된 고양이들만
      const namedCatIds = croppedCats
        .filter(cat => catNames[cat.id] && catNames[cat.id].trim())
        .map(cat => cat.id);
      
      const teachingData = {
        selected_cat_ids: namedCatIds,
        cat_names: catNames
      };
      
      console.log('=== 모델 학습 시작 ===');
      console.log('학습 데이터:', teachingData);
      
      // 단계별 상태 업데이트
      setTimeout(() => setTeachingStep('고양이 이미지 분석 중...'), 500);
      setTimeout(() => setTeachingStep('특성 추출 중...'), 2000);
      setTimeout(() => setTeachingStep('패턴 학습 중...'), 4000);
      setTimeout(() => setTeachingStep('모델 업데이트 중...'), 6000);
      setTimeout(() => setTeachingStep('검증 중...'), 8000);
      
      const response = await fetch('http://localhost:5000/api/yolo/teach-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(teachingData),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setTeachingStep('학습 완료!');
        const message = data.message;
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'success');
        } else {
          setStatusMessage({ type: 'success', text: message });
        }
        
        // 학습 결과 표시
        if (data.learning_results) {
          const results = data.learning_results;
          console.log('학습 결과:', results);
          
          setTimeout(() => {
            const uniqueGroups = new Set(Object.values(catNames).filter(name => name && name.trim())).size;
            const resultMessage = `${uniqueGroups}개 그룹의 고양이로 학습 완료! 정확도: ${(results.learning_accuracy * 100).toFixed(1)}%, 개선률: ${(results.improvement_rate * 100).toFixed(1)}%`;
            if (onShowGlobalMessage) {
              onShowGlobalMessage(resultMessage, 'success');
            } else {
              setStatusMessage({ type: 'success', text: resultMessage });
            }
          }, 2000);
        }
      } else {
        setTeachingStep('학습 실패');
        const message = data.message || '모델 학습에 실패했습니다.';
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'error');
        } else {
          setStatusMessage({ type: 'error', text: message });
        }
      }
    } catch (error) {
      console.error('모델 학습 실패:', error);
      setTeachingStep('오류 발생');
      const message = '모델 학습 중 오류가 발생했습니다.';
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'error');
      } else {
        setStatusMessage({ type: 'error', text: message });
      }
    } finally {
      // 3초 후 학습 모드 종료
      setTimeout(() => {
        setIsTeaching(false);
        setTeachingStep('');
        setStatusMessage('');
      }, 3000);
    }
  };

  const handleSelectAll = () => {
    if (selectedCats.size === filteredCats.length) {
      setSelectedCats(new Set());
    } else {
      setSelectedCats(new Set(filteredCats.map(cat => cat.id)));
    }
  };

  const handleClearSelection = () => {
    setSelectedCats(new Set());
    setBulkNameInput('');
  };

  const formatTime = (seconds) => {
    if (seconds === 0 || !seconds) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    return 'low';
  };

  const getConfidenceText = (confidence) => {
    return `${Math.round(confidence * 100)}%`;
  };

  // 로딩 상태 표시
  if (isGalleryLoading) {
    return (
      <GalleryContainer>
        <Header>
          <Title>🐱 다둥이 갤러리</Title>
          <Stats>로딩 중...</Stats>
        </Header>
        
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div style={{ fontSize: '2rem', marginBottom: '20px' }}>⏳</div>
          <p>저장된 고양이 데이터를 불러오는 중...</p>
        </div>
      </GalleryContainer>
    );
  }

  if (!croppedCats || croppedCats.length === 0) {
    return (
      <GalleryContainer>
        <Header>
          <Title>🐱 다둥이 갤러리</Title>
          <Stats>감지된 고양이: 0마리</Stats>
        </Header>
        
        <Controls>
          <Button className="secondary" onClick={onBack}>
            ← 뒤로 가기
          </Button>
          <Button className="info" onClick={onRefresh}>
            🔄 새로고침
          </Button>
          <Button className="danger" onClick={onReset}>
            🔄 다시 시작
          </Button>
        </Controls>

        <EmptyState>
          <EmptyIcon>🐾</EmptyIcon>
          <EmptyText>아직 감지된 고양이가 없습니다.</EmptyText>
          <EmptyText>영상을 업로드하고 처리해보세요!</EmptyText>
        </EmptyState>
      </GalleryContainer>
    );
  }

  return (
    <GalleryContainer>
      <Header>
        <Title>🐱 다둥이 갤러리</Title>
        <Stats>
          총 {croppedCats.length}마리 중 {selectedCats.size}마리 선택됨
          {uploadSummary && (
            <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '5px' }}>
              {uploadSummary.message}
            </div>
          )}
        </Stats>
      </Header>

      <FilterSection>
        <FilterTitle>🐱 우리 집에 사는  </FilterTitle>
        <GroupSelector>
          {Object.entries(catGroups)
            .sort(([aKey, aGroup], [bKey, bGroup]) => {
              // 전체를 첫 번째로
              if (aKey === 'all') return -1;
              if (bKey === 'all') return 1;
              // 미지정을 두 번째로
              if (aKey === 'unnamed') return -1;
              if (bKey === 'unnamed') return 1;
              // 나머지는 이름 순으로 정렬
              return aGroup.name.localeCompare(bGroup.name);
            })
            .map(([groupKey, group]) => (
            <GroupButton
              key={groupKey}
              selected={selectedGroup === groupKey}
              onClick={() => handleGroupSelect(groupKey)}
              onDoubleClick={() => handleSelectGroup(groupKey)}
              onMouseEnter={() => handleGroupHighlight(groupKey)}
              borderColor={groupKey === 'unnamed' || groupKey === 'all' ? '#e2e8f0' : generateBorderColor(group.name || groupKey)}
              backgroundColor={groupKey === 'unnamed' || groupKey === 'all' ? '#ffffff' : generateBackgroundColor(group.name || groupKey)}
            >
              {group.name}
              <GroupCount selected={selectedGroup === groupKey}>
                {group.count}
              </GroupCount>
              {groupKey !== 'unnamed' && groupKey !== 'all' && (
                <GroupActions>
                  <ActionButton
                    action="edit"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleGroupAction('edit', groupKey);
                    }}
                    title="그룹 이름 수정"
                  >
                    ✏️
                  </ActionButton>
                  <ActionButton
                    action="delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleGroupAction('delete', groupKey);
                    }}
                    title="그룹 삭제"
                  >
                    🗑️
                  </ActionButton>
                </GroupActions>
              )}
            </GroupButton>
          ))}
        </GroupSelector>
      </FilterSection>

      <BulkActionSection show={selectedCats.size > 0}>
        <BulkActionTitle>🐱 선택한 이미지의 고양이 이름은 무엇인가요? ({selectedCats.size}개 이미지)</BulkActionTitle>
        <BulkActionDescription>
          선택한 고양이 이미지들을 보고 같은 고양이라고 판단되면 이름을 지어주세요.
          같은 고양이의 다른 사진들이라면 동일한 이름을 사용하면 됩니다.
        </BulkActionDescription>
        <BulkActionControls>
          <BulkNameInput
            placeholder="고양이 이름을 입력하세요 (예: 루시, 미미, 토미)"
            value={bulkNameInput}
            onChange={(e) => setBulkNameInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleBulkNameSubmit();
              }
            }}
          />
          <Button 
            className="success"
            onClick={handleBulkNameSubmit}
            disabled={!bulkNameInput.trim()}
          >
            이름 설정
          </Button>
          <Button 
            className="warning"
            onClick={handleRemoveFromGroup}
            disabled={!Array.from(selectedCats).some(catId => catNames[catId])}
            title="선택된 고양이들을 그룹에서 제거하여 '미지정' 그룹으로 이동"
          >
            그룹에서 제거
          </Button>
          <Button 
            className="secondary"
            onClick={handleClearSelection}
          >
            선택 해제
          </Button>
        </BulkActionControls>
      </BulkActionSection>

      <Controls>
        <Button className="secondary" onClick={onBack}>
          ← 뒤로 가기
        </Button>
        <Button 
          className={selectedCats.size === filteredCats.length ? 'danger' : 'primary'}
          onClick={handleSelectAll}
        >
          {selectedCats.size === filteredCats.length ? '전체 해제' : '전체 선택'}
        </Button>
        <Button 
          className="info"
          onClick={() => handleSelectGroup(selectedGroup)}
        >
          현재 그룹 선택
        </Button>
        <Button 
          className="success"
          onClick={saveCatGroups}
          disabled={isLoading}
          title="현재 그룹 정보를 서버에 저장"
        >
          {isLoading ? '저장 중...' : ' 그룹 저장'}
        </Button>
        <Button 
          className="info"
          onClick={loadCatGroups}
          disabled={isLoading}
          title="서버에서 저장된 그룹 정보를 다시 로드"
        >
          {isLoading ? '로드 중...' : ' 그룹 로드'}
        </Button>
        <Button 
          className="danger"
          onClick={handleTeachModel}
          disabled={isLoading || !croppedCats.some(cat => catNames[cat.id] && catNames[cat.id].trim())}
          title="그룹이 지정된 고양이들을 AI 모델에게 알려주기"
        >
          🧠 알려주기
        </Button>
      </Controls>

      {statusMessage && !onShowGlobalMessage && (
        <StatusMessage className={statusMessage.type}>
          {statusMessage.text}
        </StatusMessage>
      )}

      <GallerySection>
        <GalleryHeader onClick={() => setIsGalleryCollapsed(!isGalleryCollapsed)}>
          <GalleryTitle>
            🖼️ 고양이 이미지 갤러리 ({filteredCats.length}마리)
          </GalleryTitle>
          <GalleryToggle>
            {isGalleryCollapsed ? '▼' : '▲'}
          </GalleryToggle>
        </GalleryHeader>
        
        <GalleryContent collapsed={isGalleryCollapsed}>
          <GalleryGrid>
            {filteredCats.map((cat) => (
              <CatCard
                key={cat.id}
                selected={selectedCats.has(cat.id)}
                highlighted={highlightedGroup && catNames[cat.id] === highlightedGroup}
                onClick={() => handleCatSelect(cat.id)}
                onMouseDown={(e) => e.preventDefault()} // 드래그 방지
                style={{
                  position: 'relative',
                  cursor: isShiftPressed ? 'crosshair' : 'pointer',
                  userSelect: 'none', // 텍스트 선택 방지
                  WebkitUserSelect: 'none',
                  MozUserSelect: 'none',
                  msUserSelect: 'none'
                }}
              >
                {isShiftPressed && (
                  <div style={{
                    position: 'absolute',
                    top: '5px',
                    right: '5px',
                    background: 'rgba(0, 123, 255, 0.8)',
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontSize: '0.7rem',
                    zIndex: 10
                  }}>
                    Shift
                  </div>
                )}
                
                <CatImage
                  onDragStart={(e) => e.preventDefault()} // 이미지 드래그 방지
                  style={{
                    userSelect: 'none',
                    WebkitUserSelect: 'none',
                    MozUserSelect: 'none',
                    msUserSelect: 'none'
                  }}
                >
                  {cat.url ? (
                    <img 
                      src={`http://localhost:5000${cat.url}`} 
                      alt={`고양이 ${cat.id}`}
                      draggable={false} // 이미지 드래그 비활성화
                      onDragStart={(e) => e.preventDefault()}
                      onError={(e) => {
                        console.error('이미지 로드 실패:', cat.url);
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                  ) : cat.filename ? (
                    <img 
                      src={`http://localhost:5000/cropped-images/${cat.filename}`} 
                      alt={`고양이 ${cat.id}`}
                      draggable={false} // 이미지 드래그 비활성화
                      onDragStart={(e) => e.preventDefault()}
                      onError={(e) => {
                        console.error('이미지 로드 실패:', cat.filename);
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                  ) : null}
                  <div className="no-image" style={{ display: (cat.url || cat.filename) ? 'none' : 'flex' }}>
                    🐱 이미지 없음
                  </div>
                </CatImage>
                
                <CatInfo
                  style={{
                    userSelect: 'none',
                    WebkitUserSelect: 'none',
                    MozUserSelect: 'none',
                    msUserSelect: 'none'
                  }}
                >
                  <CatTitle
                    groupColor={catNames[cat.id] ? generateBackgroundColor(catNames[cat.id]) : 'rgba(156, 163, 175, 0.1)'}
                    groupBorderColor={catNames[cat.id] ? generateBorderColor(catNames[cat.id]) : 'rgba(156, 163, 175, 0.8)'}
                  >
                    {catNames[cat.id] || '미지정'}
                  </CatTitle>
                  <CatDetails>
                    <div>시간: {formatTime(cat.timestamp || 0)}</div>
                    {cat.filename && (
                      <div style={{ fontSize: '0.8rem', color: '#888', marginTop: '5px' }}>
                        파일: {cat.filename}
                      </div>
                    )}
                  </CatDetails>
                </CatInfo>
              </CatCard>
            ))}
          </GalleryGrid>
        </GalleryContent>
      </GallerySection>

      {showModal && (
        <Modal onClick={() => setShowModal(false)}>
          <ModalContent onClick={(e) => e.stopPropagation()}>
            <ModalTitle>
              {modalAction === 'edit' ? '그룹 이름 수정' : '그룹 삭제'}
            </ModalTitle>
            {modalAction === 'edit' ? (
              <>
                <ModalInput
                  placeholder="새로운 그룹 이름을 입력하세요"
                  value={modalInputValue}
                  onChange={(e) => setModalInputValue(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleModalSubmit();
                    }
                  }}
                  autoFocus
                />
                <ModalButtons>
                  <Button 
                    className="secondary" 
                    onClick={() => setShowModal(false)}
                  >
                    취소
                  </Button>
                  <Button 
                    className="success" 
                    onClick={handleModalSubmit}
                    disabled={!modalInputValue.trim() || modalInputValue.trim() === modalGroupName}
                  >
                    수정
                  </Button>
                </ModalButtons>
              </>
            ) : (
              <>
                <p>정말로 "{modalGroupName}" 그룹을 삭제하시겠습니까?</p>
                <p>이 그룹의 모든 고양이 이름이 제거됩니다.</p>
                <ModalButtons>
                  <Button 
                    className="secondary" 
                    onClick={() => setShowModal(false)}
                  >
                    취소
                  </Button>
                  <Button 
                    className="danger" 
                    onClick={handleModalSubmit}
                  >
                    🗑️ 삭제
                  </Button>
                </ModalButtons>
              </>
            )}
          </ModalContent>
        </Modal>
      )}

      {isTeaching && (
        <TeachingOverlay>
          <TeachingContent>
            <TeachingIcon>🧠</TeachingIcon>
            <TeachingTitle>AI 모델 학습 중...</TeachingTitle>
            <TeachingDescription>
              그룹이 지정된 고양이 이미지들을 AI 모델에게 알려주고 있습니다.
              <br />
              이 과정은 몇 분 정도 소요될 수 있습니다.
            </TeachingDescription>
            <TeachingProgress>
              <TeachingProgressBar />
            </TeachingProgress>
            <TeachingStatus>
              {teachingStep || '고양이 특성 분석 중...'}
            </TeachingStatus>
          </TeachingContent>
        </TeachingOverlay>
      )}
    </GalleryContainer>
  );
}

export default CatGallery; 