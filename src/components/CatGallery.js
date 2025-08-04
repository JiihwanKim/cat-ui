import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const GalleryContainer = styled.div`
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 20px;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 2px solid #f0f0f0;
`;

const Title = styled.h2`
  color: #333;
  margin: 0;
`;

const Stats = styled.div`
  text-align: right;
  color: #666;
`;

const Controls = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
`;

const Button = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;

  &.primary {
    background: #007bff;
    color: white;
    
    &:hover {
      background: #0056b3;
    }
  }

  &.secondary {
    background: #6c757d;
    color: white;
    
    &:hover {
      background: #545b62;
    }
  }

  &.success {
    background: #28a745;
    color: white;
    
    &:hover {
      background: #1e7e34;
    }
  }

  &.danger {
    background: #dc3545;
    color: white;
    
    &:hover {
      background: #c82333;
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const GalleryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
`;

const CatCard = styled.div`
  border: 2px solid ${props => props.selected ? '#007bff' : '#e9ecef'};
  border-radius: 8px;
  overflow: hidden;
  background: white;
  transition: all 0.3s ease;
  cursor: pointer;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &.selected {
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }
`;

const CatImage = styled.div`
  position: relative;
  width: 100%;
  height: 200px;
  background: #f8f9fa;
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
    color: #6c757d;
    font-size: 0.9rem;
    text-align: center;
  }
`;

const CatInfo = styled.div`
  padding: 15px;
`;

const CatTitle = styled.h3`
  margin: 0 0 8px 0;
  font-size: 1rem;
  color: #333;
`;

const CatDetails = styled.div`
  font-size: 0.85rem;
  color: #666;
  line-height: 1.4;
`;

const ConfidenceBadge = styled.span`
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
  margin-top: 5px;

  &.high {
    background: #d4edda;
    color: #155724;
  }

  &.medium {
    background: #fff3cd;
    color: #856404;
  }

  &.low {
    background: #f8d7da;
    color: #721c24;
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 40px 20px;
  color: #6c757d;
`;

const EmptyIcon = styled.div`
  font-size: 3rem;
  margin-bottom: 15px;
  opacity: 0.5;
`;

const EmptyText = styled.p`
  margin: 0;
  font-size: 1.1rem;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
  margin: 10px 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #007bff, #28a745);
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
`;

const StatusMessage = styled.div`
  text-align: center;
  padding: 15px;
  margin: 10px 0;
  border-radius: 6px;
  font-weight: 500;

  &.success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
  }

  &.error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
  }

  &.info {
    background: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
  }
`;

function CatGallery({ croppedCats, onBack, onReset }) {
  const [selectedCats, setSelectedCats] = useState(new Set());
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  // 디버깅을 위한 로그
  useEffect(() => {
    console.log('CatGallery - croppedCats:', croppedCats);
    if (croppedCats && croppedCats.length > 0) {
      console.log('크롭된 고양이 수:', croppedCats.length);
      croppedCats.forEach((cat, index) => {
        console.log(`고양이 ${index + 1}:`, cat);
      });
    }
  }, [croppedCats]);

  const handleCatSelect = (catId) => {
    const newSelected = new Set(selectedCats);
    if (newSelected.has(catId)) {
      newSelected.delete(catId);
    } else {
      newSelected.add(catId);
    }
    setSelectedCats(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedCats.size === croppedCats.length) {
      setSelectedCats(new Set());
    } else {
      setSelectedCats(new Set(croppedCats.map(cat => cat.id)));
    }
  };

  const handleUpload = async () => {
    if (selectedCats.size === 0) {
      setStatusMessage({ type: 'error', text: '업로드할 고양이를 선택해주세요.' });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setStatusMessage({ type: 'info', text: '고양이 데이터를 업로드하는 중...' });

    try {
      const selectedCatData = croppedCats.filter(cat => selectedCats.has(cat.id));
      
      // 진행률 시뮬레이션
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await fetch('http://localhost:5000/api/cats/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          cats: selectedCatData
        })
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.ok) {
        const result = await response.json();
        setStatusMessage({ 
          type: 'success', 
          text: `${result.uploadedCount}마리의 고양이가 성공적으로 업로드되었습니다!` 
        });
        
        // 업로드 완료 후 선택 해제
        setTimeout(() => {
          setSelectedCats(new Set());
          setUploadProgress(0);
          setStatusMessage('');
        }, 3000);
      } else {
        const error = await response.json();
        setStatusMessage({ type: 'error', text: error.error || '업로드 중 오류가 발생했습니다.' });
      }
    } catch (error) {
      setStatusMessage({ type: 'error', text: '네트워크 오류가 발생했습니다.' });
    } finally {
      setIsUploading(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    return 'low';
  };

  const getConfidenceText = (confidence) => {
    return `${Math.round(confidence * 100)}%`;
  };

  if (!croppedCats || croppedCats.length === 0) {
    return (
      <GalleryContainer>
        <Header>
          <Title>🐱 고양이 갤러리</Title>
          <Stats>감지된 고양이: 0마리</Stats>
        </Header>
        
        <Controls>
          <Button className="secondary" onClick={onBack}>
            ← 뒤로 가기
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
        <Title>🐱 고양이 갤러리</Title>
        <Stats>
          총 {croppedCats.length}마리 중 {selectedCats.size}마리 선택됨
        </Stats>
      </Header>

      <Controls>
        <Button className="secondary" onClick={onBack}>
          ← 뒤로 가기
        </Button>
        <Button 
          className={selectedCats.size === croppedCats.length ? 'danger' : 'primary'}
          onClick={handleSelectAll}
        >
          {selectedCats.size === croppedCats.length ? '전체 해제' : '전체 선택'}
        </Button>
        <Button 
          className="success" 
          onClick={handleUpload}
          disabled={isUploading || selectedCats.size === 0}
        >
          {isUploading ? '업로드 중...' : '선택한 고양이 업로드'}
        </Button>
        <Button className="danger" onClick={onReset}>
          🔄 다시 시작
        </Button>
      </Controls>

      {statusMessage && (
        <StatusMessage className={statusMessage.type}>
          {statusMessage.text}
        </StatusMessage>
      )}

      {isUploading && (
        <ProgressBar>
          <ProgressFill progress={uploadProgress} />
        </ProgressBar>
      )}

      <GalleryGrid>
        {croppedCats.map((cat) => (
          <CatCard
            key={cat.id}
            selected={selectedCats.has(cat.id)}
            onClick={() => handleCatSelect(cat.id)}
          >
            <CatImage>
              {cat.url ? (
                <img 
                  src={`http://localhost:5000${cat.url}`} 
                  alt={`고양이 ${cat.id}`}
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
            
            <CatInfo>
              <CatTitle>고양이 {cat.id.split('-')[1] || cat.id}</CatTitle>
              <CatDetails>
                <div>프레임: {cat.frame}</div>
                <div>시간: {formatTime(cat.timestamp)}</div>
                <ConfidenceBadge className={getConfidenceLevel(cat.confidence)}>
                  신뢰도: {getConfidenceText(cat.confidence)}
                </ConfidenceBadge>
              </CatDetails>
            </CatInfo>
          </CatCard>
        ))}
      </GalleryGrid>
    </GalleryContainer>
  );
}

export default CatGallery; 