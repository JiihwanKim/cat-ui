import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const ConfigContainer = styled.div`
  color: #2d3748;
`;

const Title = styled.h2`
  color: #2d3748;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 24px;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const Message = styled.div`
  padding: 16px;
  margin-bottom: 20px;
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
`;

const ConfigSection = styled.div`
  margin-bottom: 32px;
  padding: 24px;
  background: #f7fafc;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
`;

const SectionTitle = styled.h3`
  color: #2d3748;
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 20px;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const ConfigItem = styled.div`
  margin-bottom: 20px;
  
  label {
    display: block;
    color: #2d3748;
    font-weight: 600;
    margin-bottom: 8px;
  }
  
  input[type="range"] {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: #e2e8f0;
    outline: none;
    -webkit-appearance: none;
    
    &::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    &::-moz-range-thumb {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
      cursor: pointer;
      border: none;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
  }
  
    input[type="number"], select {
    width: 100%;
    padding: 12px 16px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #ffffff;
    color: #2d3748;
    font-size: 1rem;

    &:focus {
      outline: none;
      border-color: #3182ce;
      box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
    }

    &::placeholder {
      color: #a0aec0;
    }
  }
  
  span {
    display: inline-block;
    margin-left: 12px;
    color: rgba(102, 126, 234, 1);
    font-weight: 700;
    font-size: 1.1rem;
  }
  
  small {
    display: block;
    color: #4a5568;
    font-size: 0.9rem;
    margin-top: 4px;
  }
  
  input[type="checkbox"] {
    margin-right: 8px;
    transform: scale(1.2);
  }
`;

const ModelStatus = styled.div`
  .status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #e2e8f0;
    
    &:last-child {
      border-bottom: none;
    }
    
    strong {
      color: #2d3748;
      font-weight: 600;
    }
    
    .status-success {
      color: #38a169;
      font-weight: 700;
    }
    
    .status-error {
      color: #e53e3e;
      font-weight: 700;
    }
  }
`;

const ModelFiles = styled.div`
  margin-top: 20px;
  
  h4 {
    color: #2d3748;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 16px;
  }
  
  .model-file {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: #f7fafc;
    border-radius: 12px;
    margin-bottom: 8px;
    border: 1px solid #e2e8f0;
    
    .file-exists {
      color: #38a169;
      font-weight: 600;
    }
    
    .file-missing {
      color: #e53e3e;
      font-weight: 600;
    }
    
    .file-size {
      color: #718096;
      font-size: 0.9rem;
    }
  }
`;

const ModelActions = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 20px;
  flex-wrap: wrap;
`;

const Button = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }
  }

  &.secondary {
    background: #f7fafc;
    color: #2d3748;
    border: 1px solid #e2e8f0;
    
    &:hover {
      background: #edf2f7;
      transform: translateY(-2px);
    }
  }

  &.success {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    color: white;
    box-shadow: 0 8px 25px rgba(86, 171, 47, 0.3);
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 35px rgba(86, 171, 47, 0.4);
    }
  }

  &.danger {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    color: white;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 35px rgba(255, 107, 107, 0.4);
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const PresetButtons = styled.div`
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
`;

const PresetButton = styled(Button)`
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  color: white;
  box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(79, 172, 254, 0.4);
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 24px;
  flex-wrap: wrap;
`;

const ConfigInfo = styled.div`
  margin-top: 32px;
  
  h3 {
    color: #2d3748;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 16px;
  }
  
  pre {
    background: #2d3748;
    padding: 20px;
    border-radius: 12px;
    color: #e2e8f0;
    font-size: 0.9rem;
    overflow-x: auto;
    border: 1px solid #e2e8f0;
  }
`;

const DownloadButton = styled(Button)`
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  font-size: 0.9rem;
  padding: 8px 16px;
  box-shadow: 0 6px 20px rgba(240, 147, 251, 0.3);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(240, 147, 251, 0.4);
  }
`;

const YOLOConfig = () => {
  const [config, setConfig] = useState({
    conf: 0.25,
    iou: 0.7,
    imgsz: 640,
    max_det: 20,
    classes: [16],
    verbose: false,
    agnostic_nms: false,
    amp: true
  });
  
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    fetchConfig();
    fetchModelStatus();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/yolo/config');
      const data = await response.json();
      if (data.success) {
        setConfig(data.inferenceConfig);
      }
    } catch (error) {
      console.error('설정 조회 실패:', error);
    }
  };

  const fetchModelStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/yolo/model-status');
      const data = await response.json();
      if (data.success) {
        setModelStatus(data);
      }
    } catch (error) {
      console.error('모델 상태 조회 실패:', error);
    }
  };

  const updateConfig = async (newConfig) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/yolo/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newConfig),
      });
      
      const data = await response.json();
      if (data.success) {
        setConfig(data.updatedConfig);
        setMessage({ text: '설정이 업데이트되었습니다.', type: 'success' });
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error) {
      setMessage({ text: '설정 업데이트 실패', type: 'error' });
      console.error('설정 업데이트 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetConfig = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/yolo/reset-config', {
        method: 'POST',
      });
      
      const data = await response.json();
      if (data.success) {
        setConfig(data.config);
        setMessage({ text: '설정이 기본값으로 초기화되었습니다.', type: 'success' });
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error) {
      setMessage({ text: '설정 초기화 실패', type: 'error' });
      console.error('설정 초기화 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const reloadModel = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/yolo/reload', {
        method: 'POST',
      });
      
      const data = await response.json();
      if (data.success) {
        setMessage({ text: '모델이 재로드되었습니다.', type: 'success' });
        setTimeout(() => setMessage(null), 3000);
        fetchModelStatus(); // 상태 새로고침
      }
    } catch (error) {
      setMessage({ text: '모델 재로드 실패', type: 'error' });
      console.error('모델 재로드 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const downloadModel = async (modelName = 'yolo11n.pt') => {
    setDownloading(true);
    try {
      const response = await fetch('http://localhost:5000/api/yolo/download-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });
      
      const data = await response.json();
      if (data.success) {
        setMessage({ text: `${modelName} 모델이 성공적으로 다운로드되었습니다.`, type: 'success' });
        setTimeout(() => setMessage(null), 3000);
        fetchModelStatus(); // 상태 새로고침
      }
    } catch (error) {
      setMessage({ text: '모델 다운로드 실패', type: 'error' });
      console.error('모델 다운로드 실패:', error);
    } finally {
      setDownloading(false);
    }
  };

  const handleInputChange = (key, value) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    updateConfig(config);
  };

  const presetConfigs = {
    highAccuracy: {
      name: '높은 정확도',
      config: { conf: 0.5, iou: 0.8, imgsz: 800, max_det: 10 }
    },
    fastProcessing: {
      name: '빠른 처리',
      config: { conf: 0.25, iou: 0.6, imgsz: 480, max_det: 20 }
    },
    balanced: {
      name: '균형잡힌 설정',
      config: { conf: 0.35, iou: 0.7, imgsz: 640, max_det: 15 }
    }
  };

  const applyPreset = (presetKey) => {
    const preset = presetConfigs[presetKey];
    const newConfig = { ...config, ...preset.config };
    setConfig(newConfig);
    updateConfig(newConfig);
  };

  return (
    <ConfigContainer>
      <Title>🐱 YOLO 설정</Title>
      
      {message && typeof message === 'object' && message.text && (
        <Message className={message.type}>
          {message.text}
        </Message>
      )}

      {/* 모델 상태 섹션 */}
      <ConfigSection>
        <SectionTitle>📦 모델 상태</SectionTitle>
        {modelStatus && (
          <ModelStatus>
            <div className="status-item">
              <strong>모델 로드 상태:</strong>
              <span className={modelStatus.modelLoaded ? 'status-success' : 'status-error'}>
                {modelStatus.modelLoaded ? '✅ 로드됨' : '❌ 로드되지 않음'}
              </span>
            </div>
            
            <ModelFiles>
              <h4>모델 파일 상태:</h4>
              {modelStatus.models.map((model, index) => (
                <div key={index} className="model-file">
                  <span className={model.exists ? 'file-exists' : 'file-missing'}>
                    {model.exists ? '📁' : '❌'} {model.name}
                  </span>
                  {model.exists && (
                    <span className="file-size">({model.size_mb} MB)</span>
                  )}
                  {!model.exists && (
                    <DownloadButton 
                      onClick={() => downloadModel(model.name)}
                      disabled={downloading}
                    >
                      {downloading ? '다운로드 중...' : '다운로드'}
                    </DownloadButton>
                  )}
                </div>
              ))}
            </ModelFiles>
            
            <ModelActions>
              <Button className="secondary" onClick={fetchModelStatus}>
                🔄 상태 새로고침
              </Button>
              <Button className="success" onClick={reloadModel} disabled={loading}>
                🔄 모델 재로드
              </Button>
            </ModelActions>
          </ModelStatus>
        )}
      </ConfigSection>

      <form onSubmit={handleSubmit}>
        <ConfigSection>
          <SectionTitle>추론 설정</SectionTitle>
          
          <ConfigItem>
            <label htmlFor="conf">신뢰도 임계값 (Confidence)</label>
            <input
              type="range"
              id="conf"
              min="0.1"
              max="0.9"
              step="0.05"
              value={config.conf}
              onChange={(e) => handleInputChange('conf', parseFloat(e.target.value))}
            />
            <span>{config.conf}</span>
            <small>0.1 (높은 감지율) - 0.9 (높은 정확도)</small>
          </ConfigItem>

          <ConfigItem>
            <label htmlFor="iou">IoU 임계값</label>
            <input
              type="range"
              id="iou"
              min="0.3"
              max="0.9"
              step="0.1"
              value={config.iou}
              onChange={(e) => handleInputChange('iou', parseFloat(e.target.value))}
            />
            <span>{config.iou}</span>
            <small>0.3 (관대한 필터링) - 0.9 (엄격한 필터링)</small>
          </ConfigItem>

          <ConfigItem>
            <label htmlFor="imgsz">이미지 크기</label>
            <select
              id="imgsz"
              value={config.imgsz}
              onChange={(e) => handleInputChange('imgsz', parseInt(e.target.value))}
            >
              <option value={320}>320 (빠른 처리)</option>
              <option value={480}>480 (빠른 처리)</option>
              <option value={640}>640 (균형잡힌 성능)</option>
              <option value={800}>800 (높은 정확도)</option>
              <option value={1280}>1280 (최고 정확도)</option>
            </select>
          </ConfigItem>

          <ConfigItem>
            <label htmlFor="max_det">최대 감지 수</label>
            <input
              type="number"
              id="max_det"
              min="1"
              max="50"
              value={config.max_det}
              onChange={(e) => handleInputChange('max_det', parseInt(e.target.value))}
            />
          </ConfigItem>
        </ConfigSection>

        <ConfigSection>
          <SectionTitle>성능 최적화</SectionTitle>
          
          <ConfigItem>
            <label>
              <input
                type="checkbox"
                checked={config.amp}
                onChange={(e) => handleInputChange('amp', e.target.checked)}
              />
              혼합 정밀도 (Mixed Precision)
            </label>
            <small>메모리 사용량 감소 및 속도 향상</small>
          </ConfigItem>

          <ConfigItem>
            <label>
              <input
                type="checkbox"
                checked={config.agnostic_nms}
                onChange={(e) => handleInputChange('agnostic_nms', e.target.checked)}
              />
              클래스별 NMS
            </label>
            <small>클래스별로 NMS 수행</small>
          </ConfigItem>

          <ConfigItem>
            <label>
              <input
                type="checkbox"
                checked={config.verbose}
                onChange={(e) => handleInputChange('verbose', e.target.checked)}
              />
              상세 출력
            </label>
            <small>추론 과정의 상세한 로그 출력</small>
          </ConfigItem>
        </ConfigSection>

        <ConfigSection>
          <SectionTitle>사전 설정</SectionTitle>
          <PresetButtons>
            {Object.entries(presetConfigs).map(([key, preset]) => (
              <PresetButton
                key={key}
                type="button"
                onClick={() => applyPreset(key)}
              >
                {preset.name}
              </PresetButton>
            ))}
          </PresetButtons>
        </ConfigSection>

        <ActionButtons>
          <Button type="submit" className="primary" disabled={loading}>
            {loading ? '업데이트 중...' : '설정 업데이트'}
          </Button>
          <Button type="button" className="secondary" onClick={resetConfig} disabled={loading}>
            기본값으로 초기화
          </Button>
          <Button type="button" className="success" onClick={reloadModel} disabled={loading}>
            모델 재로드
          </Button>
        </ActionButtons>
      </form>

      <ConfigInfo>
        <h3>현재 설정 정보</h3>
        <pre>{JSON.stringify(config, null, 2)}</pre>
      </ConfigInfo>
    </ConfigContainer>
  );
};

export default YOLOConfig; 