import React, { useState, useEffect } from 'react';
import './YOLOConfig.css';

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
  const [message, setMessage] = useState('');
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
        setMessage('설정이 업데이트되었습니다.');
        setTimeout(() => setMessage(''), 3000);
      }
    } catch (error) {
      setMessage('설정 업데이트 실패');
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
        setMessage('설정이 기본값으로 초기화되었습니다.');
        setTimeout(() => setMessage(''), 3000);
      }
    } catch (error) {
      setMessage('설정 초기화 실패');
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
        setMessage('모델이 재로드되었습니다.');
        setTimeout(() => setMessage(''), 3000);
        fetchModelStatus(); // 상태 새로고침
      }
    } catch (error) {
      setMessage('모델 재로드 실패');
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
        setMessage(`${modelName} 모델이 성공적으로 다운로드되었습니다.`);
        setTimeout(() => setMessage(''), 3000);
        fetchModelStatus(); // 상태 새로고침
      }
    } catch (error) {
      setMessage('모델 다운로드 실패');
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
    <div className="yolo-config">
      <h2>🐱 YOLO 설정</h2>
      
      {message && (
        <div className={`message ${message.includes('실패') ? 'error' : 'success'}`}>
          {message}
        </div>
      )}

      {/* 모델 상태 섹션 */}
      <div className="config-section">
        <h3>📦 모델 상태</h3>
        {modelStatus && (
          <div className="model-status">
            <div className="status-item">
              <strong>모델 로드 상태:</strong>
              <span className={modelStatus.modelLoaded ? 'status-success' : 'status-error'}>
                {modelStatus.modelLoaded ? '✅ 로드됨' : '❌ 로드되지 않음'}
              </span>
            </div>
            
            <div className="model-files">
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
                    <button 
                      onClick={() => downloadModel(model.name)}
                      disabled={downloading}
                      className="download-button"
                    >
                      {downloading ? '다운로드 중...' : '다운로드'}
                    </button>
                  )}
                </div>
              ))}
            </div>
            
            <div className="model-actions">
              <button onClick={fetchModelStatus} className="refresh-button">
                🔄 상태 새로고침
              </button>
              <button onClick={reloadModel} disabled={loading} className="reload-button">
                🔄 모델 재로드
              </button>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        <div className="config-section">
          <h3>추론 설정</h3>
          
          <div className="config-item">
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
          </div>

          <div className="config-item">
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
          </div>

          <div className="config-item">
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
          </div>

          <div className="config-item">
            <label htmlFor="max_det">최대 감지 수</label>
            <input
              type="number"
              id="max_det"
              min="1"
              max="50"
              value={config.max_det}
              onChange={(e) => handleInputChange('max_det', parseInt(e.target.value))}
            />
          </div>
        </div>

        <div className="config-section">
          <h3>성능 최적화</h3>
          
          <div className="config-item">
            <label>
              <input
                type="checkbox"
                checked={config.amp}
                onChange={(e) => handleInputChange('amp', e.target.checked)}
              />
              혼합 정밀도 (Mixed Precision)
            </label>
            <small>메모리 사용량 감소 및 속도 향상</small>
          </div>

          <div className="config-item">
            <label>
              <input
                type="checkbox"
                checked={config.agnostic_nms}
                onChange={(e) => handleInputChange('agnostic_nms', e.target.checked)}
              />
              클래스별 NMS
            </label>
            <small>클래스별로 NMS 수행</small>
          </div>

          <div className="config-item">
            <label>
              <input
                type="checkbox"
                checked={config.verbose}
                onChange={(e) => handleInputChange('verbose', e.target.checked)}
              />
              상세 출력
            </label>
            <small>추론 과정의 상세한 로그 출력</small>
          </div>
        </div>

        <div className="config-section">
          <h3>사전 설정</h3>
          <div className="preset-buttons">
            {Object.entries(presetConfigs).map(([key, preset]) => (
              <button
                key={key}
                type="button"
                onClick={() => applyPreset(key)}
                className="preset-button"
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        <div className="action-buttons">
          <button type="submit" disabled={loading}>
            {loading ? '업데이트 중...' : '설정 업데이트'}
          </button>
          <button type="button" onClick={resetConfig} disabled={loading}>
            기본값으로 초기화
          </button>
          <button type="button" onClick={reloadModel} disabled={loading}>
            모델 재로드
          </button>
        </div>
      </form>

      <div className="config-info">
        <h3>현재 설정 정보</h3>
        <pre>{JSON.stringify(config, null, 2)}</pre>
      </div>
    </div>
  );
};

export default YOLOConfig; 