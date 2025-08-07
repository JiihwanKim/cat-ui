import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const GalleryContainer = styled.div`
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  border-radius: 24px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  padding: 32px;
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  animation: fadeIn 0.4s ease-out;
  transition: all 0.3s ease;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 2px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  transition: border-color 0.3s ease;
`;

const Title = styled.h2`
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: color 0.3s ease;
`;

const Stats = styled.div`
  text-align: right;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-weight: 500;
  transition: color 0.3s ease;
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
    background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
    color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
    border: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
    transition: all 0.3s ease;
    
    &:hover {
      background: ${props => props.darkMode ? '#718096' : '#edf2f7'};
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

  &.warning {
    background: #d69e2e;
    color: white;
    box-shadow: 0 4px 12px rgba(214, 158, 46, 0.2);
    
    &:hover {
      background: #b7791f;
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(214, 158, 46, 0.3);
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none !important;
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
  border: 2px solid ${props => {
    if (props.$isActive) return '#38a169';
    if (props.selected) return '#3182ce';
    return props.$borderColor || '#e2e8f0';
  }};
  background: ${props => {
    if (props.$isActive) return '#38a169';
    if (props.selected) return '#3182ce';
    return props.$backgroundColor || '#ffffff';
  }};
  color: ${props => props.selected || props.$isActive ? 'white' : '#2d3748'};
  border-radius: 20px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 600;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 12px;
  position: relative;
  overflow: hidden;
  min-width: 140px;
  height: 60px;
  justify-content: center;
  
  &:hover {
    border-color: ${props => props.$isActive ? '#2f855a' : '#3182ce'};
    background: ${props => {
      if (props.$isActive) return '#2f855a';
      if (props.selected) return '#2c5aa0';
      return '#f7fafc';
    }};
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
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
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.3s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
  
  ${props => props.selected && `
    box-shadow: 0 4px 16px rgba(49, 130, 206, 0.3);
    transform: translateY(-1px);
  `}
  
  ${props => props.$isActive && `
    box-shadow: 0 4px 16px rgba(56, 161, 105, 0.3);
    transform: translateY(-1px);
  `}
`;

const GroupProfileImage = styled.img`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  object-fit: cover;
  border: 3px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  transition: all 0.2s ease;
  cursor: pointer;
  
  &:hover {
    transform: scale(1.15);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    border-color: ${props => props.darkMode ? '#3182ce' : '#3182ce'};
  }
`;

const GroupProfilePlaceholder = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  color: ${props => props.darkMode ? '#a0aec0' : '#718096'};
  border: 3px solid ${props => props.darkMode ? '#718096' : '#cbd5e0'};
  transition: all 0.2s ease;
  cursor: pointer;
  
  &:hover {
    background: ${props => props.darkMode ? '#3182ce' : '#3182ce'};
    color: white;
    transform: scale(1.15);
    box-shadow: 0 6px 16px rgba(49, 130, 206, 0.4);
    border-color: ${props => props.darkMode ? '#3182ce' : '#3182ce'};
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
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  padding: 40px;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  min-width: 450px;
  max-width: 550px;
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  transition: all 0.3s ease;
`;

const ModalTitle = styled.h3`
  margin: 0 0 24px 0;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 1.5rem;
  font-weight: 600;
  transition: color 0.3s ease;
`;

const ModalInput = styled.input`
  width: 100%;
  padding: 16px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
  border-radius: 12px;
  font-size: 1rem;
  margin-bottom: 24px;
  background: ${props => props.darkMode ? '#4a5568' : '#ffffff'};
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  transition: all 0.3s ease;
  
  &::placeholder {
    color: ${props => props.darkMode ? '#a0aec0' : '#a0aec0'};
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
  background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  border-radius: 16px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
  display: ${props => props.$show ? 'block' : 'none'};
  animation: fadeIn 0.5s ease-out;
  transition: all 0.3s ease;
`;

const BulkActionTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-weight: 600;
  transition: color 0.3s ease;
`;

const BulkActionDescription = styled.p`
  margin: 0 0 16px 0;
  font-size: 1rem;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  line-height: 1.5;
  transition: color 0.3s ease;
`;

const BulkActionControls = styled.div`
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
`;

const BulkNameInput = styled.input`
  padding: 12px 16px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
  border-radius: 12px;
  font-size: 1rem;
  min-width: 250px;
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  transition: all 0.3s ease;
  
  &::placeholder {
    color: ${props => props.darkMode ? '#a0aec0' : '#a0aec0'};
  }
  
  &:focus {
    outline: none;
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
  }
`;

const GalleryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 24px;
  margin-top: 24px;
`;

const CatCard = styled.div`
  background: ${props => props.darkMode ? '#4a5568' : '#ffffff'};
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  border: 2px solid ${props => {
    if (props.selected) return '#3182ce';
    if (props.$isProfile) return '#38a169';
    return props.darkMode ? '#718096' : '#e2e8f0';
  }};
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  position: relative;
  overflow: hidden;

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    border-color: ${props => {
      if (props.selected) return '#3182ce';
      if (props.$isProfile) return '#2f855a';
      return '#3182ce';
    }};
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(49, 130, 206, 0.1), transparent);
    transition: left 0.6s;
  }

  &:hover::before {
    left: 100%;
  }
  
  ${props => props.isProfile && `
    background: ${props.darkMode ? 'rgba(56, 161, 105, 0.1)' : 'rgba(56, 161, 105, 0.05)'};
    box-shadow: 0 4px 20px rgba(56, 161, 105, 0.2);
  `}
`;

const CatImage = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 12px;
  margin-bottom: 16px;
  transition: transform 0.3s ease;

  &:hover {
    transform: scale(1.05);
  }
`;

const CatInfo = styled.div`
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  transition: color 0.3s ease;
`;

const CatTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-weight: 600;
  padding: 8px 12px;
  background: ${props => props.$groupColor || 'rgba(102, 126, 234, 0.1)'};
  border-radius: 8px;
  border-left: 4px solid ${props => props.$groupColor || 'rgba(102, 126, 234, 0.8)'};
  display: inline-block;
`;

const CatDetails = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 0.9rem;
  color: ${props => props.darkMode ? '#a0aec0' : '#718096'};
  transition: color 0.3s ease;
`;

const CatDetailItem = styled.span`
  font-size: 0.9rem;
  color: ${props => props.darkMode ? '#a0aec0' : '#718096'};
  transition: color 0.3s ease;
`;

const ConfidenceBadge = styled.span`
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 600;
  background: ${props => {
    if (props.level === 'high') return 'rgba(56, 161, 105, 0.2)';
    if (props.level === 'medium') return 'rgba(214, 158, 46, 0.2)';
    return 'rgba(229, 62, 62, 0.2)';
  }};
  color: ${props => {
    if (props.level === 'high') return '#38a169';
    if (props.level === 'medium') return '#d69e2e';
    return '#e53e3e';
  }};
`;

const CatName = styled.h3`
  margin: 0 0 8px 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  transition: color 0.3s ease;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 60px 20px;
  color: ${props => props.darkMode ? '#a0aec0' : '#718096'};
  transition: color 0.3s ease;
`;

const EmptyIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 16px;
  opacity: 0.5;
`;

const EmptyText = styled.p`
  margin: 0;
  font-size: 1.2rem;
  font-weight: 500;
`;

// 상태 표시줄 컴포넌트
const StatusBar = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  border: 2px solid ${props => props.type === 'success' ? '#38a169' : props.type === 'error' ? '#e53e3e' : '#3182ce'};
  border-radius: 12px;
  padding: 16px 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  z-index: 2000;
  min-width: 300px;
  max-width: 400px;
  animation: slideInRight 0.3s ease-out;
  
  @keyframes slideInRight {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
`;

const StatusHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
`;

const StatusIcon = styled.div`
  font-size: 1.2rem;
  color: ${props => props.type === 'success' ? '#38a169' : props.type === 'error' ? '#e53e3e' : '#3182ce'};
`;

const StatusTitle = styled.h4`
  margin: 0;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 1rem;
  font-weight: 600;
  transition: color 0.3s ease;
`;

const StatusBarMessage = styled.div`
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 0.9rem;
  line-height: 1.4;
  margin-bottom: 12px;
  transition: color 0.3s ease;
`;

const StatusProgress = styled.div`
  width: 100%;
  height: 6px;
  background: ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 8px;
  transition: background 0.3s ease;
`;

const StatusProgressBar = styled.div`
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #4facfe 100%);
  border-radius: 3px;
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
`;

const StatusCloseButton = styled.button`
  position: absolute;
  top: 8px;
  right: 8px;
  background: none;
  border: none;
  color: ${props => props.darkMode ? '#a0aec0' : '#718096'};
  font-size: 1.2rem;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
    color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  }
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
  animation: fadeIn 0.3s ease-out;
`;

const TeachingContent = styled.div`
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  padding: 60px 40px;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  text-align: center;
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  max-width: 500px;
  width: 90%;
  transition: all 0.3s ease;
`;

const TeachingIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  animation: pulse 1.5s infinite;
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
  }
`;

const TeachingTitle = styled.h2`
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 16px;
  transition: color 0.3s ease;
`;

const TeachingDescription = styled.p`
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 1.1rem;
  line-height: 1.6;
  margin-bottom: 24px;
  transition: color 0.3s ease;
`;

const TeachingProgress = styled.div`
  width: 100%;
  height: 8px;
  background: ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 20px;
  transition: background 0.3s ease;
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
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 1rem;
  font-weight: 500;
  transition: color 0.3s ease;
`;

const GallerySection = styled.div`
  margin-top: 24px;
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  border-radius: 16px;
  overflow: hidden;
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  transition: all 0.3s ease;
  ${props => props.$isExpanded && `
    box-shadow: 0 8px 32px rgba(49, 130, 206, 0.15);
    border-color: #3182ce;
  `}
  ${props => props.$highlighted && `
    animation: pulse 0.4s ease-in-out;
    box-shadow: 0 8px 32px rgba(49, 130, 206, 0.25);
    border-color: #3182ce;
  `}
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.01); }
    100% { transform: scale(1); }
  }
`;

const GalleryHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  border-bottom: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => {
      if (props.$isExpanded) {
        return props.darkMode ? '#2c5aa0' : '#2c5aa0';
      }
      if (props.$isActive) {
        return props.darkMode ? '#2f855a' : '#2f855a';
      }
      return props.darkMode ? '#718096' : '#edf2f7';
    }};
  }
  
  ${props => props.$isExpanded && `
    background: ${props.darkMode ? '#3182ce' : '#3182ce'};
  `}
  
  ${props => props.$isActive && `
    background: ${props.darkMode ? '#38a169' : '#38a169'};
  `}
`;

const GalleryTitle = styled.h3`
  margin: 0;
  font-size: 1.1rem;
  color: ${props => {
    if (props.$isExpanded || props.$isActive) {
      return 'white';
    }
    return props.darkMode ? '#e2e8f0' : '#2d3748';
  }};
  font-weight: 600;
  transition: color 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  
  /* 확장되거나 활성화된 상태에서는 항상 흰색 */
  ${props => (props.$isExpanded || props.$isActive) && `
    color: white !important;
  `}
`;

const GalleryToggle = styled.button`
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: ${props => {
    if (props.$isExpanded || props.$isActive) {
      return 'white';
    }
    return props.darkMode ? '#a0aec0' : '#4a5568';
  }};
  transition: transform 0.3s ease;
  
  &:hover {
    color: ${props => {
      if (props.$isExpanded || props.$isActive) {
        return 'white';
      }
      return props.darkMode ? '#e2e8f0' : '#2d3748';
    }};
  }
  
  /* 확장되거나 활성화된 상태에서는 항상 흰색 */
  ${props => (props.$isExpanded || props.$isActive) && `
    color: white !important;
    
    &:hover {
      color: white !important;
    }
  `}
`;

const GalleryContent = styled.div`
  padding: 24px;
  display: ${props => props.$collapsed ? 'none' : 'block'};
  animation: ${props => props.$collapsed ? 'none' : 'slideDown 0.2s ease-out'};
  
  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-5px);
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
  background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  border-radius: 16px;
  border: 1px solid ${props => props.darkMode ? '#718096' : '#e2e8f0'};
  transition: all 0.3s ease;
`;

const FilterTitle = styled.h3`
  margin: 0 0 12px 0;
  font-size: 1.1rem;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-weight: 600;
  transition: color 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const FilterDescription = styled.p`
  margin: 0 0 16px 0;
  font-size: 0.9rem;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  line-height: 1.4;
  transition: color 0.3s ease;
`;

const ServerInfo = styled.div`
  margin: 16px 0;
  padding: 12px 16px;
  background: ${props => props.darkMode ? 'rgba(56, 161, 105, 0.1)' : 'rgba(56, 161, 105, 0.05)'};
  border: 1px solid ${props => props.darkMode ? 'rgba(56, 161, 105, 0.3)' : 'rgba(56, 161, 105, 0.2)'};
  border-radius: 8px;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 0.9rem;
  line-height: 1.4;
  transition: all 0.3s ease;
`;

const LearningTip = styled.div`
  margin: 12px 0;
  padding: 10px 14px;
  background: ${props => props.darkMode ? 'rgba(79, 172, 254, 0.1)' : 'rgba(79, 172, 254, 0.05)'};
  border: 1px solid ${props => props.darkMode ? 'rgba(79, 172, 254, 0.3)' : 'rgba(79, 172, 254, 0.2)'};
  border-radius: 8px;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-size: 0.85rem;
  line-height: 1.4;
  transition: all 0.3s ease;
`;

const QuickAddButton = styled.button`
  background: #38a169;
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  font-size: 14px;
  font-weight: bold;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 8px;
  transition: all 0.2s ease;
  
  &:hover {
    background: #2f855a;
    transform: scale(1.1);
  }
  
  &:active {
    transform: scale(0.95);
  }
`;

const ProfileButton = styled.button`
  background: ${props => props.className === 'active' ? '#4facfe' : '#667eea'};
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  font-size: 12px;
  font-weight: bold;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 4px;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.className === 'active' ? '#3b82f6' : '#5a6fd8'};
    transform: scale(1.1);
  }
  
  &:active {
    transform: scale(0.95);
  }
  
  &.active {
    background: #4facfe;
    box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.3);
  }
`;

const ProfileModal = styled.div`
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
  animation: fadeIn 0.3s ease-out;
`;

const ProfileModalContent = styled.div`
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  padding: 40px;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  min-width: 500px;
  max-width: 600px;
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  transition: all 0.3s ease;
`;

const ProfileModalTitle = styled.h3`
  margin: 0 0 24px 0;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 1.5rem;
  font-weight: 600;
  transition: color 0.3s ease;
`;

const ProfileGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
  max-height: 400px;
  overflow-y: auto;
`;

const ProfileImageCard = styled.div`
  background: ${props => props.darkMode ? '#4a5568' : '#ffffff'};
  border-radius: 12px;
  padding: 12px;
  border: 2px solid ${props => props.selected ? '#667eea' : props.darkMode ? '#718096' : '#e2e8f0'};
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: center;
  
  &:hover {
    border-color: #667eea;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
  }
  
  ${props => props.selected && `
    border-color: #667eea;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
  `}
`;

const ProfileImage = styled.img`
  width: 100%;
  height: 80px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 8px;
`;

const ProfileImageName = styled.div`
  font-size: 0.8rem;
  color: ${props => props.darkMode ? '#a0aec0' : '#4a5568'};
  font-weight: 500;
  transition: color 0.3s ease;
`;

const ProfileBadge = styled.div`
  position: absolute;
  top: 8px;
  right: 8px;
  background: #38a169;
  color: white;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  box-shadow: 0 2px 8px rgba(56, 161, 105, 0.3);
  z-index: 10;
`;

const ProfileModalButtons = styled.div`
  display: flex;
  gap: 12px;
  justify-content: flex-end;
`;

// 컨텍스트 메뉴 스타일 추가
const ContextMenu = styled.div`
  position: fixed;
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  padding: 8px 0;
  z-index: 3000;
  min-width: 200px;
  animation: fadeIn 0.2s ease-out;
  backdrop-filter: blur(10px);
`;

const ContextMenuItem = styled.div`
  padding: 12px 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 12px;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 0.9rem;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  }
  
  &:active {
    background: ${props => props.darkMode ? '#718096' : '#edf2f7'};
  }
  
  ${props => props.disabled && `
    opacity: 0.5;
    cursor: not-allowed;
    
    &:hover {
      background: none;
    }
  `}
`;

const ContextMenuDivider = styled.div`
  height: 1px;
  background: ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  margin: 4px 0;
`;

// 서브메뉴 스타일 추가
const SubMenu = styled.div`
  position: absolute;
  left: 100%;
  top: 0;
  background: ${props => props.darkMode ? '#2d3748' : '#ffffff'};
  border: 1px solid ${props => props.darkMode ? '#4a5568' : '#e2e8f0'};
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  padding: 8px 0;
  min-width: 180px;
  animation: fadeIn 0.2s ease-out;
  backdrop-filter: blur(10px);
  z-index: 3001;
  
  /* 화면 오른쪽 경계 체크 */
  @media (min-width: 1200px) {
    left: 100%;
  }
  
  @media (max-width: 1199px) {
    left: auto;
    right: 100%;
  }
`;

const SubMenuItem = styled.div`
  padding: 10px 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 0.85rem;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  }
  
  ${props => props.$isProfile && `
    color: #38a169;
    font-weight: 600;
  `}
`;

const ContextMenuItemWithSubmenu = styled.div`
  padding: 12px 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: ${props => props.darkMode ? '#e2e8f0' : '#2d3748'};
  font-size: 0.9rem;
  transition: all 0.2s ease;
  position: relative;
  
  &:hover {
    background: ${props => props.darkMode ? '#4a5568' : '#f7fafc'};
  }
  
  &:hover .submenu {
    display: block !important;
  }
  
  .submenu {
    display: none;
  }
`;

function CatGallery({ 
  croppedCats, 
  onBack, 
  onReset, 
  uploadSummary, 
  isLoading: isGalleryLoading, 
  onRefresh, 
  savedGroups, 
  onShowGlobalMessage,
  darkMode = false,
  activeTab = 'upload'
}) {
  const [selectedCats, setSelectedCats] = useState(new Set());
  const [catNames, setCatNames] = useState({});
  const [statusMessage, setStatusMessage] = useState(null);
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
  const [isGalleryHighlighted, setIsGalleryHighlighted] = useState(false);
  
  // Shift 키 연결 선택을 위한 상태 추가
  const [lastSelectedCat, setLastSelectedCat] = useState(null);
  const [isShiftPressed, setIsShiftPressed] = useState(false);

  // 프로필 관련 상태 추가
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [selectedProfileGroup, setSelectedProfileGroup] = useState('');
  const [selectedProfileImage, setSelectedProfileImage] = useState('');
  const [catProfiles, setCatProfiles] = useState({});

  // 컨텍스트 메뉴 관련 상태 추가
  const [contextMenu, setContextMenu] = useState({
    show: false,
    x: 0,
    y: 0,
    catId: null,
    catName: null
  });

  // 상태 표시줄 관련 상태 추가
  const [statusBar, setStatusBar] = useState({
    show: false,
    type: 'info', // 'info', 'success', 'error'
    title: '',
    message: '',
    progress: 0,
    checkpointInfo: null
  });

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

    // 파일명 기반 색상 생성 함수들 (연한 색상)
  const generateFilenameColor = (filename) => {
    if (!filename) return {
      background: 'rgba(156, 163, 175, 0.1)',
      border: 'rgba(156, 163, 175, 0.3)',
      text: 'rgba(156, 163, 175, 0.8)'
    };
    
    // 파일명에서 확장자 제거
    const nameWithoutExt = filename.replace(/\.[^/.]+$/, "");
    
    const { hue, saturation, lightness } = generateColorFromString(nameWithoutExt);
    
    // 연한 색상으로 조정 (높은 lightness, 낮은 saturation)
    return {
      background: `hsla(${hue}, ${Math.max(20, saturation - 30)}%, ${Math.min(95, lightness + 30)}%, 0.15)`,
      border: `1px solid hsla(${hue}, ${Math.max(30, saturation - 20)}%, ${Math.min(85, lightness + 20)}%, 0.6)`,
      color: `hsla(${hue}, ${Math.max(40, saturation - 10)}%, ${Math.max(30, lightness - 15)}%, 0.9)`
    };
  };



  // 저장된 그룹 정보가 변경될 때마다 catNames 업데이트
  useEffect(() => {
    if (savedGroups && Object.keys(savedGroups).length > 0) {
      console.log('저장된 그룹 정보 적용:', savedGroups);
      setCatNames(savedGroups);
    }
  }, [savedGroups]);

  // 저장된 프로필 정보 로드
  useEffect(() => {
    const loadSavedProfiles = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/cat-groups');
        const data = await response.json();
        
        if (data.success && data.profiles) {
          console.log('저장된 프로필 정보 로드:', data.profiles);
          setCatProfiles(data.profiles);
        }
      } catch (error) {
        console.error('저장된 프로필 정보 로드 실패:', error);
      }
    };
    
    loadSavedProfiles();
  }, []);

  // 컴포넌트 마운트 시 백엔드 연결 테스트
  // 백엔드 연결 테스트를 한 번만 실행하기 위한 ref
  const hasTestedBackend = React.useRef(false);

  useEffect(() => {
    // 갤러리 탭이 활성화되었고 아직 백엔드 테스트를 하지 않았을 때만 실행
    if (activeTab === 'gallery' && !hasTestedBackend.current) {
      hasTestedBackend.current = true;
      
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
    }
  }, [activeTab]);

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
      
      if (data.success) {
        // 그룹 정보 로드
        if (data.groups) {
          setCatNames(data.groups);
        }
        
        // 프로필 정보 로드
        if (data.profiles) {
          setCatProfiles(data.profiles);
        }
        
        const groupCount = data.groups ? Object.keys(data.groups).length : 0;
        const profileCount = data.profiles ? Object.keys(data.profiles).length : 0;
        
        let message = `저장된 그룹 정보를 로드했습니다.`;
        message += `\n📊 통계: ${groupCount}개의 고양이 이름, ${profileCount}개의 프로필 이미지`;
        
        if (profileCount > 0 && data.profiles) {
          const groupsWithProfiles = Object.keys(data.profiles);
          message += `\n👤 프로필 설정된 그룹: ${groupsWithProfiles.join(', ')}`;
        }
        
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'success');
        } else {
          setStatusMessage({ type: 'success', text: message });
          setTimeout(() => setStatusMessage(null), 5000);
        }
      }
    } catch (error) {
      console.error('그룹 정보 로드 실패:', error);
      const message = '저장된 그룹 정보 로드에 실패했습니다.';
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'error');
      } else {
        setStatusMessage({ type: 'error', text: message });
        setTimeout(() => setStatusMessage(null), 3000);
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
      console.log('저장할 프로필 정보:', catProfiles);
      console.log('그룹 정보 타입:', typeof catNames);
      console.log('그룹 정보 키:', Object.keys(catNames));
      
      // 프로필 정보 상세 로그
      if (Object.keys(catProfiles).length > 0) {
        console.log('프로필 설정된 그룹들:', Object.keys(catProfiles));
        Object.entries(catProfiles).forEach(([groupName, profileFilename]) => {
          console.log(`그룹 "${groupName}"의 프로필: ${profileFilename}`);
        });
      } else {
        console.log('설정된 프로필이 없습니다.');
      }
      
      // 빈 객체가 아닌지 확인
      if (Object.keys(catNames).length === 0) {
        const message = '저장할 그룹 정보가 없습니다.';
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'info');
        } else {
          setStatusMessage({ type: 'info', text: message });
          setTimeout(() => setStatusMessage(null), 3000);
        }
        return;
      }
      
      // 그룹 정보와 프로필 정보를 함께 전송
      const requestData = {
        groups: catNames,
        profiles: catProfiles,
        metadata: {
          total_groups: Object.keys(catNames).length,
          total_profiles: Object.keys(catProfiles).length,
          groups_with_profiles: Object.keys(catProfiles),
          timestamp: new Date().toISOString()
        }
      };
      
      const requestBody = JSON.stringify(requestData);
      console.log('요청 URL:', 'http://localhost:5000/api/cat-groups');
      console.log('요청 메서드:', 'POST');
      console.log('요청 본문:', requestBody);
      console.log('전송 데이터 구조:', {
        groups_count: Object.keys(catNames).length,
        profiles_count: Object.keys(catProfiles).length,
        groups_with_profiles: Object.keys(catProfiles),
        metadata: requestData.metadata
      });
      
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
        const groupCount = Object.keys(catNames).length;
        const profileCount = Object.keys(catProfiles).length;
        const groupsWithProfiles = Object.keys(catProfiles);
        
        let message = `그룹 정보가 성공적으로 저장되었습니다!`;
        message += `\n📊 통계: ${groupCount}개의 고양이 이름, ${profileCount}개의 프로필 이미지`;
        
        if (profileCount > 0) {
          message += `\n👤 프로필 설정된 그룹: ${groupsWithProfiles.join(', ')}`;
        }
        
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'success');
        } else {
          setStatusMessage({ type: 'success', text: message });
          setTimeout(() => setStatusMessage(null), 5000);
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
        setTimeout(() => setStatusMessage(null), 5000);
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

  // 현재 선택된 그룹의 고양이들 필터링 (프로필 이미지 우선 정렬)
  const getFilteredCats = () => {
    if (!croppedCats || croppedCats.length === 0) {
      return [];
    }
    
    let filteredCats = [];
    
    if (selectedGroup === 'all') {
      filteredCats = croppedCats;
    } else if (selectedGroup === 'unnamed') {
      filteredCats = croppedCats.filter(cat => !cat || !catNames[cat.id]);
    } else {
      filteredCats = croppedCats.filter(cat => cat && catNames[cat.id] === selectedGroup);
    }
    
    // 특정 그룹 선택 시에만 프로필 이미지를 맨 앞으로 정렬
    if (selectedGroup !== 'all' && selectedGroup !== 'unnamed' && catProfiles[selectedGroup]) {
      const profileFilename = catProfiles[selectedGroup];
      const profileCat = filteredCats.find(cat => cat.filename === profileFilename);
      
      if (profileCat) {
        // 프로필 이미지를 제외한 나머지 고양이들
        const otherCats = filteredCats.filter(cat => cat.filename !== profileFilename);
        // 프로필 이미지를 맨 앞에 배치
        return [profileCat, ...otherCats];
      }
    }
    
    return filteredCats;
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
          setStatusMessage(null);
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
          setStatusMessage(null);
        }, 3000);
      }
    }
  };

  const handleGroupSelect = (groupName) => {
    // 같은 그룹을 다시 클릭한 경우 갤러리 토글
    if (selectedGroup === groupName) {
      setIsGalleryCollapsed(!isGalleryCollapsed);
      
      const message = isGalleryCollapsed ? 
        '갤러리를 펼쳤습니다.' : 
        '갤러리를 접었습니다.';
      
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'info');
      } else {
        setStatusMessage({ type: 'info', text: message });
        setTimeout(() => setStatusMessage(null), 2000);
      }
      return;
    }
    
    // 다른 그룹을 선택한 경우
    setSelectedGroup(groupName);
    setSelectedCats(new Set()); // 그룹 변경 시 선택 해제
    
    // 그룹 선택 시 갤러리가 접혀있다면 자동으로 펼치기
    if (isGalleryCollapsed) {
      setIsGalleryCollapsed(false);
      
      // 갤러리 펼침 효과를 위한 하이라이트
      setIsGalleryHighlighted(true);
      setTimeout(() => setIsGalleryHighlighted(false), 1000);
    }
    
    // 선택된 그룹에 대한 시각적 피드백
    const message = groupName === 'all' ? '전체 고양이를 표시합니다.' : 
                   groupName === 'unnamed' ? '미지정 고양이를 표시합니다.' :
                   `"${groupName}" 그룹의 고양이를 표시합니다.`;
    
    if (onShowGlobalMessage) {
      onShowGlobalMessage(message, 'info');
    } else {
      setStatusMessage({ type: 'info', text: message });
      setTimeout(() => setStatusMessage(null), 2000);
    }
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
    
    // 그룹 선택 시 갤러리가 접혀있다면 자동으로 펼치기
    if (isGalleryCollapsed) {
      setIsGalleryCollapsed(false);
      
      // 갤러리 펼침 효과를 위한 하이라이트
      setIsGalleryHighlighted(true);
      setTimeout(() => setIsGalleryHighlighted(false), 1000);
    }
    
    const message = `${groupDisplayName} 그룹의 ${catIds.length}마리를 선택했습니다.`;
    if (onShowGlobalMessage) {
      onShowGlobalMessage(message, 'info');
    } else {
      setStatusMessage({ type: 'info', text: message });
      setTimeout(() => {
        setStatusMessage(null);
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
        setStatusMessage(null);
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
        setTimeout(() => setStatusMessage(null), 3000);
      }
      return;
    }

    // 그룹별 이미지 개수 확인
    const groupCounts = {};
    croppedCats.forEach(cat => {
      const name = catNames[cat.id];
      if (name && name.trim()) {
        groupCounts[name] = (groupCounts[name] || 0) + 1;
      }
    });

    // 3개 미만인 그룹들 확인
    const insufficientGroups = Object.entries(groupCounts)
      .filter(([groupName, count]) => count < 3)
      .map(([groupName, count]) => ({ name: groupName, count }));

    if (insufficientGroups.length > 0) {
      const groupList = insufficientGroups
        .map(group => `"${group.name}" (${group.count}개)`)
        .join(', ');
      
      const message = `다음 그룹들의 이미지가 3개 미만입니다: ${groupList}\n\n더 많은 이미지를 업로드하거나 다른 그룹의 이미지를 추가해주세요.`;
      
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'error');
      } else {
        setStatusMessage({ type: 'error', text: message });
        setTimeout(() => setStatusMessage(null), 5000);
      }
      return;
    }

    try {
      // 상태 표시줄 표시
      setStatusBar({
        show: true,
        type: 'info',
        title: '🧠 모델 학습 시작',
        message: '데이터 준비 중...',
        progress: 0,
        checkpointInfo: null
      });
      
      // 학습 데이터 준비 - 이름이 지정된 고양이들만
      const namedCatIds = croppedCats
        .filter(cat => catNames[cat.id] && catNames[cat.id].trim())
        .map(cat => cat.id);
      
      const teachingData = {
        selected_cat_ids: namedCatIds,
        cat_names: catNames
      };
      
      console.log('=== 실제 모델 학습 시작 ===');
      console.log('학습 데이터:', teachingData);
      
      // 진행 상황 업데이트
      setStatusBar(prev => ({
        ...prev,
        message: '고양이 이미지 분석 중...',
        progress: 20
      }));
      
      const response = await fetch('http://localhost:5000/api/yolo/teach-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(teachingData),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setStatusBar(prev => ({
          ...prev,
          type: 'success',
          title: '✅ 학습 완료',
          message: data.message,
          progress: 100,
          checkpointInfo: data.checkpoint_info
        }));
        
        // 체크포인트 다운로드 정보가 있으면 다운로드 시작
        if (data.checkpoint_info && data.checkpoint_info.success) {
          setTimeout(() => {
            downloadCheckpoint(data.checkpoint_info);
          }, 2000);
        }
      } else {
        setStatusBar(prev => ({
          ...prev,
          type: 'error',
          title: '❌ 학습 실패',
          message: data.message || '모델 학습에 실패했습니다.',
          progress: 0
        }));
      }
    } catch (error) {
      console.error('모델 학습 실패:', error);
      setStatusBar(prev => ({
        ...prev,
        type: 'error',
        title: '❌ 오류 발생',
        message: '모델 학습 중 오류가 발생했습니다.',
        progress: 0
      }));
    }
  };

  // 체크포인트 다운로드 함수
  const downloadCheckpoint = async (checkpointInfo) => {
    try {
      setStatusBar(prev => ({
        ...prev,
        message: '체크포인트 파일 다운로드 중...',
        progress: 90
      }));
      
      const response = await fetch(`http://localhost:5000${checkpointInfo.download_url}`);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = checkpointInfo.checkpoint_file;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        setStatusBar(prev => ({
          ...prev,
          message: '체크포인트 파일 다운로드 완료!',
          progress: 100
        }));
      } else {
        throw new Error('체크포인트 다운로드 실패');
      }
    } catch (error) {
      console.error('체크포인트 다운로드 실패:', error);
      setStatusBar(prev => ({
        ...prev,
        type: 'error',
        title: '❌ 다운로드 실패',
        message: '체크포인트 파일 다운로드에 실패했습니다.',
        progress: 0
      }));
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

  // 빠른 그룹 추가 함수
  const handleQuickAddToGroup = (groupName) => {
    if (selectedCats.size === 0) {
      const message = '먼저 추가할 고양이 이미지를 선택해주세요.';
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'info');
      } else {
        setStatusMessage({ type: 'info', text: message });
        setTimeout(() => setStatusMessage(null), 3000);
      }
      return;
    }

    const newCatNames = { ...catNames };
    selectedCats.forEach(catId => {
      newCatNames[catId] = groupName;
    });
    
    setCatNames(newCatNames);
    setSelectedCats(new Set());
    
    const message = `${selectedCats.size}마리의 고양이를 "${groupName}" 그룹에 추가했습니다!`;
    if (onShowGlobalMessage) {
      onShowGlobalMessage(message, 'success');
    } else {
      setStatusMessage({ type: 'success', text: message });
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  // 프로필 설정 관련 함수들
  const handleOpenProfileModal = (groupName) => {
    setSelectedProfileGroup(groupName);
    setSelectedProfileImage('');
    setShowProfileModal(true);
  };

  const handleProfileImageSelect = (catId) => {
    setSelectedProfileImage(catId);
  };

  const handleSetProfile = () => {
    if (selectedProfileImage) {
      // 선택된 고양이의 파일명 가져오기
      const profileCat = croppedCats.find(cat => cat.id === selectedProfileImage);
      const profileFilename = profileCat ? profileCat.filename : selectedProfileImage;
      
      const newCatProfiles = { ...catProfiles };
      newCatProfiles[selectedProfileGroup] = profileFilename;
      setCatProfiles(newCatProfiles);
      
      const message = `"${selectedProfileGroup}" 그룹의 프로필 이미지가 설정되었습니다!`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'success');
      } else {
        setStatusMessage({ type: 'success', text: message });
        setTimeout(() => setStatusMessage(null), 3000);
      }
      
      setShowProfileModal(false);
    }
  };

  const handleRemoveProfile = () => {
    const newCatProfiles = { ...catProfiles };
    delete newCatProfiles[selectedProfileGroup];
    setCatProfiles(newCatProfiles);
    
    const message = `"${selectedProfileGroup}" 그룹의 프로필 이미지가 제거되었습니다.`;
    if (onShowGlobalMessage) {
      onShowGlobalMessage(message, 'info');
    } else {
      setStatusMessage({ type: 'info', text: message });
      setTimeout(() => setStatusMessage(null), 3000);
    }
    
    setShowProfileModal(false);
  };

  // 그룹별 이미지 가져오기
  const getGroupImages = (groupName) => {
    if (groupName === 'all' || groupName === 'unnamed') {
      return [];
    }
    
    return croppedCats.filter(cat => catNames[cat.id] === groupName);
  };

  // 프로필 이미지 URL 가져오기
  const getProfileImageUrl = (groupName) => {
    if (!catProfiles[groupName]) return null;
    
    // catProfiles에는 파일명이 저장되어 있음
    const profileFilename = catProfiles[groupName];
    
    // 파일명으로 URL 생성
    return `http://localhost:5000/cropped-images/${profileFilename}`;
  };

  // 컨텍스트 메뉴 관련 함수들
  const handleContextMenu = (e, catId, catName) => {
    e.preventDefault();
    e.stopPropagation();
    
    // 화면 크기 고려하여 위치 조정
    const menuWidth = 200;
    const menuHeight = 300; // 예상 높이
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    
    let x = e.clientX;
    let y = e.clientY;
    
    // 오른쪽 경계 체크
    if (x + menuWidth > screenWidth) {
      x = screenWidth - menuWidth - 10;
    }
    
    // 아래쪽 경계 체크
    if (y + menuHeight > screenHeight) {
      y = screenHeight - menuHeight - 10;
    }
    
    setContextMenu({
      show: true,
      x: x,
      y: y,
      catId: catId,
      catName: catName
    });
  };

  const hideContextMenu = () => {
    setContextMenu({
      show: false,
      x: 0,
      y: 0,
      catId: null,
      catName: null
    });
  };

    const handleContextMenuAction = (action) => {
    const { catId, catName } = contextMenu;
    
    switch (action) {
      case 'set-profile':
        if (catName && catName !== '미지정') {
          setSelectedProfileGroup(catName);
          setSelectedProfileImage(catId);
          setShowProfileModal(true);
        } else {
          const message = '미지정 고양이는 프로필로 설정할 수 없습니다. 먼저 그룹에 추가해주세요.';
          if (onShowGlobalMessage) {
            onShowGlobalMessage(message, 'warning');
          } else {
            setStatusMessage({ type: 'warning', text: message });
            setTimeout(() => setStatusMessage(null), 3000);
          }
        }
        break;
        
      case 'add-to-group':
        if (catId) {
          setSelectedCats(new Set([catId]));
          setBulkNameInput(catName || '');
          
          // 갤러리가 접혀있다면 펼치기
          if (isGalleryCollapsed) {
            setIsGalleryCollapsed(false);
          }
          
          const message = '고양이를 선택했습니다. 위의 이름 입력란에서 그룹명을 입력해주세요.';
          if (onShowGlobalMessage) {
            onShowGlobalMessage(message, 'info');
          } else {
            setStatusMessage({ type: 'info', text: message });
            setTimeout(() => setStatusMessage(null), 3000);
          }
        }
        break;
        
      case 'remove-from-group':
        if (catId && catName && catName !== '미지정') {
          const newCatNames = { ...catNames };
          delete newCatNames[catId];
          setCatNames(newCatNames);
          
          const message = `"${catName}" 그룹에서 제거되었습니다.`;
          if (onShowGlobalMessage) {
            onShowGlobalMessage(message, 'info');
          } else {
            setStatusMessage({ type: 'info', text: message });
            setTimeout(() => setStatusMessage(null), 3000);
          }
        }
        break;
        
      case 'select-similar':
        if (catName && catName !== '미지정') {
          const similarCats = croppedCats.filter(cat => catNames[cat.id] === catName);
          setSelectedCats(new Set(similarCats.map(cat => cat.id)));
          
          const message = `"${catName}" 그룹의 ${similarCats.length}마리를 선택했습니다.`;
          if (onShowGlobalMessage) {
            onShowGlobalMessage(message, 'info');
          } else {
            setStatusMessage({ type: 'info', text: message });
            setTimeout(() => setStatusMessage(null), 3000);
          }
        }
        break;
        
      case 'copy-filename':
        if (catId) {
          const cat = croppedCats.find(c => c.id === catId);
          if (cat && cat.filename) {
            navigator.clipboard.writeText(cat.filename);
            
            const message = '파일명이 클립보드에 복사되었습니다.';
            if (onShowGlobalMessage) {
              onShowGlobalMessage(message, 'success');
            } else {
              setStatusMessage({ type: 'success', text: message });
              setTimeout(() => setStatusMessage(null), 2000);
            }
          }
        }
        break;
        
      case 'view-details':
        if (catId) {
          const cat = croppedCats.find(c => c.id === catId);
          if (cat) {
            const details = `파일명: ${cat.filename || '없음'}\n시간: ${formatTime(cat.timestamp || 0)}\n그룹: ${catNames[catId] || '미지정'}\nID: ${catId}`;
            alert(details);
          }
        }
        break;
    }
    
    hideContextMenu();
  };

  // 그룹 추가 함수
  const handleAddToSpecificGroup = (groupName) => {
    const { catId } = contextMenu;
    if (catId) {
      const newCatNames = { ...catNames };
      newCatNames[catId] = groupName;
      setCatNames(newCatNames);
      
      const message = `고양이가 "${groupName}" 그룹에 추가되었습니다!`;
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'success');
      } else {
        setStatusMessage({ type: 'success', text: message });
        setTimeout(() => setStatusMessage(null), 3000);
      }
      
      hideContextMenu();
    }
  };

  // 프로필 설정 함수 (직접 설정)
  const handleSetProfileDirect = () => {
    const { catId, catName } = contextMenu;
    if (catId && catName && catName !== '미지정') {
      const cat = croppedCats.find(c => c.id === catId);
      if (cat && cat.filename) {
        const newCatProfiles = { ...catProfiles };
        newCatProfiles[catName] = cat.filename;
        setCatProfiles(newCatProfiles);
        
        const message = `"${catName}" 그룹의 프로필 이미지가 설정되었습니다!`;
        if (onShowGlobalMessage) {
          onShowGlobalMessage(message, 'success');
        } else {
          setStatusMessage({ type: 'success', text: message });
          setTimeout(() => setStatusMessage(null), 3000);
        }
        
        hideContextMenu();
      }
    } else {
      const message = '미지정 고양이는 프로필로 설정할 수 없습니다. 먼저 그룹에 추가해주세요.';
      if (onShowGlobalMessage) {
        onShowGlobalMessage(message, 'warning');
      } else {
        setStatusMessage({ type: 'warning', text: message });
        setTimeout(() => setStatusMessage(null), 3000);
      }
    }
  };

  // 전역 클릭 이벤트로 컨텍스트 메뉴 숨김
  useEffect(() => {
    const handleGlobalClick = () => {
      hideContextMenu();
    };

    document.addEventListener('click', handleGlobalClick);
    document.addEventListener('contextmenu', handleGlobalClick);

    return () => {
      document.removeEventListener('click', handleGlobalClick);
      document.removeEventListener('contextmenu', handleGlobalClick);
    };
  }, []);

  // 로딩 상태 표시
  if (isGalleryLoading) {
    return (
      <GalleryContainer darkMode={darkMode}>
        <Header darkMode={darkMode}>
          <Title darkMode={darkMode}>🐱 고양이 갤러리</Title>
          <Stats darkMode={darkMode}>로딩 중...</Stats>
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
      <GalleryContainer darkMode={darkMode}>
        <Header darkMode={darkMode}>
          <Title darkMode={darkMode}>🐱 고양이 갤러리</Title>
          <Stats darkMode={darkMode}>감지된 고양이: 0마리</Stats>
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

        <EmptyState darkMode={darkMode}>
          <EmptyIcon>🐾</EmptyIcon>
          <EmptyText>아직 감지된 고양이가 없습니다.</EmptyText>
          <EmptyText>영상을 업로드하고 처리해보세요!</EmptyText>
        </EmptyState>
      </GalleryContainer>
    );
  }

  return (
    <GalleryContainer darkMode={darkMode}>
      <Header darkMode={darkMode}>
        <Title darkMode={darkMode}>🐱 우리집 고양이 알려주기</Title>
        <Stats darkMode={darkMode}>
          총 {filteredCats.length}마리 중 {selectedCats.size}마리 선택됨
          {uploadSummary && (
            <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '5px' }}>
              {uploadSummary.message}
            </div>
          )}
        </Stats>
      </Header>

              <FilterSection darkMode={darkMode}>
          <FilterTitle darkMode={darkMode}>
            🐱 우리 집에 사는 고양이들
            <span style={{ 
              fontSize: '0.8rem', 
              color: darkMode ? '#a0aec0' : '#718096',
              fontWeight: 'normal'
            }}>
              (클릭하면 해당 그룹의 이미지가 갤러리에 표시됩니다)
            </span>
          </FilterTitle>
          <FilterDescription darkMode={darkMode}>
            고양이 그룹을 클릭하면 해당 그룹의 이미지들이 갤러리에 표시됩니다. 
            같은 그룹을 다시 클릭하면 갤러리가 접히거나 펼쳐집니다.
          </FilterDescription>
          <ServerInfo darkMode={darkMode}>
            💾 <strong>서버에 전송해서 고양이들을 구별할 수 있게 소개할게요.</strong>
            <br />
            고양이들에게 이름을 지어주시면 AI가 각 고양이의 특징을 학습하여 
            향후 영상에서 같은 고양이를 자동으로 구별할 수 있도록 도와드립니다.
          </ServerInfo>
          <LearningTip darkMode={darkMode}>
            💡 <strong>학습 팁:</strong> 다양한 영상을 업로드해서 다양한 이미지를 학습시킬수록 성능이 올라갑니다.
            <br />
            각 고양이의 다양한 각도, 표정, 자세를 포함한 영상들을 업로드하면 
            AI가 더 정확하게 고양이들을 구별할 수 있게 됩니다.
            <br />
            <strong>서버 학습을 위해서는 각 그룹당 3개 이상의 이미지가 필요합니다.</strong>
          </LearningTip>
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
              $isActive={selectedGroup === groupKey && !isGalleryCollapsed}
              onClick={() => handleGroupSelect(groupKey)}
              onDoubleClick={() => handleSelectGroup(groupKey)}
              onMouseEnter={() => handleGroupHighlight(groupKey)}
              $borderColor={groupKey === 'unnamed' || groupKey === 'all' ? '#e2e8f0' : generateBorderColor(group.name || groupKey)}
              $backgroundColor={groupKey === 'unnamed' || groupKey === 'all' ? '#ffffff' : generateBackgroundColor(group.name || groupKey)}
              title={`${group.name} 그룹의 ${group.count}마리 고양이 보기${selectedGroup === groupKey ? ' (다시 클릭하면 갤러리 토글)' : ''}`}
            >
              {/* 프로필 이미지 표시 */}
              {groupKey !== 'unnamed' && groupKey !== 'all' && catProfiles[groupKey] && getProfileImageUrl(groupKey) && (
                <GroupProfileImage 
                  src={getProfileImageUrl(groupKey)} 
                  alt={`${group.name} 그룹의 프로필`}
                  darkMode={darkMode}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleOpenProfileModal(groupKey);
                  }}
                  title={`${group.name} 그룹의 프로필 이미지 (클릭하여 변경)`}
                  onError={(e) => {
                    console.error('프로필 이미지 로드 실패:', getProfileImageUrl(groupKey));
                    e.target.style.display = 'none';
                    // 프로필 이미지 로드 실패 시 프로필 제거
                    const newCatProfiles = { ...catProfiles };
                    delete newCatProfiles[groupKey];
                    setCatProfiles(newCatProfiles);
                    
                    const message = `"${groupKey}" 그룹의 프로필 이미지를 찾을 수 없어 제거했습니다.`;
                    if (onShowGlobalMessage) {
                      onShowGlobalMessage(message, 'warning');
                    }
                  }}
                />
              )}
              {groupKey !== 'unnamed' && groupKey !== 'all' && (!catProfiles[groupKey] || !getProfileImageUrl(groupKey)) && (
                <GroupProfilePlaceholder 
                  darkMode={darkMode}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleOpenProfileModal(groupKey);
                  }}
                  title={`${group.name} 그룹의 프로필 이미지 설정`}
                >
                  👤
                </GroupProfilePlaceholder>
              )}
              
              {group.name}
              <GroupCount selected={selectedGroup === groupKey}>
                {group.count}
              </GroupCount>
              {selectedCats.size > 0 && groupKey !== 'unnamed' && groupKey !== 'all' && (
                <QuickAddButton
                  onClick={(e) => {
                    e.stopPropagation();
                    handleQuickAddToGroup(groupKey);
                  }}
                  title={`선택된 ${selectedCats.size}마리를 "${group.name}" 그룹에 추가`}
                >
                  +
                </QuickAddButton>
              )}
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

              <BulkActionSection $show={selectedCats.size > 0} darkMode={darkMode}>
                  <BulkActionTitle darkMode={darkMode}>🐱 선택한 이미지의 고양이 이름은 무엇인가요? ({selectedCats.size}개 이미지)</BulkActionTitle>
          <BulkActionDescription darkMode={darkMode}>
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
              darkMode={darkMode}
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
          title={`현재 그룹 정보와 프로필 정보를 서버에 저장 (${Object.keys(catNames).length}개 그룹, ${Object.keys(catProfiles).length}개 프로필)`}
        >
          {isLoading ? '저장 중...' : ' 그룹 저장'}
        </Button>
        <Button 
          className="info"
          onClick={loadCatGroups}
          disabled={isLoading}
          title="서버에서 저장된 그룹 정보와 프로필 정보를 다시 로드"
        >
          {isLoading ? '로드 중...' : ' 그룹 로드'}
        </Button>
        <Button 
          className="danger"
          onClick={handleTeachModel}
          disabled={isLoading || !croppedCats.some(cat => catNames[cat.id] && catNames[cat.id].trim())}
          title="그룹이 지정된 고양이들을 AI 모델에게 알려주기 (각 그룹당 3개 이상의 이미지 필요)"
        >
          🧠 서버에 알려주기
        </Button>
      </Controls>

      {statusMessage && typeof statusMessage === 'object' && statusMessage.text && !onShowGlobalMessage && (
        <StatusMessage className={statusMessage.type}>
          {statusMessage.text}
        </StatusMessage>
      )}

              <GallerySection 
                darkMode={darkMode} 
                $isExpanded={!isGalleryCollapsed}
                $highlighted={isGalleryHighlighted}
              >
        <GalleryHeader 
          onClick={() => setIsGalleryCollapsed(!isGalleryCollapsed)} 
          darkMode={darkMode}
          $isExpanded={!isGalleryCollapsed}
          $isActive={selectedGroup !== 'all' && !isGalleryCollapsed}
        >
          <GalleryTitle 
            darkMode={darkMode} 
            $isExpanded={!isGalleryCollapsed}
            $isActive={selectedGroup !== 'all' && !isGalleryCollapsed}
          >
            🖼️ 고양이 이미지 갤러리 ({filteredCats.length}마리)
            {selectedGroup !== 'all' && (
              <span style={{ 
                fontSize: '0.9rem', 
                opacity: 0.8,
                marginLeft: '8px'
              }}>
                - {selectedGroup === 'unnamed' ? '미지정' : selectedGroup} 그룹
                {selectedGroup !== 'unnamed' && catProfiles[selectedGroup] && (
                  <span style={{ 
                    color: '#38a169',
                    marginLeft: '4px'
                  }}>
                    👤
                  </span>
                )}
              </span>
            )}
            {selectedGroup !== 'all' && !isGalleryCollapsed && (
              <span style={{ 
                fontSize: '0.8rem', 
                marginLeft: '8px',
                opacity: 0.7 
              }}>
                (활성)
              </span>
            )}
          </GalleryTitle>
          <GalleryToggle 
            darkMode={darkMode} 
            $isExpanded={!isGalleryCollapsed}
            $isActive={selectedGroup !== 'all' && !isGalleryCollapsed}
          >
            {isGalleryCollapsed ? '▼' : '▲'}
          </GalleryToggle>
        </GalleryHeader>
        
        <GalleryContent $collapsed={isGalleryCollapsed}>
          {filteredCats.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              padding: '40px 20px',
              color: darkMode ? '#a0aec0' : '#718096'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '16px', opacity: 0.5 }}>🐾</div>
              <p style={{ fontSize: '1.1rem', margin: '0' }}>
                {selectedGroup === 'all' ? '표시할 고양이가 없습니다.' :
                 selectedGroup === 'unnamed' ? '미지정 고양이가 없습니다.' :
                 `"${selectedGroup}" 그룹에 속한 고양이가 없습니다.`}
              </p>
            </div>
          ) : (
            <GalleryGrid>
              {filteredCats.map((cat, index) => {
                // 전체 선택 시에도 프로필 이미지 표시
                const isProfile = selectedGroup === 'all' ? 
                  // 전체 선택 시: 모든 그룹의 프로필 이미지 확인
                  Object.values(catProfiles).includes(cat.filename) :
                  // 특정 그룹 선택 시: 해당 그룹의 프로필 이미지만 확인
                  selectedGroup !== 'unnamed' && catProfiles[selectedGroup] === cat.filename;
                
                return (
                  <CatCard
                    key={cat.id}
                    selected={selectedCats.has(cat.id)}
                    $highlighted={highlightedGroup && catNames[cat.id] === highlightedGroup}
                    $isProfile={isProfile}
                    onClick={() => handleCatSelect(cat.id)}
                    onContextMenu={(e) => handleContextMenu(e, cat.id, catNames[cat.id])}
                    onMouseDown={(e) => e.preventDefault()} // 드래그 방지
                    darkMode={darkMode}
                    style={{
                      position: 'relative',
                      cursor: isShiftPressed ? 'crosshair' : 'pointer',
                      userSelect: 'none', // 텍스트 선택 방지
                      WebkitUserSelect: 'none',
                      MozUserSelect: 'none',
                      msUserSelect: 'none'
                    }}
                  >
                    {isProfile && (
                      <ProfileBadge title="프로필 이미지">
                        👤
                      </ProfileBadge>
                    )}
                    
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
                    src={cat.url ? `http://localhost:5000${cat.url}` : cat.filename ? `http://localhost:5000/cropped-images/${cat.filename}` : ''}
                    alt={`고양이 ${cat.id}`}
                    draggable={false}
                    onDragStart={(e) => e.preventDefault()}
                    onError={(e) => {
                      console.error('이미지 로드 실패:', cat.url || cat.filename);
                      e.target.style.display = 'none';
                      const noImageDiv = e.target.parentNode.querySelector('.no-image');
                      if (noImageDiv) {
                        noImageDiv.style.display = 'flex';
                      }
                    }}
                    style={{
                      userSelect: 'none',
                      WebkitUserSelect: 'none',
                      MozUserSelect: 'none',
                      msUserSelect: 'none',
                      display: (cat.url || cat.filename) ? 'block' : 'none'
                    }}
                  />
                  <div className="no-image" style={{ 
                    display: (cat.url || cat.filename) ? 'none' : 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '200px',
                    backgroundColor: darkMode ? '#4a5568' : '#f7fafc',
                    borderRadius: '12px',
                    color: darkMode ? '#a0aec0' : '#718096',
                    fontSize: '1rem',
                    border: `1px solid ${darkMode ? '#718096' : '#e2e8f0'}`
                  }}>
                    🐱 이미지 없음
                  </div>
                  
                  <CatInfo
                    darkMode={darkMode}
                    style={{
                      userSelect: 'none',
                      WebkitUserSelect: 'none',
                      MozUserSelect: 'none',
                      msUserSelect: 'none'
                    }}
                  >
                    <CatTitle
                      $groupColor={catNames[cat.id] ? generateBackgroundColor(catNames[cat.id]) : 'rgba(156, 163, 175, 0.1)'}
                      $groupBorderColor={catNames[cat.id] ? generateBorderColor(catNames[cat.id]) : 'rgba(156, 163, 175, 0.8)'}
                      darkMode={darkMode}
                    >
                      {catNames[cat.id] || '미지정'}
                      {isProfile && (
                        <span style={{ 
                          marginLeft: '8px',
                          fontSize: '0.8rem',
                          color: '#38a169',
                          fontWeight: 'bold'
                        }}>
                          👤 프로필
                        </span>
                      )}
                    </CatTitle>
                    <CatDetails darkMode={darkMode} style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                      {cat.timestamp && (
                        <CatDetailItem darkMode={darkMode}>시간: {formatTime(cat.timestamp || 0)}</CatDetailItem>
                      )}
                      <CatDetailItem 
                        darkMode={darkMode}
                        style={{
                          padding: '6px 10px',
                          borderRadius: '8px',
                          fontSize: '0.8rem',
                          fontWeight: '500',
                          display: 'inline-block',
                          marginTop: '4px',
                          ...generateFilenameColor(cat.filename)
                        }}
                      >
                        📄 {cat.filename || '파일명 없음'}
                      </CatDetailItem>
                    </CatDetails>
                  </CatInfo>
                </CatCard>
                );
              })}
            </GalleryGrid>
          )}
        </GalleryContent>
      </GallerySection>

      {showModal && (
        <Modal onClick={() => setShowModal(false)}>
          <ModalContent onClick={(e) => e.stopPropagation()} darkMode={darkMode}>
            <ModalTitle darkMode={darkMode}>
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
                  darkMode={darkMode}
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

      {/* 상태 표시줄 */}
      {statusBar.show && (
        <StatusBar 
          darkMode={darkMode} 
          type={statusBar.type}
        >
          <StatusCloseButton 
            darkMode={darkMode}
            onClick={() => setStatusBar(prev => ({ ...prev, show: false }))}
          >
            ×
          </StatusCloseButton>
          
          <StatusHeader>
            <StatusIcon type={statusBar.type}>
              {statusBar.type === 'success' ? '✅' : 
               statusBar.type === 'error' ? '❌' : '🔄'}
            </StatusIcon>
            <StatusTitle darkMode={darkMode}>
              {statusBar.title}
            </StatusTitle>
          </StatusHeader>
          
          <StatusBarMessage darkMode={darkMode}>
            {statusBar.message}
          </StatusBarMessage>
          
          {statusBar.progress > 0 && (
            <StatusProgress darkMode={darkMode}>
              <StatusProgressBar 
                progress={statusBar.progress}
              />
            </StatusProgress>
          )}
          
          {statusBar.checkpointInfo && statusBar.checkpointInfo.success && (
            <div style={{ 
              fontSize: '0.8rem', 
              color: darkMode ? '#a0aec0' : '#718096',
              marginTop: '8px'
            }}>
              📁 체크포인트: {statusBar.checkpointInfo.checkpoint_file}
              <br />
              📏 크기: {(statusBar.checkpointInfo.file_size / 1024 / 1024).toFixed(1)} MB
            </div>
          )}
        </StatusBar>
      )}

      {isTeaching && (
        <TeachingOverlay>
          <TeachingContent darkMode={darkMode}>
            <TeachingIcon>🧠</TeachingIcon>
            <TeachingTitle darkMode={darkMode}>AI 모델 학습 중...</TeachingTitle>
            <TeachingDescription darkMode={darkMode}>
              그룹이 지정된 고양이 이미지들을 AI 모델에게 알려주고 있습니다.
              <br />
              이 과정은 몇 분 정도 소요될 수 있습니다.
            </TeachingDescription>
            <TeachingProgress darkMode={darkMode}>
              <TeachingProgressBar />
            </TeachingProgress>
            <TeachingStatus darkMode={darkMode}>
              {teachingStep || '고양이 특성 분석 중...'}
            </TeachingStatus>
          </TeachingContent>
        </TeachingOverlay>
      )}

      {showProfileModal && (
        <ProfileModal onClick={() => setShowProfileModal(false)}>
          <ProfileModalContent onClick={(e) => e.stopPropagation()} darkMode={darkMode}>
            <ProfileModalTitle darkMode={darkMode}>
              👤 {selectedProfileGroup} 그룹의 프로필 이미지 설정
            </ProfileModalTitle>
            
            <div style={{ marginBottom: '16px', color: darkMode ? '#a0aec0' : '#4a5568' }}>
              {getGroupImages(selectedProfileGroup).length > 0 ? 
                '그룹의 이미지 중 하나를 선택하여 프로필로 설정하세요.' :
                '이 그룹에는 이미지가 없습니다. 먼저 이미지를 추가해주세요.'
              }
            </div>

            {getGroupImages(selectedProfileGroup).length > 0 && (
              <ProfileGrid>
                {getGroupImages(selectedProfileGroup).map((cat) => (
                  <ProfileImageCard
                    key={cat.id}
                    selected={selectedProfileImage === cat.id}
                    onClick={() => handleProfileImageSelect(cat.id)}
                    darkMode={darkMode}
                  >
                    <ProfileImage
                      src={cat.url ? `http://localhost:5000${cat.url}` : cat.filename ? `http://localhost:5000/cropped-images/${cat.filename}` : ''}
                      alt={`고양이 ${cat.id}`}
                      onError={(e) => {
                        console.error('프로필 선택 이미지 로드 실패:', cat.url || cat.filename);
                        e.target.style.display = 'none';
                      }}
                    />
                    <ProfileImageName darkMode={darkMode}>
                      {cat.filename ? cat.filename.split('_').pop() : `이미지 ${cat.id}`}
                    </ProfileImageName>
                  </ProfileImageCard>
                ))}
              </ProfileGrid>
            )}

            <ProfileModalButtons>
              <Button 
                className="secondary" 
                onClick={() => setShowProfileModal(false)}
              >
                취소
              </Button>
              {catProfiles[selectedProfileGroup] && (
                <Button 
                  className="danger" 
                  onClick={handleRemoveProfile}
                >
                  프로필 제거
                </Button>
              )}
              <Button 
                className="success" 
                onClick={handleSetProfile}
                disabled={!selectedProfileImage}
              >
                프로필 설정
              </Button>
            </ProfileModalButtons>
          </ProfileModalContent>
        </ProfileModal>
      )}

      {/* 컨텍스트 메뉴 */}
      {contextMenu.show && (
        <ContextMenu
          style={{
            left: contextMenu.x,
            top: contextMenu.y
          }}
          darkMode={darkMode}
        >
          <ContextMenuItem
            onClick={() => handleContextMenuAction('view-details')}
            darkMode={darkMode}
          >
            📋 상세 정보 보기
          </ContextMenuItem>
          
          <ContextMenuDivider darkMode={darkMode} />
          
          {/* 그룹에 추가 - 서브메뉴 */}
          <ContextMenuItemWithSubmenu darkMode={darkMode}>
            ➕ 그룹에 추가
            <span style={{ fontSize: '0.8rem' }}>▶</span>
            <SubMenu className="submenu" darkMode={darkMode} style={{ display: 'none' }}>
              {/* 기존 그룹들 */}
              {Object.keys(catGroups)
                .filter(groupKey => groupKey !== 'all' && groupKey !== 'unnamed')
                .map((groupKey) => (
                  <SubMenuItem
                    key={groupKey}
                    onClick={() => handleAddToSpecificGroup(groupKey)}
                    darkMode={darkMode}
                    $isProfile={catProfiles[groupKey] === contextMenu.catId}
                  >
                    {catGroups[groupKey].name}
                    {catProfiles[groupKey] && (
                      <span style={{ color: '#38a169', fontSize: '0.7rem' }}>👤</span>
                    )}
                  </SubMenuItem>
                ))}
              <ContextMenuDivider darkMode={darkMode} />
              <SubMenuItem
                onClick={() => handleContextMenuAction('add-to-group')}
                darkMode={darkMode}
              >
                ✏️ 새 그룹 만들기
              </SubMenuItem>
            </SubMenu>
          </ContextMenuItemWithSubmenu>
          
          {contextMenu.catName && contextMenu.catName !== '미지정' && (
            <>
              <ContextMenuItem
                onClick={handleSetProfileDirect}
                darkMode={darkMode}
              >
                👤 이 이미지를 프로필로 설정
              </ContextMenuItem>
              
              <ContextMenuItem
                onClick={() => handleContextMenuAction('select-similar')}
                darkMode={darkMode}
              >
                🎯 같은 그룹 선택
              </ContextMenuItem>
              
              <ContextMenuItem
                onClick={() => handleContextMenuAction('remove-from-group')}
                darkMode={darkMode}
              >
                🗑️ 그룹에서 제거
              </ContextMenuItem>
            </>
          )}
          
          <ContextMenuDivider darkMode={darkMode} />
          
          <ContextMenuItem
            onClick={() => handleContextMenuAction('copy-filename')}
            darkMode={darkMode}
          >
            📄 파일명 복사
          </ContextMenuItem>
          
          <ContextMenuItem
            onClick={() => {
              const cat = croppedCats.find(c => c.id === contextMenu.catId);
              if (cat && cat.url) {
                window.open(`http://localhost:5000${cat.url}`, '_blank');
              }
            }}
            darkMode={darkMode}
          >
            🔗 새 탭에서 보기
          </ContextMenuItem>
        </ContextMenu>
      )}
    </GalleryContainer>
  );
}

export default CatGallery; 
