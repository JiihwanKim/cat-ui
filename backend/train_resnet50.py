# 데모 모드 설정 (파일 맨 앞에 추가)
DEMO_MODE = False  # True: 데모 모드, False: 실제 학습 모드

# matplotlib 백엔드 설정 (GUI 오류 방지)
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 대신 Agg 백엔드 사용

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from functools import lru_cache
# DINO 모델을 위한 추가 import
import torch.hub
# Rich 라이브러리 추가
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

# Rich 콘솔 초기화
console = Console()

# NT-Xent Contrastive Loss 함수 (사용되는 함수만 유지)
def nt_xent_loss(features, labels, temperature=0.07):
    """NT-Xent contrastive learning loss for paired augmentations"""
    b, n, d = features.shape
    assert n == 2, "Contrastive loss requires paired augmentations"
    
    # Normalize features
    features = F.normalize(features, dim=2)
    
    # Concatenate features
    features_flat = features.view(b * n, d)
    
    # Similarity matrix
    similarity_matrix = torch.matmul(features_flat, features_flat.T) / temperature
    
    # Mask to remove self-similarity
    mask = torch.eye(b * n, device=features.device, dtype=torch.bool)
    similarity_matrix = similarity_matrix[~mask].view(b * n, -1)
    
    # Labels for loss
    labels_flat = torch.arange(b * n, device=features.device) // n
    labels_flat = labels_flat.contiguous().view(-1, 1)
    mask = torch.eq(labels_flat, labels_flat.T).float()
    mask = mask[~torch.eye(b * n, device=features.device, dtype=torch.bool)].view(b * n, -1)
    
    # Positive and negative similarities
    positives = similarity_matrix[mask.bool()].view(b * n, -1)
    negatives = similarity_matrix[~mask.bool()].view(b * n, -1)
    
    # Logits
    logits = torch.cat([positives, negatives], dim=1)
    
    # Ground truth labels
    labels = torch.zeros(b * n, dtype=torch.long, device=features.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss

# 유틸리티 함수들
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(model, optimizer, epoch, acc, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
    }, filepath)

# cleanup_old_checkpoints 함수 제거 또는 단순화
def cleanup_old_checkpoints(save_dir, max_files=5):
    """체크포인트 관리 - 백업 없이 단순히 완료 메시지만 출력"""
    try:
        console.print(f"[green]✓[/green] 체크포인트 관리 완료")
        
    except Exception as e:
        console.print(f"[red]✗[/red] 체크포인트 관리 중 오류: {e}")

# 설정 클래스 정의
class Config:
    def __init__(self):
        # 기본 설정 - 과적합 방지를 위한 수정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 3
        self.batch_size = 128  
        self.num_epochs = 20  # 에포크 감소
        self.learning_rate = 1e-4  # 학습률을 1e-4로 수정
        self.weight_decay = 1e-4  # Weight decay를 1e-4로 수정
        self.temperature = 0.07  # Temperature를 0.07로 낮춤 (origin_train.py와 동일)
        
        # 모델 설정 - 일반 ResNet50으로 변경
        self.backbone_name = 'resnet50_pretrained'
        self.projection_dim = 128  # 특징 차원을 256으로 유지 (origin_train.py와 동일)
        self.image_size = 192  # 이미지 크기를 192로 수정
        
        # Validation 설정
        self.val_batch_size = 128
        self.val_samples_per_class = 200
        
        # 경로 설정
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 데모 모드에 따른 데이터셋 경로 설정
        if DEMO_MODE:
            self.dataset_dir = os.path.join(self.base_dir, 'datasets_fix')
            console.print(f"[yellow]⚠[/yellow] 데모 모드 활성화 - 데이터셋 경로: {self.dataset_dir}")
        else:
            self.dataset_dir = os.path.join(self.base_dir, 'datasets')
            console.print(f"[blue]ℹ[/blue] 실제 학습 모드 - 데이터셋 경로: {self.dataset_dir}")
        
        # 체크포인트를 백엔드 디렉토리에 저장
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        
        # 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# Projection Head 클래스 - 단순화된 버전 (Dropout 제거)
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

# Pretrained ResNet50 기반 모델 클래스
class CatDiscriminationModel(nn.Module):
    def __init__(self, config):
        super(CatDiscriminationModel, self).__init__()
        
        # Pretrained ResNet50 백본
        if config.backbone_name == 'resnet50_pretrained':
            # Pretrained ResNet50 모델 로드
            self.backbone = models.resnet50(pretrained=True)
            # 마지막 fully connected layer 제거
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 2048  # ResNet50의 특징 차원
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone_name}")
            
        # Projection head (단순화된 버전)
        self.projection_head = ProjectionHead(
            input_dim=backbone_dim,
            hidden_dim=512,
            output_dim=config.projection_dim
        )
        
        # 분류 헤드 - 단일 선형 레이어로 단순화
        self.classifier = nn.Linear(config.projection_dim, config.num_classes)
        
    def forward(self, x, return_features=False):
        # Pretrained ResNet50 백본에서 특징 추출
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Projection head
        projected_features = self.projection_head(features)
        
        # L2 정규화
        projected_features = F.normalize(projected_features, dim=1)
        
        if return_features:
            return projected_features
        
        # 분류 - 단일 선형 레이어
        logits = self.classifier(projected_features)
        return logits, projected_features

# 데이터셋 클래스 정의 (Contrastive Learning용) - 캐싱 제거로 초기 로딩 속도 개선
class CatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = self._load_image(image_path)
        
        if self.transform:
            if self.is_training:
                # 대조 학습을 위해 두 개의 다른 증강을 적용
                image1 = self.transform(image)
                image2 = self.transform(image)
                return (image1, image2), label
            else:
                return self.transform(image), label
            
        return image, label

# 데이터 로더 생성 함수 (최적화된 버전)
def create_data_loaders(config, train_samples_per_class=50):  # 50으로 제한
    """데이터셋을 한 번만 스캔하고 train/val 로더를 생성하여 최적화"""
    start_time = time.time()
    
    console.print("[blue]ℹ[/blue] 데이터 로더 생성 시작...")
    
    # 데이터셋 스캔
    scan_start = time.time()
    console.print("[blue]ℹ[/blue] 데이터셋 스캔 중...")
    
    image_paths = []
    labels = []
    
    # 각 클래스별로 모든 이미지 경로와 라벨 수집 (한 번만 수행)
    for class_id in range(1, config.num_classes + 1):
        class_dir = os.path.join(config.dataset_dir, str(class_id))
        if os.path.exists(class_dir):
            class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_paths.extend(class_images)
            labels.extend([class_id - 1] * len(class_images))
    scan_end = time.time()
    console.print(f"[blue]ℹ[/blue] 데이터셋 스캔 완료. 소요 시간: {scan_end - scan_start:.2f}초")

    console.print(f"[blue]ℹ[/blue] 전체 이미지 수: {len(image_paths)}")
    console.print(f"[blue]ℹ[/blue] 클래스별 이미지 수: {[labels.count(i) for i in range(config.num_classes)]}")
    
    # Train/validation 분할 (전체 데이터 대상)
    split_start = time.time()
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=0.2,  # 20%를 validation으로 사용 (더 많은 학습 데이터)
        stratify=labels, 
        random_state=42
    )
    split_end = time.time()
    console.print(f"[blue]ℹ[/blue] Train/Validation 분할 완료. 소요 시간: {split_end - split_start:.2f}초")

    
    # Training 데이터는 각 클래스별로 제한된 샘플만 사용 (필요시)
    if train_samples_per_class is not None:
        limit_start = time.time()
        limited_train_paths = []
        limited_train_labels = []
        for class_id in range(config.num_classes):
            class_indices = [i for i, label in enumerate(train_labels) if label == class_id]
            class_paths_for_label = [train_paths[i] for i in class_indices]
            
            if len(class_paths_for_label) > train_samples_per_class:
                selected_paths = random.sample(class_paths_for_label, train_samples_per_class)
            else:
                selected_paths = class_paths_for_label
                console.print(f"[yellow]⚠[/yellow] Warning: Class {class_id} has only {len(selected_paths)} training images")
            
            limited_train_paths.extend(selected_paths)
            limited_train_labels.extend([class_id] * len(selected_paths))
        
        train_paths, train_labels = limited_train_paths, limited_train_labels
        limit_end = time.time()
        console.print(f"[blue]ℹ[/blue] 학습 데이터 샘플링 완료. 소요 시간: {limit_end - limit_start:.2f}초")


    console.print(f"[blue]ℹ[/blue] 학습 샘플 수: {len(train_paths)}")
    console.print(f"[blue]ℹ[/blue] 검증 샘플 수: {len(val_paths)}")
    console.print(f"[blue]ℹ[/blue] 클래스별 학습 이미지 수: {[train_labels.count(i) for i in range(config.num_classes)]}")
    console.print(f"[blue]ℹ[/blue] 클래스별 검증 이미지 수: {[val_labels.count(i) for i in range(config.num_classes)]}")
    

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 회전 각도 증가
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 더 강한 색상 변화
        # RandomGrayscale 제거 - 고양이 품종 구분에 색상 정보가 중요하므로
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 생성 (캐싱 비활성화)
    dataset_start = time.time()
    train_dataset = CatDataset(train_paths, train_labels, train_transform, is_training=True)
    val_dataset = CatDataset(val_paths, val_labels, val_transform, is_training=False)
    dataset_end = time.time()
    console.print(f"[blue]ℹ[/blue] Dataset 객체 생성 완료. 소요 시간: {dataset_end - dataset_start:.2f}초")

    
    # 데이터 로더 생성 (속도 최적화)
    loader_start = time.time()

    pin = (config.device.type == 'cuda')  # CUDA이면 고정 메모리 활성화

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,                  # Windows에서 안정성 위해 유지
        pin_memory=pin,                 # ← 변경: CUDA일 때 True
        persistent_workers=False,       # num_workers=0이므로 False
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size, # validation용 배치 크기 사용
        shuffle=False, 
        num_workers=0,                  # Windows에서 안정성 위해 유지
        pin_memory=pin,                 # ← 변경: CUDA일 때 True
        persistent_workers=False,       # num_workers=0이므로 False
        drop_last=False
    )
    loader_end = time.time()
    console.print(f"[blue]ℹ[/blue] DataLoader 객체 생성 완료. 소요 시간: {loader_end - loader_start:.2f}초")
    
    end_time = time.time()
    console.print(f"[blue]ℹ[/blue] 전체 데이터 로더 생성 완료. 총 소요 시간: {end_time - start_time:.2f}초")
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

def train_epoch(model, train_loader, criterion, optimizer, config, epoch, scaler=None):
    """한 에포크 학습 - Rich 적용"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    contrastive_losses = AverageMeter()  # arc_losses를 contrastive_losses로 변경
    
    # Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Training", total=len(train_loader))
        
        for batch_idx, ((images1, images2), labels) in enumerate(train_loader):
            images1 = images1.to(config.device, non_blocking=True)
            images2 = images2.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            
            # AMP autocast
            with torch.cuda.amp.autocast(enabled=(config.device.type == 'cuda')):
                # Forward pass for both augmentations
                logits1, features1 = model(images1)
                logits2, features2 = model(images2)

                # Contrastive learning을 위한 특징 결합
                features = torch.stack([features1, features2], dim=1)  # [batch_size, 2, feature_dim]
                
                # Loss 계산
                ce_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2
                
                # Contrastive loss (NT-Xent) - arc_margin_loss 대신 nt_xent_loss 사용
                contrastive_loss_val = nt_xent_loss(
                    features, labels, config.temperature
                )
                
                # 전체 loss
                total_loss = ce_loss + 0.1 * contrastive_loss_val

            optimizer.zero_grad(set_to_none=True)
            if scaler and scaler.is_enabled():
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 메트릭 업데이트
            prec1 = (accuracy(logits1, labels, topk=(1,))[0] + accuracy(logits2, labels, topk=(1,))[0]) / 2
            losses.update(total_loss.item(), images1.size(0))
            top1.update(prec1.item(), images1.size(0))
            contrastive_losses.update(contrastive_loss_val.item(), images1.size(0))  # arc_losses를 contrastive_losses로 변경
            
            # Progress 업데이트
            progress.update(task, advance=1)
            
            # Description 업데이트 - Arc를 Contrastive로 변경
            progress.update(task, description=f"Epoch {epoch+1} | Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}% | Contrastive: {contrastive_losses.avg:.4f}")
            
            # 메트릭 업데이트 후 진행률 기록
            if ((batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader)):
                pass # progress_writer 제거
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, config, epoch=None):
    """검증 - Rich 적용"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Validation", total=len(val_loader))
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(config.device, non_blocking=True)
                labels = labels.to(config.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=(config.device.type == 'cuda')):
                    logits, _ = model(images)
                
                # Loss 계산
                loss = criterion(logits, labels)
                
                # 메트릭 업데이트
                prec1 = accuracy(logits, labels, topk=(1,))[0]
                losses.update(loss.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))
                
                # Progress 업데이트
                progress.update(task, advance=1)
                progress.update(task, description=f"Validation | Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}%")
                
                # 진행률 기록
                if ((batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(val_loader)):
                    pass # progress_writer 제거
    
    return losses.avg, top1.avg

def extract_features(model, data_loader, config):
    """모델에서 특징 추출 - 최적화된 버전 (GPU-CPU 전송 최적화)"""
    model.eval()
    features_list = []
    labels_list = []
    
    total_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # 배치 구조에 따른 처리 (단순화)
            try:
                if isinstance(batch, tuple) and len(batch) == 2:
                    first_item = batch[0]
                    labels = batch[1]
                    
                    # 첫 번째 아이템이 튜플인 경우 (paired augmentations)
                    if isinstance(first_item, tuple):
                        images1, images2 = first_item
                        images = images1.to(config.device, non_blocking=True)  # 첫 번째 augmentation만 사용
                    else:
                        # 일반적인 경우
                        images = first_item.to(config.device, non_blocking=True)
                    
                    labels = labels.to(config.device, non_blocking=True)
                    
                elif isinstance(batch, list) and len(batch) == 2:
                    first_item = batch[0]
                    labels = batch[1]
                    
                    # 첫 번째 아이템이 튜플인 경우 (paired augmentations)
                    if isinstance(first_item, tuple):
                        images1, images2 = first_item
                        images = images1.to(config.device, non_blocking=True)  # 첫 번째 augmentation만 사용
                    else:
                        # 일반적인 경우
                        images = first_item.to(config.device, non_blocking=True)
                    
                    labels = labels.to(config.device, non_blocking=True)
                
                else:
                    console.print(f"[red]✗[/red] Unexpected batch structure: {type(batch)}")
                    continue
                
                # 특징 추출
                features = model(images, return_features=True)
                
                # GPU에서 CPU로 전송을 최소화하기 위해 리스트에 추가
                features_list.append(features)
                labels_list.append(labels)
                
                # 간단한 진행 표시 (매 30번째 배치마다, 더 많은 샘플이므로)
                if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) == total_batches:
                    processed_samples = batch_idx * config.val_batch_size + images.size(0)
                    console.print(f'[blue]ℹ[/blue] Feature extraction: [{batch_idx + 1}/{total_batches}] '
                                  f'Processed: {processed_samples} samples')
                
            except Exception as e:
                console.print(f"[red]✗[/red] Error processing batch {batch_idx}: {e}")
                continue
    
    if not features_list:
        raise ValueError("No features were extracted. Check the data loader structure.")
    
    # 모든 특징을 한 번에 CPU로 전송 (최적화)
    console.print("[blue]ℹ[/blue] GPU에서 CPU로 특징 전송 중...")
    all_features = torch.cat(features_list, dim=0).cpu().numpy()
    all_labels = torch.cat(labels_list, dim=0).cpu().numpy()
    
    return all_features, all_labels

from sklearn.decomposition import PCA


def visualize_tsne(features, labels, class_names, save_path):
    """t-SNE를 사용하여 특징점 시각화"""
    console.print("[blue]ℹ[/blue] t-SNE 차원 축소 중...")
    
    # perplexity를 샘플 수에 맞게 조정
    n_samples = len(features)
    perplexity = min(30, max(1, n_samples - 1))  # 최소 1, 최대 30, 샘플 수보다 1 작게
    
    console.print(f"[blue]ℹ[/blue] 샘플 수: {n_samples}, 조정된 perplexity: {perplexity}")
    
    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 색상 팔레트 설정
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 빨강, 청록, 파랑
    markers = ['o', 's', '^']  # 원, 사각형, 삼각형
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1], 
            c=colors[i], 
            marker=markers[i],
            s=100, 
            alpha=0.7, 
            label=f'Class {i+1}: {class_name}',
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title('t-SNE Visualization of Cat Features (ResNet50 + Contrastive Learning)', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 축 레이블 제거
    plt.xticks([])
    plt.yticks([])
    
    # 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # plt.show() 대신 plt.close() 사용
    
    console.print(f"[green]✓[/green] t-SNE 시각화가 {save_path}에 저장되었습니다.")

def visualize_pca(features, labels, class_names, save_path):
    """PCA를 사용하여 특징점 시각화"""
    console.print("[blue]ℹ[/blue] Applying PCA dimensionality reduction...")
    
    # PCA 적용
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    
    # 설명된 분산 비율 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance = sum(explained_variance_ratio)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1], 
            c=colors[i], 
            marker=markers[i],
            s=100, 
            alpha=0.7, 
            label=f'Class {i+1}: {class_name}',
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title(f'PCA Visualization of Cat Features (ResNet50 + Contrastive Learning)\n(Explained Variance: {total_variance:.1%})', 
              fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # plt.show() 대신 plt.close() 사용
    
    console.print(f"[green]✓[/green] PCA visualization saved to: {save_path}")

def visualize_umap(features, labels, class_names, save_path):
    """UMAP을 사용하여 특징점 시각화"""
    try:
        import umap
        console.print("[blue]ℹ[/blue] UMAP 차원 축소 중...")
        
        # n_neighbors를 샘플 수에 맞게 조정
        n_samples = len(features)
        n_neighbors = min(15, max(1, n_samples - 1))  # 최소 1, 최대 15, 샘플 수보다 1 작게
        
        console.print(f"[blue]ℹ[/blue] 샘플 수: {n_samples}, 조정된 n_neighbors: {n_neighbors}")
        
        # UMAP 적용
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        features_2d = reducer.fit_transform(features)
        
        # 시각화
        plt.figure(figsize=(12, 10))
        
        # 색상 팔레트 설정
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 빨강, 청록, 파랑
        markers = ['o', 's', '^']  # 원, 사각형, 삼각형
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1], 
                c=colors[i], 
                marker=markers[i],
                s=100, 
                alpha=0.7, 
                label=f'Class {i+1}: {class_name}',
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.title('UMAP Visualization of Cat Features (Pretrained ResNet50 + Contrastive Learning)', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP Component 1', fontsize=12)
        plt.ylabel('UMAP Component 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 축 레이블 제거
        plt.xticks([])
        plt.yticks([])
        
        # 저장
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # plt.show() 대신 plt.close() 사용
        
        console.print(f"[green]✓[/green] UMAP 시각화가 {save_path}에 저장되었습니다.")
        return save_path
        
    except ImportError:
        console.print("[red]✗[/red] UMAP 라이브러리가 설치되지 않았습니다. UMAP 시각화를 건너뜁니다.")
        return None

def create_visualization_comparison(features, labels, class_names, save_dir):
    """여러 시각화 방법으로 비교"""
    console.print("[blue]ℹ[/blue] 여러 시각화 방법으로 특징 비교 중...")
    
    # t-SNE 시각화
    tsne_path = os.path.join(save_dir, 'tsne_visualization_resnet50_contrastive.png')
    visualize_tsne(features, labels, class_names, tsne_path)
    
    # PCA 시각화
    pca_path = os.path.join(save_dir, 'pca_visualization_resnet50_contrastive.png')
    visualize_pca(features, labels, class_names, pca_path)
    
    return tsne_path, pca_path, None  # umap_path 대신 None 반환

def create_summary_table(config, train_size, val_size):
    """설정 요약 테이블 생성"""
    table = Table(title="Training Configuration", box=box.ROUNDED)
    
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Backbone", config.backbone_name)
    table.add_row("Learning Rate", f"{config.learning_rate:.1e}")
    table.add_row("Batch Size", str(config.batch_size))
    table.add_row("Epochs", str(config.num_epochs))
    table.add_row("Temperature", str(config.temperature))
    table.add_row("Training Samples", str(train_size))
    table.add_row("Validation Samples", str(val_size))
    table.add_row("Device", str(config.device))
    
    return table

def create_epoch_table(epoch, train_loss, train_acc, val_loss, val_acc, best_acc, lr):
    """에포크 결과 테이블 생성"""
    table = Table(title=f"Epoch {epoch} Results", box=box.ROUNDED)
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Train Loss", f"{train_loss:.4f}")
    table.add_row("Train Accuracy", f"{train_acc:.2f}%")
    table.add_row("Val Loss", f"{val_loss:.4f}")
    table.add_row("Val Accuracy", f"{val_acc:.2f}%")
    table.add_row("Best Accuracy", f"{best_acc:.2f}%")
    table.add_row("Learning Rate", f"{lr:.6f}")
    
    return table

def main():
    main_start_time = time.time()
    
    # Rich 시작 메시지
    console.print(Panel.fit(
        "[bold blue]ResNet50 + Contrastive Learning Training[/bold blue]\n"
        f"[dim]Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="blue"
    ))
    
    config = Config()

    # 시작 상태 보고
    # progress_writer({"status":"starting","phase":"init","epoch":0,"num_epochs":config.num_epochs}) # 제거
    
    # 데이터 로더 생성
    train_loader, val_loader, train_size, val_size = create_data_loaders(config, train_samples_per_class=200)
    
    # 설정 요약 테이블 출력
    summary_table = create_summary_table(config, train_size, val_size)
    console.print(summary_table)
    
    # 모델 생성
    with console.status("[bold green]Loading model...") as status:
        model = CatDiscriminationModel(config).to(config.device)
        status.update("[bold green]Model loaded successfully!")
    
    console.print(f"[green]✓[/green] Model loaded: {config.backbone_name}")

    # 옵티마이저 단일화 - origin_train.py와 동일하게 모든 파라미터에 동일한 학습률 적용
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,  # 모든 파라미터에 동일한 학습률 적용
        weight_decay=config.weight_decay
    )
    
    # 학습 가능한 파라미터 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    projection_params = sum(p.numel() for p in model.projection_head.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    console.print(f"[INFO] 전체 파라미터 수: {total_params:,}")
    console.print(f"[INFO] 학습 가능한 파라미터 수: {trainable_params:,}")
    console.print(f"[INFO] 백본 파라미터 수: {backbone_params:,} (학습률: {config.learning_rate:.1e})")
    console.print(f"[INFO] Projection head 파라미터 수: {projection_params:,} (학습률: {config.learning_rate:.1e})")
    console.print(f"[INFO] Classifier 파라미터 수: {classifier_params:,} (학습률: {config.learning_rate:.1e})")
    console.print(f"[INFO] 백본 freeze 상태: False (전체 모델 학습, 단일 학습률)")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler - CosineAnnealingLR로 변경
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    if config.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')  # PyTorch 2.x 권장(가능한 경우)
        except Exception:
            pass

    scaler = torch.cuda.amp.GradScaler(enabled=(config.device.type == 'cuda'))
    
    # 학습 루프
    best_acc = 0.0
    patience_counter = 0
    patience = 5
    validation_interval = 3  # 3 에포크마다 validation
    
    console.print(Panel.fit(
        f"[bold yellow]Training Configuration[/bold yellow]\n"
        f"Total Epochs: {config.num_epochs}\n"
        f"Validation Interval: Every {validation_interval} epochs\n"
        f"Early Stopping Patience: {patience}\n"
        f"Scheduler: CosineAnnealingLR (T_max={config.num_epochs}, eta_min=1e-6)",
        border_style="yellow"
    ))
    
    # 전체 학습 진행률을 위한 Rich progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        epoch_task = progress.add_task("Training Progress", total=config.num_epochs)
        
        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()
            
            # 학습
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, config, epoch, scaler=scaler
            )
            
            # Validation 간격 조절
            if (epoch + 1) % validation_interval == 0 or epoch == 0:
                val_loss, val_acc = validate(
                    model, val_loader, criterion, config, epoch=epoch
                )
                
                # Learning rate 업데이트 - CosineAnnealingLR는 매 에포크마다 호출
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()  # 매 에포크마다 호출
                new_lr = optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    console.print(f"[yellow]⚠[/yellow] Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
                
                # 에포크 결과 테이블 출력
                epoch_table = create_epoch_table(epoch+1, train_loss, train_acc, val_loss, val_acc, best_acc, new_lr)
                console.print(epoch_table)
                
                # 모델 저장
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience_counter = 0
                    
                    # 같은 파일명으로 저장 (타임스탬프 제거)
                    checkpoint_filename = 'best_model_resnet50_contrastive.pth'
                    checkpoint_path = os.path.join(config.save_dir, checkpoint_filename)
                    
                    save_checkpoint(
                        model, optimizer, epoch, val_acc,
                        checkpoint_path
                    )
                    console.print(f"[green]✓[/green] New best model saved! Accuracy: {best_acc:.2f}%")
                    console.print(f"[green]✓[/green] Checkpoint: {checkpoint_filename}")
                else:
                    patience_counter += 1
                    console.print(f"[red]✗[/red] No improvement for {patience_counter} epochs")
                
                # Early stopping
                if patience_counter >= patience:
                    console.print(f"[red]⚠[/red] Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # Validation을 하지 않는 에포크에서도 스케줄러 업데이트
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()  # 매 에포크마다 호출
                new_lr = optimizer.param_groups[0]['lr']
                
                # Validation을 하지 않는 에포크
                console.print(f"[dim]Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%[/dim]")
                console.print(f"[dim]Validation: Skipped (next validation at epoch {epoch + validation_interval})[/dim]")
                console.print(f"[dim]Learning Rate: {new_lr:.6f}[/dim]")
            
            # 전체 진행률 업데이트
            progress.update(epoch_task, advance=1)
            progress.update(epoch_task, description=f"Training | Best Val Acc: {best_acc:.2f}%")
            
            epoch_end_time = time.time()
            console.print(f"[dim]Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f}s[/dim]")

            # 에포크 요약 기록
            # progress_writer({ # 제거
            #     "status": "running",
            #     "phase": "epoch_end",
            #     "epoch": epoch + 1,
            #     "num_epochs": config.num_epochs,
            #     "train_loss": float(train_loss),
            #     "train_acc": float(train_acc),
            #     "val_loss": float(val_loss),
            #     "val_acc": float(val_acc),
            #     "best_acc": float(best_acc),
            #     "lr": float(optimizer.param_groups[0]['lr'])
            # })

        # 완료 기록
        # progress_writer({"status":"completed","phase":"done","best_acc": float(best_acc)}) # 제거
    
    # 학습 완료 메시지
    console.print(Panel.fit(
        f"[bold green]Training Completed![/bold green]\n"
        f"Best Accuracy: {best_acc:.2f}%\n"
        f"Total Time: {time.time() - main_start_time:.2f}s",
        border_style="green"
    ))
    
    # 특징 추출
    console.print(Panel.fit(
        "[bold blue]Feature Extraction[/bold blue]",
        border_style="blue"
    ))
    
    with console.status("[bold green]Extracting features...") as status:
        val_features, val_labels = extract_features(model, val_loader, config)
        status.update("[bold green]Feature extraction completed!")

    console.print(f"[green]✓[/green] Total features: {len(val_features)}")
    console.print(f"[green]✓[/green] Feature dimension: {val_features.shape[1]}")
    console.print(f"[green]✓[/green] Class distribution: {[np.sum(val_labels == i) for i in range(config.num_classes)]}")
    
    # 클래스 이름 정의
    class_names = ['Class 1', 'Class 2', 'Class 3']
    
    # 여러 시각화 방법으로 비교
    tsne_path, pca_path, umap_path = create_visualization_comparison(
        val_features, val_labels, class_names, config.save_dir
    )

    # 특징 통계 정보 저장
    with console.status("[bold green]Saving feature statistics...") as status:
        stats_save_path = os.path.join(config.save_dir, 'feature_stats_resnet50_contrastive.txt')
        with open(stats_save_path, 'w', encoding='utf-8') as f:
            f.write(f"=== ResNet50 + Contrastive Learning 학습 후 특징 분석 결과 ===\n")
            f.write(f"백본 모델: {config.backbone_name}\n")
            f.write(f"학습 방식: 전체 모델 학습 (단일 학습률) + Contrastive Learning\n")
            f.write(f"Temperature: {config.temperature}\n")
            f.write(f"Projection Dimension: {config.projection_dim}\n")
            f.write(f"총 샘플 수: {len(val_features)}\n")
            f.write(f"특징 차원: {val_features.shape[1]}\n")
            f.write(f"클래스별 샘플 수:\n")
            for i in range(config.num_classes):
                f.write(f"  Class {i+1}: {np.sum(val_labels == i)}개\n")
            f.write(f"\n특징 통계:\n")
            f.write(f"  평균: {np.mean(val_features):.4f}\n")
            f.write(f"  표준편차: {np.std(val_features):.4f}\n")
            f.write(f"  최소값: {np.min(val_features):.4f}\n")
            f.write(f"  최대값: {np.max(val_features):.4f}\n")
            f.write(f"\n시각화 파일:\n")
            f.write(f"  t-SNE: {tsne_path}\n")
            f.write(f"  PCA: {pca_path}\n")
            if umap_path:
                f.write(f"  UMAP: {umap_path}\n")
        status.update("[bold green]Feature statistics saved!")

    console.print(f"[green]✓[/green] Feature statistics saved to: {stats_save_path}")
    console.print(f"[green]✓[/green] Visualizations saved:")
    console.print(f"  • t-SNE: {tsne_path}")
    console.print(f"  • PCA: {pca_path}")
    if umap_path:
        console.print(f"  • UMAP: {umap_path}")
    
    # 최종 완료 메시지
    console.print(Panel.fit(
        f"[bold green]All tasks completed successfully![/bold green]\n"
        f"Total execution time: {time.time() - main_start_time:.2f}s",
        border_style="green"
    ))

    # 학습 완료 후 체크포인트 관리 (백업 없이)
    cleanup_old_checkpoints(config.save_dir, max_files=3)


if __name__ == "__main__":
    main()
