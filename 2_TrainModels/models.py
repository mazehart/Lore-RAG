#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Encoder Model and Contrastive Learning Loss Function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union, Tuple
import os
import math

import torch
import torch.nn as nn
import math

class ContrastiveLoss(nn.Module):
    """
    三级对比学习损失函数，基于InfoNCE：
    - 1: 正样本 (完全匹配)
    - 0: 弱负样本 (部分相关但不匹配)
    - -1: 强负样本 (完全不相关)
    
    损失计算策略：
    - 正样本：最大化与query的相似度
    - 弱负样本和强负样本：分别作为负样本，强负样本权重更大
    """
    def __init__(self, temperature: float = 0.05, alpha: float = 0.3, beta: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 弱负样本权重
        self.beta = beta    # 强负样本权重

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: [num_chunks] similarity scores
            labels: [num_chunks] 标签 (1: 正样本, 0: 弱负样本, -1: 强负样本)
        Returns:
            loss: scalar loss
        """
        # 缩放相似度
        scaled = similarities / self.temperature
        exp_sim = torch.exp(scaled)

        # 分别获取不同类型样本的mask
        pos_mask = labels == 1      # 正样本
        weak_neg_mask = labels == 0 # 弱负样本
        strong_neg_mask = labels == -1  # 强负样本

        # 如果没有正样本，返回0损失
        if not pos_mask.any():
            return torch.tensor(0.0, device=similarities.device, requires_grad=True)

        # 计算分母：包含弱负样本和强负样本，但权重不同
        denom = torch.tensor(0.0, device=similarities.device)
        
        if weak_neg_mask.any():
            denom += (exp_sim[weak_neg_mask] * self.alpha).sum()
        
        if strong_neg_mask.any():
            denom += (exp_sim[strong_neg_mask] * self.beta).sum()
        
        # 避免分母为0
        denom = denom.clamp_min(1e-8)

        # 计算正样本的概率和损失
        pos_exp = exp_sim[pos_mask]
        pos_probs = pos_exp / (pos_exp + denom)
        pos_probs = pos_probs.clamp_min(1e-8)
        
        # 计算负对数似然损失
        loss = -torch.log(pos_probs).mean()

        return loss

class DualEncoderModel(nn.Module):
    """
    Dual Encoder Model
    Input: a query and a list of chunks
    Output: similarity list
    Features: automatic loss computation, backpropagation, save/load, eval mode
    """
    def __init__(self, model_name='BAAI/bge-m3', temperature: float = 0.05, alpha: float = 0.3, beta: float = 1.0):
        super().__init__()
        
        # Document encoder (frozen parameters)
        self.document_encoder = SentenceTransformer(model_name)
        self.document_encoder.eval()
        
        # Freeze document encoder parameters
        for param in self.document_encoder.parameters():
            param.requires_grad = False
        
        # Query encoder (trainable)
        self.query_encoder = SentenceTransformer(model_name)
        
        # Loss function
        self.criterion = ContrastiveLoss(temperature=temperature, alpha=alpha, beta=beta)
        
        # Model name for saving and loading
        self.model_name = model_name
        
        print(f"Initialized dual encoder model using {model_name}")
        print(f"Document encoder parameters frozen, query encoder parameters trainable")
        print(f"Loss function: temperature={temperature}, alpha={alpha} (弱负样本权重), beta={beta} (强负样本权重)")
    
    def encode_documents(self, texts: List[str]) -> torch.Tensor:
        """
        Encode documents (chunks)
        """
        with torch.no_grad():
            embeddings = self.document_encoder.encode(
                texts, 
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        return embeddings.clone()
    
    def encode_queries(self, texts: List[str]) -> torch.Tensor:
        """
        Encode queries (questions)
        """
        features = self.query_encoder.tokenize(texts)
        target_device = next(self.query_encoder.parameters()).device
        for key in features:
            features[key] = features[key].to(target_device)

        output_features = self.query_encoder(features)
        if isinstance(output_features, dict) and 'sentence_embedding' in output_features:
            embeddings = output_features['sentence_embedding']
        else:
            embeddings = output_features

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self,
                questions: List[str],
                chunks: List[List[str]],
                labels: Optional[List[List[int]]] = None) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        Batch forward propagation
        """
        query_embeddings = self.batch_encode_queries(questions)
        chunk_embeddings_list = self.batch_encode_documents(chunks)
        
        similarities_list = []
        total_loss = 0.0
        
        for i, (query_emb, chunk_embs) in enumerate(zip(query_embeddings, chunk_embeddings_list)):
            similarities = torch.matmul(query_emb.unsqueeze(0), chunk_embs.T).squeeze(0)  # [num_chunks]
            similarities_list.append(similarities)
            
            if labels is not None:
                if isinstance(labels[i], torch.Tensor):
                    labels_tensor = labels[i].clone().detach().to(similarities.device)
                else:
                    labels_tensor = torch.as_tensor(labels[i], dtype=torch.float32).to(similarities.device)
                
                loss = self.criterion(similarities, labels_tensor)
                total_loss += loss
                
        if labels is None:
            return similarities_list
        else:
            return similarities_list, total_loss / len(questions)
    
    def batch_encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self.encode_queries(queries)
    
    def batch_encode_documents(self, all_chunks: List[List[str]]) -> List[torch.Tensor]:
        flat_chunks = []
        chunk_counts = []
        
        for chunks in all_chunks:
            flat_chunks.extend(chunks)
            chunk_counts.append(len(chunks))
        
        if flat_chunks:
            flat_embeddings = self.encode_documents(flat_chunks)
        else:
            return []
        
        chunk_embeddings_list = []
        start_idx = 0
        
        for count in chunk_counts:
            end_idx = start_idx + count
            chunk_embeddings_list.append(flat_embeddings[start_idx:end_idx])
            start_idx = end_idx
        
        return chunk_embeddings_list
    
    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        
        query_encoder_path = os.path.join(save_path, 'query_encoder')
        self.query_encoder.save(query_encoder_path)
        
        config = {
            'model_name': self.model_name,
            'temperature': self.criterion.temperature,
            'alpha': self.criterion.alpha,
            'beta': self.criterion.beta
        }
        
        config_path = os.path.join(save_path, 'config.json')
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to: {save_path}")
    
    def load(self, load_path: str):
        config_path = os.path.join(load_path, 'config.json')
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.criterion.temperature = config['temperature']
        if 'alpha' in config:
            self.criterion.alpha = config['alpha']
        if 'beta' in config:
            self.criterion.beta = config['beta']
        
        query_encoder_path = os.path.join(load_path, 'query_encoder')
        self.query_encoder = SentenceTransformer(query_encoder_path)
        
        print(f"Model loaded from {load_path}")
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.document_encoder.eval()
        if mode:
            self.query_encoder.train()
        else:
            self.query_encoder.eval()
        return self
    
    def eval(self):
        return self.train(False)