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
    Three-tier Contrastive Loss based on InfoNCE framework
    
    Following the paper's fine-grained sample classification:
    - P (Positive, label=1): Chunks that can fully answer the query
    - N1 (Distractor, label=-1): Chunks used by LLM to rewrite query, seemingly relevant but cannot answer
    - N2 (Negative, label=0): Other unused negative chunks
    
    Loss computation strategy (following paper formulation):
    - Apply differentiated weights to N1 and N2 in logit space: β > α > 0
    - N1 (Distractor) uses weight β for stronger penalty, forcing fine-grained discrimination
    - N2 (Negative) uses weight α for weaker penalty
    - Only negatives (N1∪N2) in denominator to avoid diluting positive signal
    - Emphasize hierarchical separation: P ≻ N1 ≻ N2
    """
    def __init__(self, temperature: float = 0.05, alpha: float = 0.3, beta: float = 1.0):
        super().__init__()
        self.temperature = temperature  # Temperature scaling parameter τ
        self.alpha = alpha  # Weight for N2 (Negative)
        self.beta = beta    # Weight for N1 (Distractor), β > α

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute contrastive loss
        
        Args:
            similarities: [num_chunks] similarity scores s_k (cosine similarity)
            labels: [num_chunks] sample labels
                    1  -> P (Positive)
                    -1 -> N1 (Distractor) 
                    0  -> N2 (Negative)
        Returns:
            loss: Scalar loss value L(q) = -1/|P| Σ_{k∈P} log p_k
        """
        # Apply temperature scaling: s_k / τ
        scaled = similarities / self.temperature
        exp_sim = torch.exp(scaled)

        # Get masks for different sample types
        pos_mask = labels == 1      # P (Positive)
        n2_mask = labels == 0       # N2 (Negative)
        n1_mask = labels == -1      # N1 (Distractor)

        # Return zero loss if no positive samples
        if not pos_mask.any():
            return torch.tensor(0.0, device=similarities.device, requires_grad=True)

        # Compute denominator: apply differentiated weights to N1 and N2 in logit space
        # Paper formula: denominator contains Σ_{t∈N1∪N2} exp(s̃_t)
        # where s̃_t = s_t/τ + log(β) for t∈N1, s̃_t = s_t/τ + log(α) for t∈N2
        # Equivalent to: exp(s_t/τ) * β for N1, exp(s_t/τ) * α for N2
        denom = torch.tensor(0.0, device=similarities.device)
        
        if n2_mask.any():
            denom += (exp_sim[n2_mask] * self.alpha).sum()  # N2 uses α weight
        
        if n1_mask.any():
            denom += (exp_sim[n1_mask] * self.beta).sum()   # N1 uses β weight
        
        # Avoid division by zero
        denom = denom.clamp_min(1e-8)

        # Compute positive sample probability: p_k = exp(s̃_k) / (Σ_{t∈N1∪N2} exp(s̃_t) + exp(s̃_k))
        pos_exp = exp_sim[pos_mask]
        pos_probs = pos_exp / (pos_exp + denom)
        pos_probs = pos_probs.clamp_min(1e-8)
        
        # Compute negative log-likelihood loss: L(q) = -1/|P| Σ_{k∈P} log p_k
        loss = -torch.log(pos_probs).mean()

        return loss

class DualEncoderModel(nn.Module):
    """
    Dual Encoder Model - Core architecture of LORE method
    
    Following the paper's approach:
    - Query encoder M_q: Trainable, encodes query q to produce normalized embedding h_q
    - Document encoder M_d: Frozen, encodes candidate chunks c_k to produce normalized embedding h_k
    - Similarity computation: s_k = cos(h_q, h_k) cosine similarity
    - Contrastive loss: Three-tier contrastive learning loss based on InfoNCE framework
    
    Training process:
    - Document encoder M_d keeps frozen pre-trained parameters
    - Query encoder M_q is fine-tuned through contrastive loss
    - Optimization objective: θ_q* = argmin_θ_q E_{q~D}[L(q)]
    """
    def __init__(self, model_name='BAAI/bge-m3', temperature: float = 0.05, alpha: float = 0.3, beta: float = 1.0):
        super().__init__()
        
        # Document encoder M_d (frozen pre-trained parameters)
        self.document_encoder = SentenceTransformer(model_name)
        self.document_encoder.eval()
        
        # Freeze all document encoder parameters
        for param in self.document_encoder.parameters():
            param.requires_grad = False
        
        # Query encoder M_q (trainable)
        self.query_encoder = SentenceTransformer(model_name)
        
        # Contrastive learning loss function
        self.criterion = ContrastiveLoss(temperature=temperature, alpha=alpha, beta=beta)
        
        # Model name for saving and loading
        self.model_name = model_name
        
        print(f"Initialized dual encoder model with base model: {model_name}")
        print(f"Document encoder (M_d) parameters frozen, query encoder (M_q) parameters trainable")
        print(f"Loss function config: temperature(τ)={temperature}, alpha(N2 weight)={alpha}, beta(N1 weight)={beta}")
    
    def encode_documents(self, texts: List[str]) -> torch.Tensor:
        """
        Encode candidate chunks using document encoder M_d
        
        Args:
            texts: List of candidate chunk texts [c_1, c_2, ..., c_n]
        Returns:
            embeddings: Normalized embedding vectors [h_1, h_2, ..., h_n] where h_k = M_d(c_k) ∈ R^d
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
        Encode queries using query encoder M_q
        
        Args:
            texts: List of query texts [q_1, q_2, ..., q_m]
        Returns:
            embeddings: Normalized embedding vectors [h_q1, h_q2, ..., h_qm] where h_q = M_q(q) ∈ R^d
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
        Batch forward propagation to compute similarities and loss
        
        Args:
            questions: List of queries [q_1, q_2, ..., q_batch]
            chunks: List of candidate chunk lists [[c_{1,1}, ..., c_{1,n1}], [c_{2,1}, ..., c_{2,n2}], ...]
            labels: Optional label lists [[label_{1,1}, ...], [label_{2,1}, ...], ...]
                   Each label ∈ {1, -1, 0} corresponding to {P, N1, N2}
        Returns:
            If labels is None: List of similarities [s_1, s_2, ...]
            If labels is not None: (List of similarities, average loss)
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