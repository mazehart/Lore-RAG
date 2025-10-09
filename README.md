# LORE: Logic-ORiented Retriever Enhancement via Contrastive Learning

[English](#english) | [中文](#中文)

---

## English

### Overview

LORE (Logic-ORiented Retriever Enhancement) is a novel embedding enhancement method that improves retrieval performance through fine-grained contrastive learning. This repository contains the implementation of the paper "Logic-ORiented Retriever Enhancement via Contrastive Learning".

### Key Features

- **Three-tier Contrastive Learning**: Fine-grained sample classification with P (Positive), N1 (Distractor), and N2 (Negative) samples
- **Dual Encoder Architecture**: Frozen document encoder M_d and trainable query encoder M_q
- **InfoNCE-based Loss**: Differentiated weights (β > α) for hierarchical separation P ≻ N1 ≻ N2
- **Query Rewriting**: LLM-assisted dataset construction with discourse relations from Rhetorical Structure Theory (RST)
- **No External Dependencies**: Requires no external supervision, resources, or pre-retrieval analysis

### Architecture

The LORE method addresses the limitation of existing embedding models that struggle with complex logical expressions by:

1. **Dataset Construction**: Using LLMs to rewrite queries with distractor content while preserving original meaning
2. **Fine-grained Classification**: Categorizing chunks into three tiers based on their relevance and utility
3. **Contrastive Training**: Applying differentiated penalties to force fine-grained discrimination
4. **Asymmetric Training**: Keeping document encoder frozen while fine-tuning query encoder

### Repository Structure

```
CoEn-RAG/
├── 1_BuildDatasets/          # Dataset construction and preprocessing
├── 2_TrainModels/           # LORE training implementation
│   ├── train.py            # Main training script
│   ├── models.py           # Dual encoder model and contrastive loss
│   ├── train.sh            # Training shell script
│   └── utils/              # Training utilities
├── 3_Test/                 # Evaluation and testing
│   ├── evaluate.py         # Evaluation script with hard negative analysis
│   ├── models.py           # Model definitions for evaluation
│   ├── evaluate.sh         # Evaluation shell script
│   └── utils/              # Evaluation utilities
└── Logic-ORiented Retriever Enhancement via Contrastive Learning/  # Paper source
```

### Quick Start

#### 1. Training

```bash
cd 2_TrainModels
bash train.sh
```

#### 2. Evaluation

```bash
cd 3_Test
bash evaluate.sh
```

### Model Architecture Details

#### Contrastive Loss Function

The three-tier contrastive loss is based on the InfoNCE framework:

```
L(q) = -1/|P| Σ_{k∈P} log p_k
```

Where:
- **P (Positive, label=1)**: Chunks sufficient to answer the query
- **N1 (Distractor, label=-1)**: Chunks used by LLM in query rewriting, seemingly relevant but unhelpful
- **N2 (Negative, label=0)**: Other unused negative chunks

#### Dual Encoder Model

- **Query Encoder M_q**: Trainable, produces normalized embeddings h_q = M_q(q)
- **Document Encoder M_d**: Frozen, produces normalized embeddings h_k = M_d(c_k)
- **Similarity**: s_k = cos(h_q, h_k)

### Evaluation Metrics

- **Recall@k**: Standard retrieval recall at k=1,3,5,10
- **Hard Negative Interference**: Analysis of distractor impact on retrieval
- **Theoretical vs Actual Recall**: Comparison to measure distractor effects

### Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2024lore,
  title={Logic-ORiented Retriever Enhancement via Contrastive Learning},
  author={Zhang, Wenxuan and Jiang, Yuan-Hao and Qi, Changyong and Jia, Rui and Wu, Yonghe},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 中文

### 概述

LORE（逻辑导向的检索器增强）是一种新颖的嵌入增强方法，通过细粒度对比学习提高检索性能。本仓库包含论文《Logic-ORiented Retriever Enhancement via Contrastive Learning》的实现。

### 主要特性

- **三级对比学习**：使用P（正样本）、N1（干扰样本）和N2（负样本）的细粒度样本分类
- **双编码器架构**：冻结的文档编码器M_d和可训练的查询编码器M_q
- **基于InfoNCE的损失**：差异化权重（β > α）实现层级分离P ≻ N1 ≻ N2
- **查询重写**：基于修辞结构理论（RST）的LLM辅助数据集构建
- **无外部依赖**：不需要外部监督、资源或预检索分析

### 架构原理

LORE方法通过以下方式解决现有嵌入模型在复杂逻辑表达上的局限性：

1. **数据集构建**：使用LLM重写查询，在保持原意的同时融入干扰内容
2. **细粒度分类**：根据相关性和实用性将文档块分为三个层级
3. **对比训练**：应用差异化惩罚强制细粒度区分
4. **非对称训练**：保持文档编码器冻结，仅微调查询编码器

### 仓库结构

```
CoEn-RAG/
├── 1_BuildDatasets/          # 数据集构建和预处理
├── 2_TrainModels/           # LORE训练实现
│   ├── train.py            # 主训练脚本
│   ├── models.py           # 双编码器模型和对比损失
│   ├── train.sh            # 训练shell脚本
│   └── utils/              # 训练工具
├── 3_Test/                 # 评估和测试
│   ├── evaluate.py         # 带硬负样本分析的评估脚本
│   ├── models.py           # 评估用模型定义
│   ├── evaluate.sh         # 评估shell脚本
│   └── utils/              # 评估工具
└── Logic-ORiented Retriever Enhancement via Contrastive Learning/  # 论文源码
```

### 快速开始

#### 1. 训练

```bash
cd 2_TrainModels
bash train.sh
```

#### 2. 评估

```bash
cd 3_Test
bash evaluate.sh
```

### 模型架构详情

#### 对比损失函数

三级对比损失基于InfoNCE框架：

```
L(q) = -1/|P| Σ_{k∈P} log p_k
```

其中：
- **P（正样本，label=1）**：能够充分回答查询的文档块
- **N1（干扰样本，label=-1）**：被LLM用于查询重写的文档块，看似相关但无法回答查询
- **N2（负样本，label=0）**：其他未使用的负样本文档块

#### 双编码器模型

- **查询编码器M_q**：可训练，产生标准化嵌入h_q = M_q(q)
- **文档编码器M_d**：冻结，产生标准化嵌入h_k = M_d(c_k)
- **相似度**：s_k = cos(h_q, h_k)

### 评估指标

- **Recall@k**：标准检索召回率，k=1,3,5,10
- **硬负样本干扰**：分析干扰样本对检索的影响
- **理论vs实际召回**：比较以衡量干扰样本效果

### 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{zhang2024lore,
  title={Logic-ORiented Retriever Enhancement via Contrastive Learning},
  author={Zhang, Wenxuan and Jiang, Yuan-Hao and Qi, Changyong and Jia, Rui and Wu, Yonghe},
  journal={arXiv preprint},
  year={2024}
}
```