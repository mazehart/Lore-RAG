#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LORE (Logic-ORiented Retriever Enhancement) Evaluation Script

This script evaluates the retrieval performance of LORE-enhanced embedding models
as described in the paper: "Logic-ORiented Retriever Enhancement via Contrastive Learning"

Key Evaluation Features:
- Multi-tier retrieval analysis distinguishing P/N1/N2 sample types
- Hard negative interference analysis to measure distractor impact
- Recall@k metrics (k=1,3,5,10) for comprehensive performance assessment
- Batch evaluation for efficient processing of large datasets
- Detailed analysis of how distractors (N1) affect positive sample retrieval
- Comparison between theoretical and actual recall rates
"""

import os
import torch
import numpy as np
import json
from datasets import load_from_disk
from models import DualEncoderModel
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
from utils.unified_monitor import UnifiedMonitor

# Initialize unified monitor
unified_monitor = UnifiedMonitor(device_id="2")
    
    # Setup monitor environment
unified_monitor.setup()

def analyze_hard_negative_impact(similarities: torch.Tensor, labels: List[int], k_values: List[int] = [3, 5, 10]) -> Dict[str, Dict[str, float]]:
    """
    Analyze the interference impact of hard negative samples on retrieval results
    
    Args:
        similarities: similarity scores [num_chunks]
        labels: true labels [num_chunks] (1: positive, 0: negative, -1: hard negative)
        k_values: list of k values to analyze
    
    Returns:
        dictionary containing hard negative interference analysis for each k value
    """
    analysis = {}
    
    # Get indices for each sample type
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    hard_negative_indices = [i for i, label in enumerate(labels) if label == -1]
    
    # Get sorted indices
    _, sorted_indices = torch.sort(similarities, descending=True)
    sorted_indices = sorted_indices.cpu().tolist()
    
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        
        # Analyze ranking situation of hard negative samples
        hard_negative_ranks = []
        for hn_idx in hard_negative_indices:
            if hn_idx in sorted_indices:
                rank = sorted_indices.index(hn_idx) + 1  # 1-based ranking
                hard_negative_ranks.append(rank)
        
        # Analyze cases where positive samples are pushed out of top-k by hard negative samples
        positive_in_topk = sum(1 for idx in top_k_indices if labels[idx] == 1)
        hard_negative_in_topk = sum(1 for idx in top_k_indices if labels[idx] == -1)
        
        # Calculate theoretical positive recall without hard negative samples
        non_hard_negative_indices = [i for i, label in enumerate(labels) if label != -1]
        non_hard_negative_similarities = similarities[non_hard_negative_indices]
        non_hard_negative_labels = [labels[i] for i in non_hard_negative_indices]
        
        if len(non_hard_negative_similarities) > 0:
            _, non_hn_sorted_indices = torch.sort(non_hard_negative_similarities, descending=True)
            non_hn_top_k = non_hn_sorted_indices[:min(k, len(non_hn_sorted_indices))].cpu().tolist()
            theoretical_positive_recall = sum(1 for idx in non_hn_top_k if non_hard_negative_labels[idx] == 1) / len(positive_indices) if len(positive_indices) > 0 else 0.0
        else:
            theoretical_positive_recall = 0.0
        
        actual_positive_recall = positive_in_topk / len(positive_indices) if len(positive_indices) > 0 else 0.0
        
        analysis[f'@{k}'] = {
            'hard_negative_interference_rate': hard_negative_in_topk / k if k > 0 else 0.0,
            'avg_hard_negative_rank': np.mean(hard_negative_ranks) if hard_negative_ranks else float('inf'),
            'theoretical_positive_recall': theoretical_positive_recall,
            'actual_positive_recall': actual_positive_recall,
            'recall_loss_due_to_hard_negatives': max(0, theoretical_positive_recall - actual_positive_recall),
            'hard_negatives_above_positives': sum(1 for hn_rank in hard_negative_ranks if hn_rank <= k)
        }
    
    return analysis

def calculate_metrics_at_k(similarities: torch.Tensor, labels: List[int], k: int) -> Dict[str, float]:
    """
    Calculate recall at @k, distinguishing between positive (1), negative (0), and hard negative (-1) samples
    
    Args:
        similarities: similarity scores [num_chunks]
        labels: true labels [num_chunks] (1: positive, 0: negative, -1: hard negative)
        k: top-k
    
    Returns:
        dictionary containing recall rates and interference analysis for each sample type
    """
    # Get top-k indices
    _, top_k_indices = torch.topk(similarities, min(k, len(similarities)), largest=True)
    top_k_indices = top_k_indices.cpu().numpy()
    
    # Count number of samples for each type
    positive_count = sum(1 for label in labels if label == 1)  # positive samples
    negative_count = sum(1 for label in labels if label == 0)  # negative samples
    hard_negative_count = sum(1 for label in labels if label == -1)  # hard negative samples
    
    # Count number of each sample type in top-k
    retrieved_positive = sum(1 for i in top_k_indices if labels[i] == 1)
    retrieved_negative = sum(1 for i in top_k_indices if labels[i] == 0)
    retrieved_hard_negative = sum(1 for i in top_k_indices if labels[i] == -1)
    
    # Calculate recall rates
    positive_recall = retrieved_positive / positive_count if positive_count > 0 else 0.0
    negative_recall = retrieved_negative / negative_count if negative_count > 0 else 0.0
    hard_negative_recall = retrieved_hard_negative / hard_negative_count if hard_negative_count > 0 else 0.0
    
    # Calculate interference analysis metrics
    total_retrieved = len(top_k_indices)
    positive_precision = retrieved_positive / total_retrieved if total_retrieved > 0 else 0.0
    hard_negative_interference = retrieved_hard_negative / total_retrieved if total_retrieved > 0 else 0.0
    
    return {
        'positive_recall': positive_recall,
        'negative_recall': negative_recall,
        'hard_negative_recall': hard_negative_recall,
        'positive_precision': positive_precision,
        'hard_negative_interference': hard_negative_interference,
        'retrieved_positive': retrieved_positive,
        'retrieved_negative': retrieved_negative,
        'retrieved_hard_negative': retrieved_hard_negative,
        'total_positive': positive_count,
        'total_negative': negative_count,
        'total_hard_negative': hard_negative_count
    }

def evaluate_dataset(model: DualEncoderModel, dataset, dataset_name: str, batch_size: int = 16) -> Tuple[Dict[str, Dict[str, float]], List[Dict]]:
    """
    Evaluate a single dataset (batch version)
    
    Args:
        model: dual encoder model
        dataset: dataset
        dataset_name: name of the dataset
        batch_size: batch size
    
    Returns:
        dictionary containing metrics for each k value and a list of prediction results
    """
    print(f"\nEvaluating dataset: {dataset_name}")
    print(f"Number of samples: {len(dataset)}")
    
    model.eval()
    
    # Store metrics for all samples
    all_metrics = {k: {
        'positive_recall': [],
        'negative_recall': [],
        'hard_negative_recall': [],
        'positive_precision': [],
        'hard_negative_interference': [],
        'retrieved_positive': [],
        'retrieved_negative': [],
        'retrieved_hard_negative': [],
        'total_positive': [],
        'total_negative': [],
        'total_hard_negative': []
    } for k in [3, 5, 10]}
    
    # Store hard negative interference analysis
    all_hard_negative_analysis = {k: {
        'hard_negative_interference_rate': [],
        'avg_hard_negative_rank': [],
        'theoretical_positive_recall': [],
        'actual_positive_recall': [],
        'recall_loss_due_to_hard_negatives': [],
        'hard_negatives_above_positives': []
    } for k in [3, 5, 10]}
    
    # Store prediction results
    predictions = []
    
    with torch.no_grad():
        # Create batch indices
        total_samples = len(dataset)
        batch_indices = [(i, min(i + batch_size, total_samples)) for i in range(0, total_samples, batch_size)]
        
        # Batch evaluation
        for start_idx, end_idx in tqdm(batch_indices, desc=f"Evaluating {dataset_name}"):
            # Collect samples for current batch
            batch_queries = []
            batch_chunks = []
            batch_labels = []
            
            # Get samples for the current batch
            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                batch_queries.append(sample['query'])
                batch_chunks.append(sample['chunks'])
                batch_labels.append(sample['labels'])
            
            # Get similarity scores in batch
            similarities_list = model.forward(batch_queries, batch_chunks)
            
            # Process results for each sample
            for j, (similarities, query, chunks, labels) in enumerate(zip(similarities_list, batch_queries, batch_chunks, batch_labels)):
                sample_idx = start_idx + j
                
                # Get sorted indices
                _, sorted_indices = torch.sort(similarities, descending=True)
                sorted_indices = sorted_indices.cpu().tolist()
                
                # Build prediction result dictionary
                sample_data = dataset[sample_idx]
                prediction = {
                    'id': sample_idx,
                    'query': query,
                    'chunks': [chunks[idx] for idx in sorted_indices[:10]],  # save only top 10 results
                    'similarities': [similarities[idx].item() for idx in sorted_indices[:10]],  # save only top 10 similarity scores
                    'sorted_indices': sorted_indices[:10],  # save only top 10 indices
                    'labels': [labels[idx] for idx in sorted_indices[:10]],  # corresponding labels
                    'original_labels': labels,
                    'original_chunks': chunks  # save original chunks list
                }
                
                # Add extra fields from new dataset format (if they exist)
                if 'original_query' in sample_data:
                    prediction['original_query'] = sample_data['original_query']
                if 'used_distractors' in sample_data:
                    prediction['used_distractors'] = sample_data['used_distractors']
                if 'discourse_relation' in sample_data:
                    prediction['discourse_relation'] = sample_data['discourse_relation']
                if 'answer' in sample_data:
                    prediction['answer'] = sample_data['answer']
                if 'uuid' in sample_data:
                    prediction['uuid'] = sample_data['uuid']
                predictions.append(prediction)
                
                # Calculate metrics for each k value
                for k in [3, 5, 10]:
                    metrics = calculate_metrics_at_k(similarities, labels, k)
                    for metric_name, value in metrics.items():
                        all_metrics[k][metric_name].append(value)
                    
                    # Calculate hard negative interference analysis
                    hard_negative_analysis = analyze_hard_negative_impact(similarities, labels, [k])
                    for analysis_name, value in hard_negative_analysis[f'@{k}'].items():
                        all_hard_negative_analysis[k][analysis_name].append(value)
    
    # Calculate average metrics
    avg_metrics = {}
    for k in [3, 5, 10]:
        avg_metrics[f'@{k}'] = {
            'positive_recall': np.mean(all_metrics[k]['positive_recall']),
            'negative_recall': np.mean(all_metrics[k]['negative_recall']),
            'hard_negative_recall': np.mean(all_metrics[k]['hard_negative_recall']),
            'positive_precision': np.mean(all_metrics[k]['positive_precision']),
            'hard_negative_interference': np.mean(all_metrics[k]['hard_negative_interference']),
            'avg_retrieved_positive': np.mean(all_metrics[k]['retrieved_positive']),
            'avg_retrieved_negative': np.mean(all_metrics[k]['retrieved_negative']),
            'avg_retrieved_hard_negative': np.mean(all_metrics[k]['retrieved_hard_negative']),
            'avg_total_positive': np.mean(all_metrics[k]['total_positive']),
            'avg_total_negative': np.mean(all_metrics[k]['total_negative']),
            'avg_total_hard_negative': np.mean(all_metrics[k]['total_hard_negative']),
            # Hard negative interference analysis
            'hard_negative_interference_rate': np.mean(all_hard_negative_analysis[k]['hard_negative_interference_rate']),
            'avg_hard_negative_rank': np.mean(all_hard_negative_analysis[k]['avg_hard_negative_rank']),
            'theoretical_positive_recall': np.mean(all_hard_negative_analysis[k]['theoretical_positive_recall']),
            'actual_positive_recall': np.mean(all_hard_negative_analysis[k]['actual_positive_recall']),
            'recall_loss_due_to_hard_negatives': np.mean(all_hard_negative_analysis[k]['recall_loss_due_to_hard_negatives']),
            'hard_negatives_above_positives': np.mean(all_hard_negative_analysis[k]['hard_negatives_above_positives'])
        }
    
    return avg_metrics, predictions

def collect_dataset_stats(dataset, dataset_name: str, dataset_type: str) -> Dict:
    """
    Collect dataset statistics
    
    Args:
        dataset: dataset
        dataset_name: name of the dataset
        dataset_type: type of the dataset
    
    Returns:
        dictionary containing statistical information
    """
    stats = {
        'total_samples': len(dataset),
        'avg_chunks': 0,
        'avg_positive_chunks': 0
    }
    
    if len(dataset) > 0:
        total_chunks = sum(len(sample['chunks']) for sample in dataset)
        total_positive = sum(sum(1 for label in sample['labels'] if label == 1) for sample in dataset)
        total_negative = sum(sum(1 for label in sample['labels'] if label == 0) for sample in dataset)
        total_hard_negative = sum(sum(1 for label in sample['labels'] if label == -1) for sample in dataset)
        
        stats['avg_chunks'] = total_chunks / len(dataset)
        stats['avg_positive_chunks'] = total_positive / len(dataset)
        stats['avg_negative_chunks'] = total_negative / len(dataset)
        stats['avg_hard_negative_chunks'] = total_hard_negative / len(dataset)
        stats['total_positive'] = total_positive
        stats['total_negative'] = total_negative
        stats['total_hard_negative'] = total_hard_negative
        
        # Collect discourse relation statistics
        discourse_relations = {}
        for sample in dataset:
            if 'discourse_relation' in sample:
                relation = sample['discourse_relation']
                discourse_relations[relation] = discourse_relations.get(relation, 0) + 1
        if discourse_relations:
            stats['discourse_relations'] = discourse_relations
    
    return stats

def save_predictions(predictions: List[Dict], dataset_name: str, output_dir: str) -> None:
    """
    Save prediction results and ground truth pairs to a JSON file
    
    Args:
        predictions: list of prediction results
        dataset_name: name of the dataset
        output_dir: output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output file path
    output_file = os.path.join(output_dir, f"{dataset_name}_predictions.json")
    
    # Prepare serializable data, highlighting prediction results and ground truth pairs
    serializable_predictions = []
    for pred in predictions:
        # Deep copy prediction results to avoid modifying original data
        serializable_pred = dict(pred)
        # Remove non-serializable data
        if 'similarities' in serializable_pred:
            serializable_pred['similarities'] = [float(s) for s in serializable_pred['similarities']]
        
        # Ensure original_chunks is serializable
        if 'original_chunks' in serializable_pred:
            # If elements in original_chunks are not simple strings, they may need conversion
            serializable_pred['original_chunks'] = [str(chunk) for chunk in serializable_pred['original_chunks']]
        
        # Reorganize data, highlighting prediction results and ground truth pairs
        prediction_answer_pairs = []
        for i in range(len(serializable_pred['chunks'])):
            # Get original indices
            original_idx = serializable_pred['sorted_indices'][i] if i < len(serializable_pred['sorted_indices']) else -1
            
            label_value = serializable_pred['labels'][i] if i < len(serializable_pred['labels']) else 0
            pair = {
                'predicted_chunk': serializable_pred['chunks'][i],
                'similarity_score': float(serializable_pred['similarities'][i]) if i < len(serializable_pred['similarities']) else 0.0,
                'label': int(label_value),
                'is_positive': label_value == 1,
                'is_negative': label_value == 0,
                'is_hard_negative': label_value == -1,
                'original_index': original_idx
            }
            prediction_answer_pairs.append(pair)
        
        # Update prediction result dictionary
        serializable_pred['prediction_answer_pairs'] = prediction_answer_pairs
        serializable_pred['query'] = pred['query']
        
        # Preserve additional fields from new dataset format
        for field in ['original_query', 'used_distractors', 'discourse_relation', 'answer', 'uuid']:
            if field in pred:
                serializable_pred[field] = pred[field]
        
        # Add ground truth information
        ground_truth_chunks = []
        for idx, label in enumerate(pred['original_labels']):
            if label == 1:  # only positive samples are ground truth
                if 'original_chunks' in pred and idx < len(pred['original_chunks']):
                    ground_truth_chunks.append(pred['original_chunks'][idx])
        
        serializable_pred['ground_truth_chunks'] = ground_truth_chunks
        serializable_pred['original_labels'] = [int(label) for label in pred['original_labels']]
        
        serializable_predictions.append(serializable_pred)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"Prediction results and ground truth pairs saved to: {output_file}")


def print_results(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], dataset_stats: Dict[str, Dict[str, Dict]] = None):
    """
    Print evaluation results
    """
    print("\n" + "="*80)
    print("Retrieval System Evaluation Results")
    print("="*80)
    
    for dataset_name, dataset_results in results.items():
        print(f"\nDataset: {dataset_name}")
        
        # Display dataset statistics
        if dataset_stats and dataset_name in dataset_stats:
            print("\n数据集统计信息:")
            for dataset_type in ['test', 'modfied_test']:
                if dataset_type in dataset_stats[dataset_name]:
                    stats = dataset_stats[dataset_name][dataset_type]
                    print(f"  {dataset_type}:")
                    print(f"    样本数量: {stats.get('total_samples', 'N/A')}")
                    print(f"    平均chunks数: {stats.get('avg_chunks', 'N/A'):.2f}" if isinstance(stats.get('avg_chunks'), (int, float)) else f"    平均chunks数: {stats.get('avg_chunks', 'N/A')}")
                    print(f"    平均正样本数: {stats.get('avg_positive_chunks', 'N/A'):.2f}" if isinstance(stats.get('avg_positive_chunks'), (int, float)) else f"    平均正样本数: {stats.get('avg_positive_chunks', 'N/A')}")
                    print(f"    平均负样本数: {stats.get('avg_negative_chunks', 'N/A'):.2f}" if isinstance(stats.get('avg_negative_chunks'), (int, float)) else f"    平均负样本数: {stats.get('avg_negative_chunks', 'N/A')}")
                    print(f"    平均强负样本数: {stats.get('avg_hard_negative_chunks', 'N/A'):.2f}" if isinstance(stats.get('avg_hard_negative_chunks'), (int, float)) else f"    平均强负样本数: {stats.get('avg_hard_negative_chunks', 'N/A')}")
                    print(f"    总正样本数: {stats.get('total_positive', 'N/A')}")
                    print(f"    总负样本数: {stats.get('total_negative', 'N/A')}")
                    print(f"    总强负样本数: {stats.get('total_hard_negative', 'N/A')}")
                    if 'discourse_relations' in stats:
                        print(f"    话语关系分布: {stats['discourse_relations']}")
        
        print("-" * 100)
        print(f"{'类型':<15} {'指标':<20} {'@3':<12} {'@5':<12} {'@10':<12}")
        print("-" * 100)
        
        for dataset_type in ['test', 'modfied_test']:
            if dataset_type in dataset_results:
                metrics = dataset_results[dataset_type]
                
                # 正样本召回率
                row = f"{dataset_type:<15} {'正样本召回率':<20}"
                for k in ['@3', '@5', '@10']:
                    value = metrics[k]['positive_recall']
                    row += f"{value:<12.4f}"
                print(row)
                
                # 正样本精确率
                row = f"{'':15} {'正样本精确率':<20}"
                for k in ['@3', '@5', '@10']:
                    value = metrics[k]['positive_precision']
                    row += f"{value:<12.4f}"
                print(row)
                
                # 强负样本干扰率
                row = f"{'':15} {'强负样本干扰率':<20}"
                for k in ['@3', '@5', '@10']:
                    value = metrics[k]['hard_negative_interference']
                    row += f"{value:<12.4f}"
                print(row)
                
                # 强负样本召回率
                row = f"{'':15} {'强负样本召回率':<20}"
                for k in ['@3', '@5', '@10']:
                    value = metrics[k]['hard_negative_recall']
                    row += f"{value:<12.4f}"
                print(row)
                
                # 平均正样本数
                row = f"{'':15} {'平均正样本数':<20}"
                for k in ['@3', '@5', '@10']:
                    value = metrics[k]['avg_total_positive']
                    row += f"{value:<12.2f}"
                print(row)
                
                # 平均强负样本数
                row = f"{'':15} {'平均强负样本数':<20}"
                for k in ['@3', '@5', '@10']:
                    value = metrics[k]['avg_total_hard_negative']
                    row += f"{value:<12.2f}"
                print(row)
                
                print()
                
                # 强负样本干扰分析详细信息
                if dataset_type == 'modfied_test':  # 只对modified_test显示详细分析
                    print(f"{'':15} 强负样本干扰分析:")
                    
                    # 干扰率
                    row = f"{'':15} {'  干扰率':<20}"
                    for k in ['@3', '@5', '@10']:
                        value = metrics[k]['hard_negative_interference_rate']
                        row += f"{value:<12.4f}"
                    print(row)
                    
                    # 平均强负样本排名
                    row = f"{'':15} {'  平均强负排名':<20}"
                    for k in ['@3', '@5', '@10']:
                        value = metrics[k]['avg_hard_negative_rank']
                        row += f"{value:<12.2f}"
                    print(row)
                    
                    # 理论召回率
                    row = f"{'':15} {'  理论召回率':<20}"
                    for k in ['@3', '@5', '@10']:
                        value = metrics[k]['theoretical_positive_recall']
                        row += f"{value:<12.4f}"
                    print(row)
                    
                    # 实际召回率
                    row = f"{'':15} {'  实际召回率':<20}"
                    for k in ['@3', '@5', '@10']:
                        value = metrics[k]['actual_positive_recall']
                        row += f"{value:<12.4f}"
                    print(row)
                    
                    # 召回率损失
                    row = f"{'':15} {'  召回率损失':<20}"
                    for k in ['@3', '@5', '@10']:
                        value = metrics[k]['recall_loss_due_to_hard_negatives']
                        row += f"{value:<12.4f}"
                    print(row)
                    
                    # 强负样本排在正样本前的数量
                    row = f"{'':15} {'  强负超越正样本':<20}"
                    for k in ['@3', '@5', '@10']:
                        value = metrics[k]['hard_negatives_above_positives']
                        row += f"{value:<12.2f}"
                    print(row)
                
                print("-" * 100)

def main():
    parser = argparse.ArgumentParser(description='评估检索系统性能')
    parser.add_argument('--model_path', type=str, default="/share/home/ecnuzwx/CoEn-RAG/2_TrainModels/save/BAAI--bge-m3/single_stage_seed_0_lr_1e-05_epoch_3_alpha_0.3/final_model/query_encoder", help='训练好的模型路径')
    parser.add_argument('--tasks_dir', type=str, default='./tasks', help='任务数据集目录')
    parser.add_argument('--datasets', nargs='+', default=['hotpotqa', 'msmarco', 'musique'], help='要评估的数据集')
    parser.add_argument('--batch_size', type=int, default=160, help='评估时的批次大小')
    
    args = parser.parse_args()
    
    # Initialize model
    print("Initializing model...")
    model = DualEncoderModel(model_name="/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3")
    
    # If model path is provided, load the trained model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model: {args.model_path}")
        model.load(args.model_path)
    else:
        print("Using pre-trained model (not fine-tuned)")
    
    print("Starting retrieval system performance evaluation")
    
    # Evaluate all datasets
    results = {}
    dataset_stats = {}
    
    for dataset_name in args.datasets:
        print(f"\nStarting dataset evaluation: {dataset_name}")
        dataset_results = {}
        dataset_stats[dataset_name] = {}
        
        # 评估两种数据集类型
        for dataset_type in ['test', 'modfied_test']:
            dataset_path = os.path.join(args.tasks_dir, dataset_name, dataset_type)
            if not os.path.exists(dataset_path):
                print(f"警告: 数据集路径不存在: {dataset_path}")
                continue
            
            # 加载数据集
            dataset = load_from_disk(dataset_path)
            print(f"成功加载数据集: {dataset_name}/{dataset_type}")
            
            # 收集数据集统计信息
            stats = collect_dataset_stats(dataset, dataset_name, dataset_type)
            dataset_stats[dataset_name][dataset_type] = stats
            
            # Evaluate dataset (without saving prediction results)
            type_results, _ = evaluate_dataset(model, dataset, f"{dataset_name}_{dataset_type}", batch_size=args.batch_size)
            dataset_results[dataset_type] = type_results
        
        if dataset_results:
            results[dataset_name] = dataset_results
    
    # Print results
    if results:
        print_results(results, dataset_stats)
    else:
        print("No datasets were successfully evaluated")

if __name__ == '__main__':
    main()