#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索系统评估脚本
计算@1、@3、@5的精确率、召回率和F1分数
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
    分析强负样本对检索结果的干扰影响
    
    Args:
        similarities: 相似度分数 [num_chunks]
        labels: 真实标签 [num_chunks] (1: 正样本, 0: 负样本, -1: 强负样本)
        k_values: 要分析的k值列表
    
    Returns:
        包含各k值下强负样本干扰分析的字典
    """
    analysis = {}
    
    # 获取各类样本的索引
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    hard_negative_indices = [i for i, label in enumerate(labels) if label == -1]
    
    # 获取排序后的索引
    _, sorted_indices = torch.sort(similarities, descending=True)
    sorted_indices = sorted_indices.cpu().tolist()
    
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        
        # 分析强负样本的排名情况
        hard_negative_ranks = []
        for hn_idx in hard_negative_indices:
            if hn_idx in sorted_indices:
                rank = sorted_indices.index(hn_idx) + 1  # 1-based ranking
                hard_negative_ranks.append(rank)
        
        # 分析正样本被强负样本挤出top-k的情况
        positive_in_topk = sum(1 for idx in top_k_indices if labels[idx] == 1)
        hard_negative_in_topk = sum(1 for idx in top_k_indices if labels[idx] == -1)
        
        # 计算如果没有强负样本，正样本的理论召回率
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
    计算@k的召回率，区分正样本(1)、负样本(0)和强负样本(-1)
    
    Args:
        similarities: 相似度分数 [num_chunks]
        labels: 真实标签 [num_chunks] (1: 正样本, 0: 负样本, -1: 强负样本)
        k: top-k
    
    Returns:
        包含各类样本召回率和干扰分析的字典
    """
    # 获取top-k索引
    _, top_k_indices = torch.topk(similarities, min(k, len(similarities)), largest=True)
    top_k_indices = top_k_indices.cpu().numpy()
    
    # 统计各类样本数量
    positive_count = sum(1 for label in labels if label == 1)  # 正样本
    negative_count = sum(1 for label in labels if label == 0)  # 负样本
    hard_negative_count = sum(1 for label in labels if label == -1)  # 强负样本
    
    # 统计top-k中各类样本的数量
    retrieved_positive = sum(1 for i in top_k_indices if labels[i] == 1)
    retrieved_negative = sum(1 for i in top_k_indices if labels[i] == 0)
    retrieved_hard_negative = sum(1 for i in top_k_indices if labels[i] == -1)
    
    # 计算召回率
    positive_recall = retrieved_positive / positive_count if positive_count > 0 else 0.0
    negative_recall = retrieved_negative / negative_count if negative_count > 0 else 0.0
    hard_negative_recall = retrieved_hard_negative / hard_negative_count if hard_negative_count > 0 else 0.0
    
    # 计算干扰分析指标
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
    评估单个数据集（批处理版本）
    
    Args:
        model: 双编码器模型
        dataset: 数据集
        dataset_name: 数据集名称
        batch_size: 批次大小
    
    Returns:
        包含各个k值指标的字典和预测结果列表
    """
    print(f"\n评估数据集: {dataset_name}")
    print(f"样本数量: {len(dataset)}")
    
    model.eval()
    
    # 存储所有样本的指标
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
    
    # 存储强负样本干扰分析
    all_hard_negative_analysis = {k: {
        'hard_negative_interference_rate': [],
        'avg_hard_negative_rank': [],
        'theoretical_positive_recall': [],
        'actual_positive_recall': [],
        'recall_loss_due_to_hard_negatives': [],
        'hard_negatives_above_positives': []
    } for k in [3, 5, 10]}
    
    # 存储预测结果
    predictions = []
    
    with torch.no_grad():
        # 创建批次索引
        total_samples = len(dataset)
        batch_indices = [(i, min(i + batch_size, total_samples)) for i in range(0, total_samples, batch_size)]
        
        # 批处理评估
        for start_idx, end_idx in tqdm(batch_indices, desc=f"评估{dataset_name}"):
            # 收集当前批次的样本
            batch_queries = []
            batch_chunks = []
            batch_labels = []
            
            # 获取当前批次的样本
            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                batch_queries.append(sample['query'])
                batch_chunks.append(sample['chunks'])
                batch_labels.append(sample['labels'])
            
            # 批量获取相似度分数
            similarities_list = model.forward(batch_queries, batch_chunks)
            
            # 处理每个样本的结果
            for j, (similarities, query, chunks, labels) in enumerate(zip(similarities_list, batch_queries, batch_chunks, batch_labels)):
                sample_idx = start_idx + j
                
                # 获取排序后的索引
                _, sorted_indices = torch.sort(similarities, descending=True)
                sorted_indices = sorted_indices.cpu().tolist()
                
                # 构建预测结果字典
                sample_data = dataset[sample_idx]
                prediction = {
                    'id': sample_idx,
                    'query': query,
                    'chunks': [chunks[idx] for idx in sorted_indices[:10]],  # 只保存前10个结果
                    'similarities': [similarities[idx].item() for idx in sorted_indices[:10]],  # 只保存前10个相似度分数
                    'sorted_indices': sorted_indices[:10],  # 只保存前10个索引
                    'labels': [labels[idx] for idx in sorted_indices[:10]],  # 对应的标签
                    'original_labels': labels,
                    'original_chunks': chunks  # 保存原始的chunks列表
                }
                
                # 添加新数据集格式中的额外字段（如果存在）
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
                
                # 计算各个k值的指标
                for k in [3, 5, 10]:
                    metrics = calculate_metrics_at_k(similarities, labels, k)
                    for metric_name, value in metrics.items():
                        all_metrics[k][metric_name].append(value)
                    
                    # 计算强负样本干扰分析
                    hard_negative_analysis = analyze_hard_negative_impact(similarities, labels, [k])
                    for analysis_name, value in hard_negative_analysis[f'@{k}'].items():
                        all_hard_negative_analysis[k][analysis_name].append(value)
    
    # 计算平均指标
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
            # 强负样本干扰分析
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
    收集数据集统计信息
    
    Args:
        dataset: 数据集
        dataset_name: 数据集名称
        dataset_type: 数据集类型
    
    Returns:
        包含统计信息的字典
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
        
        # 收集话语关系统计
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
    保存预测结果和标准答案对到JSON文件
    
    Args:
        predictions: 预测结果列表
        dataset_name: 数据集名称
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"{dataset_name}_predictions.json")
    
    # 准备可序列化的数据，重点突出预测结果和标准答案对
    serializable_predictions = []
    for pred in predictions:
        # 深拷贝预测结果，避免修改原始数据
        serializable_pred = dict(pred)
        # 移除不可序列化的数据
        if 'similarities' in serializable_pred:
            serializable_pred['similarities'] = [float(s) for s in serializable_pred['similarities']]
        
        # 确保original_chunks是可序列化的
        if 'original_chunks' in serializable_pred:
            # 如果original_chunks中的元素不是简单的字符串，可能需要进行转换
            serializable_pred['original_chunks'] = [str(chunk) for chunk in serializable_pred['original_chunks']]
        
        # 重新组织数据，突出预测结果和标准答案对
        prediction_answer_pairs = []
        for i in range(len(serializable_pred['chunks'])):
            # 获取原始索引
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
        
        # 更新预测结果字典
        serializable_pred['prediction_answer_pairs'] = prediction_answer_pairs
        serializable_pred['query'] = pred['query']
        
        # 保留新数据集格式中的额外字段
        for field in ['original_query', 'used_distractors', 'discourse_relation', 'answer', 'uuid']:
            if field in pred:
                serializable_pred[field] = pred[field]
        
        # 添加标准答案信息
        ground_truth_chunks = []
        for idx, label in enumerate(pred['original_labels']):
            if label == 1:  # 只有正样本才是ground truth
                if 'original_chunks' in pred and idx < len(pred['original_chunks']):
                    ground_truth_chunks.append(pred['original_chunks'][idx])
        
        serializable_pred['ground_truth_chunks'] = ground_truth_chunks
        serializable_pred['original_labels'] = [int(label) for label in pred['original_labels']]
        
        serializable_predictions.append(serializable_pred)
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"预测结果和标准答案对已保存到: {output_file}")


def print_results(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], dataset_stats: Dict[str, Dict[str, Dict]] = None):
    """
    打印评估结果
    """
    print("\n" + "="*80)
    print("检索系统评估结果")
    print("="*80)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n数据集: {dataset_name}")
        
        # 显示数据集统计信息
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
    
    # 初始化模型
    print("初始化模型...")
    model = DualEncoderModel(model_name="/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3")
    
    # 如果提供了模型路径，加载训练好的模型
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载模型: {args.model_path}")
        model.load(args.model_path)
    else:
        print("使用预训练模型（未经过微调）")
    
    print("开始评估检索系统性能")
    
    # 评估所有数据集
    results = {}
    dataset_stats = {}
    
    for dataset_name in args.datasets:
        print(f"\n开始评估数据集: {dataset_name}")
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
            
            # 评估数据集（不保存预测结果）
            type_results, _ = evaluate_dataset(model, dataset, f"{dataset_name}_{dataset_type}", batch_size=args.batch_size)
            dataset_results[dataset_type] = type_results
        
        if dataset_results:
            results[dataset_name] = dataset_results
    
    # 打印结果
    if results:
        print_results(results, dataset_stats)
    else:
        print("没有成功评估任何数据集")

if __name__ == '__main__':
    main()