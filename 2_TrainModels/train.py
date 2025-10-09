#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LORE (Logic-ORiented Retriever Enhancement) Training Script

This script implements the contrastive learning training methodology described in the paper:
"Logic-ORiented Retriever Enhancement via Contrastive Learning"

Key Features:
- Three-tier contrastive learning with fine-grained sample classification:
  * P (Positive, label=1): Chunks sufficient to answer the query
  * N1 (Distractor, label=-1): Chunks used by LLM in query rewriting, seemingly relevant but unhelpful
  * N2 (Negative, label=0): Other unused negative chunks
- Dual encoder architecture: frozen document encoder M_d and trainable query encoder M_q
- InfoNCE-based loss with differentiated weights (β > α) for hierarchical separation P ≻ N1 ≻ N2
- Temperature scaling and logit-space weighting for enhanced discrimination
"""

import os
# Set environment variable to avoid tokenizer parallelization warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from models import DualEncoderModel
import matplotlib.pyplot as plt
from utils.unified_monitor import UnifiedMonitor
from transformers import Trainer, TrainingArguments, TrainerCallback
import json
import argparse

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class ContrastiveDataset(Dataset):
    """
    Contrastive learning dataset: adapts to the current dataset structure
    Using labels field directly as tier_labels
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        query = item['query']
        chunks = item['chunks']
        labels = item['labels']
        
        
        return {
            'question': query,  # Keep 'question' as key for compatibility with existing code
            'chunks': chunks,
            'tier_labels': torch.tensor(labels, dtype=torch.float32)
        }


def collate_fn(batch):
    """
    Batch processing function
    """
    questions = [item['question'] for item in batch]
    all_chunks = []
    all_labels = []
    
    for item in batch:
        chunks = item['chunks']
        labels = item['tier_labels']
        
        all_chunks.append(chunks)  # Maintain nested structure
        all_labels.append(labels)
    
    return {
        'questions': questions,
        'chunks': all_chunks,
        'labels': all_labels
    }


class ContrastiveTrainer(Trainer):
    """Custom Trainer to adapt to current model's forward signature and compute loss"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarities_callback = None
    
    def set_similarities_callback(self, callback):
        self.similarities_callback = callback
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        questions = inputs['questions']
        chunks = inputs['chunks']
        labels = inputs.get('labels', None)
        outputs = model.forward(questions, chunks, labels)
        if labels is not None:
            similarities_list, loss = outputs
            # Record similarities data to callback
            if self.similarities_callback is not None:
                self.similarities_callback.record_similarities(similarities_list, labels, is_training=model.training)
        else:
            similarities_list = outputs
            loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        return (loss, similarities_list) if return_outputs else loss


class DualEncoderSaveCallback(TrainerCallback):
    """Save model using its built-in save interface at the end of each epoch, maintaining original saving habits"""
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch_idx = int(state.epoch) - 1 if state.epoch is not None else 0
        save_path = os.path.join(args.output_dir, f'model_epoch_{epoch_idx}')
        model.save(save_path)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        final_path = os.path.join(args.output_dir, 'final_model')
        model.save(final_path)


class TrainingCurvePlotCallback(TrainerCallback):
    """Record and plot training/evaluation loss curves and similarities curves, save data in JSON format"""
    def __init__(self, output_name: str = 'training_curves.png', json_name: str = 'training_data.json', window_size: int = 10):
        # Loss data
        self.train_steps = []
        self.train_losses = []
        self.eval_steps = []
        self.eval_losses = []
        
        # Similarities data - recorded by category, including average, max, min values
        self.train_similarities_pos = {'avg': [], 'max': [], 'min': []}  # P (Positive, label=1) similarity stats
        self.train_similarities_n2 = {'avg': [], 'max': [], 'min': []}   # N2 (Negative, label=0) similarity stats
        self.train_similarities_n1 = {'avg': [], 'max': [], 'min': []}   # N1 (Distractor, label=-1) similarity stats
        self.eval_similarities_pos = {'avg': [], 'max': [], 'min': []}
        self.eval_similarities_n2 = {'avg': [], 'max': [], 'min': []}
        self.eval_similarities_n1 = {'avg': [], 'max': [], 'min': []}
        
        # Temporary storage for computing sliding averages
        self.temp_train_sims_pos = []
        self.temp_train_sims_n2 = []
        self.temp_train_sims_n1 = []
        self.temp_eval_sims_pos = []
        self.temp_eval_sims_n2 = []
        self.temp_eval_sims_n1 = []
        
        self.output_name = output_name
        self.json_name = json_name
        self.window_size = window_size
        self.current_step = 0
    
    def record_similarities(self, similarities_list, labels_list, is_training=True):
        """Record similarities data, classified by P/N1/N2"""
        try:
            # Process similarities and labels for each batch
            for similarities, labels in zip(similarities_list, labels_list):
                # similarities: [num_chunks], labels: [num_chunks]
                similarities = similarities.detach().cpu().numpy() if hasattr(similarities, 'detach') else similarities
                labels = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else labels
                
                # Collect similarities by label classification
                pos_sims = similarities[labels == 1]   # P (Positive, label=1)
                n2_sims = similarities[labels == 0]    # N2 (Negative, label=0)
                n1_sims = similarities[labels == -1]   # N1 (Distractor, label=-1)
                
                if is_training:
                    self.temp_train_sims_pos.extend(pos_sims.tolist())
                    self.temp_train_sims_n2.extend(n2_sims.tolist())
                    self.temp_train_sims_n1.extend(n1_sims.tolist())
                else:
                    self.temp_eval_sims_pos.extend(pos_sims.tolist())
                    self.temp_eval_sims_n2.extend(n2_sims.tolist())
                    self.temp_eval_sims_n1.extend(n1_sims.tolist())
        except Exception as e:
            print(f"Error recording similarities: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if 'loss' in logs and state.global_step is not None and state.global_step > 0:
            self.train_steps.append(state.global_step)
            self.train_losses.append(float(logs['loss']))
            
            # Calculate and record training similarities statistics (sliding window)
            if self.temp_train_sims_pos:
                window_data = self.temp_train_sims_pos[-self.window_size:]
                self.train_similarities_pos['avg'].append(np.mean(window_data))
                self.train_similarities_pos['max'].append(np.max(window_data))
                self.train_similarities_pos['min'].append(np.min(window_data))
            else:
                self.train_similarities_pos['avg'].append(0.0)
                self.train_similarities_pos['max'].append(0.0)
                self.train_similarities_pos['min'].append(0.0)
                
            if self.temp_train_sims_n2:
                window_data = self.temp_train_sims_n2[-self.window_size:]
                self.train_similarities_n2['avg'].append(np.mean(window_data))
                self.train_similarities_n2['max'].append(np.max(window_data))
                self.train_similarities_n2['min'].append(np.min(window_data))
            else:
                self.train_similarities_n2['avg'].append(0.0)
                self.train_similarities_n2['max'].append(0.0)
                self.train_similarities_n2['min'].append(0.0)
                
            if self.temp_train_sims_n1:
                window_data = self.temp_train_sims_n1[-self.window_size:]
                self.train_similarities_n1['avg'].append(np.mean(window_data))
                self.train_similarities_n1['max'].append(np.max(window_data))
                self.train_similarities_n1['min'].append(np.min(window_data))
            else:
                self.train_similarities_n1['avg'].append(0.0)
                self.train_similarities_n1['max'].append(0.0)
                self.train_similarities_n1['min'].append(0.0)
            
            self._plot_and_save(args)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if 'eval_loss' in metrics and state.global_step is not None and state.global_step > 0:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(float(metrics['eval_loss']))
            
            # Calculate and record evaluation similarities statistics
            if self.temp_eval_sims_pos:
                self.eval_similarities_pos['avg'].append(np.mean(self.temp_eval_sims_pos))
                self.eval_similarities_pos['max'].append(np.max(self.temp_eval_sims_pos))
                self.eval_similarities_pos['min'].append(np.min(self.temp_eval_sims_pos))
                self.temp_eval_sims_pos.clear()  # Clear temporary data
            else:
                self.eval_similarities_pos['avg'].append(0.0)
                self.eval_similarities_pos['max'].append(0.0)
                self.eval_similarities_pos['min'].append(0.0)
                
            if self.temp_eval_sims_n2:
                self.eval_similarities_n2['avg'].append(np.mean(self.temp_eval_sims_n2))
                self.eval_similarities_n2['max'].append(np.max(self.temp_eval_sims_n2))
                self.eval_similarities_n2['min'].append(np.min(self.temp_eval_sims_n2))
                self.temp_eval_sims_n2.clear()
            else:
                self.eval_similarities_n2['avg'].append(0.0)
                self.eval_similarities_n2['max'].append(0.0)
                self.eval_similarities_n2['min'].append(0.0)
                
            if self.temp_eval_sims_n1:
                self.eval_similarities_n1['avg'].append(np.mean(self.temp_eval_sims_n1))
                self.eval_similarities_n1['max'].append(np.max(self.temp_eval_sims_n1))
                self.eval_similarities_n1['min'].append(np.min(self.temp_eval_sims_n1))
                self.temp_eval_sims_n1.clear()
            else:
                self.eval_similarities_n1['avg'].append(0.0)
                self.eval_similarities_n1['max'].append(0.0)
                self.eval_similarities_n1['min'].append(0.0)
            
            self._plot_and_save(args)

    def _plot_and_save(self, args):
        """Plot curves and save data"""
        if not self.train_steps and not self.eval_steps:
            return
        
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot loss curves
            if self.train_steps:
                ax1.plot(self.train_steps, self.train_losses, label='Train Loss', color='blue')
            if self.eval_steps:
                ax1.plot(self.eval_steps, self.eval_losses, label='Eval Loss', color='red')
            ax1.set_xlabel('Global Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Evaluation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot similarities curves - average as main line, max/min as upper/lower bounds
            if self.train_steps and len(self.train_similarities_pos['avg']) == len(self.train_steps):
                # P (Positive) - green
                ax2.plot(self.train_steps, self.train_similarities_pos['avg'], label='Train P (Positive, label=1)', color='green', linestyle='-', linewidth=2)
                ax2.fill_between(self.train_steps, self.train_similarities_pos['min'], self.train_similarities_pos['max'], 
                               color='green', alpha=0.2, label='Train P Range')
                
                # N2 (Negative) - orange
                ax2.plot(self.train_steps, self.train_similarities_n2['avg'], label='Train N2 (Negative, label=0)', color='orange', linestyle='-', linewidth=2)
                ax2.fill_between(self.train_steps, self.train_similarities_n2['min'], self.train_similarities_n2['max'], 
                               color='orange', alpha=0.2, label='Train N2 Range')
                
                # N1 (Distractor) - purple
                ax2.plot(self.train_steps, self.train_similarities_n1['avg'], label='Train N1 (Distractor, label=-1)', color='purple', linestyle='-', linewidth=2)
                ax2.fill_between(self.train_steps, self.train_similarities_n1['min'], self.train_similarities_n1['max'], 
                               color='purple', alpha=0.2, label='Train N1 Range')
            
            if self.eval_steps and len(self.eval_similarities_pos['avg']) == len(self.eval_steps):
                # P (Positive) - green dashed line
                ax2.plot(self.eval_steps, self.eval_similarities_pos['avg'], label='Eval P (Positive, label=1)', color='green', linestyle='--', linewidth=2)
                ax2.fill_between(self.eval_steps, self.eval_similarities_pos['min'], self.eval_similarities_pos['max'], 
                               color='green', alpha=0.1, label='Eval P Range')
                
                # N2 (Negative) - orange dashed line
                ax2.plot(self.eval_steps, self.eval_similarities_n2['avg'], label='Eval N2 (Negative, label=0)', color='orange', linestyle='--', linewidth=2)
                ax2.fill_between(self.eval_steps, self.eval_similarities_n2['min'], self.eval_similarities_n2['max'], 
                               color='orange', alpha=0.1, label='Eval N2 Range')
                
                # N1 (Distractor) - purple dashed line
                ax2.plot(self.eval_steps, self.eval_similarities_n1['avg'], label='Eval N1 (Distractor, label=-1)', color='purple', linestyle='--', linewidth=2)
                ax2.fill_between(self.eval_steps, self.eval_similarities_n1['min'], self.eval_similarities_n1['max'], 
                               color='purple', alpha=0.1, label='Eval N1 Range')
            
            ax2.set_xlabel('Global Step')
            ax2.set_ylabel('Similarity')
            ax2.set_title('Similarities by Category: P (Positive) / N1 (Distractor) / N2 (Negative)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, self.output_name), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save data as JSON
            data = {
                'train_data': {
                    'steps': self.train_steps,
                    'losses': self.train_losses,
                    'similarities_pos': self.train_similarities_pos,
                    'similarities_n2': self.train_similarities_n2,
                    'similarities_n1': self.train_similarities_n1
                },
                'eval_data': {
                    'steps': self.eval_steps,
                    'losses': self.eval_losses,
                    'similarities_pos': self.eval_similarities_pos,
                    'similarities_n2': self.eval_similarities_n2,
                    'similarities_n1': self.eval_similarities_n1
                },
                'metadata': {
                    'window_size': self.window_size,
                    'description': 'Training curves data with loss and similarities by category following paper taxonomy',
                    'label_definition': 'P (Positive, label=1): Chunks answering query | N1 (Distractor, label=-1): LLM-used chunks for rewriting | N2 (Negative, label=0): Other negative chunks',
                    'similarities_format': 'Each similarity category contains avg, max, min statistics',
                    'plot_strategy': 'Plots average values as main lines with max/min values as shaded ranges for all 3 categories'
                }
            }
            
            with open(os.path.join(args.output_dir, self.json_name), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error during plotting or saving: {e}")
            try:
                plt.close()
            except Exception:
                pass


def load_train_eval_dataset(dataset_root: str, train_dir: str = None, eval_dir: str = None):
    """
    Load train/evaluate datasets flexibly:
    - If train_dir and eval_dir are provided, load them separately
    - Else try to load a DatasetDict saved at dataset_root
    - Else try to load Dataset from dataset_root/train and dataset_root/evaluate
    """
    from datasets import load_from_disk
    if train_dir and eval_dir:
        train_ds = load_from_disk(train_dir)
        eval_ds = load_from_disk(eval_dir)
        return {'train': train_ds, 'evaluate': eval_ds}
    # try load as DatasetDict
    loaded = load_from_disk(dataset_root)
    if hasattr(loaded, 'keys') and 'train' in loaded and 'evaluate' in loaded:
        return loaded
    # fallback to subdirs
    train_path = os.path.join(dataset_root, 'train')
    eval_path = os.path.join(dataset_root, 'evaluate')
    train_ds = load_from_disk(train_path)
    eval_ds = load_from_disk(eval_path)
    return {'train': train_ds, 'evaluate': eval_ds}


def main():
    parser = argparse.ArgumentParser(description='CoEn-RAG Training')
    # data
    parser.add_argument('--dataset_root', type=str, default=os.environ.get('COEN_DATA_DIR', './data'), help='Root directory for dataset')
    parser.add_argument('--train_dir', type=str, default=None, help='Optional path to train split directory saved via save_to_disk')
    parser.add_argument('--eval_dir', type=str, default=None, help='Optional path to evaluate split directory saved via save_to_disk')
    # model
    parser.add_argument('--base_model', type=str, default=os.environ.get('COEN_BASE_MODEL', 'BAAI/bge-m3'), help='Backbone model name or local path for SentenceTransformer')
    # hyperparams
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.3, help='weight for N2 (Negative, label=0)')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for N1 (Distractor, label=-1)')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory to save checkpoints and final model')

    args = parser.parse_args()
    
    # Initialize unified monitor
    unified_monitor = UnifiedMonitor(device_id=args.gpu_id)
    
    # Setup monitor environment
    unified_monitor.setup()
    
    # Device info only for printing (migration handled automatically by Trainer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    model_name_safe = args.base_model.replace('/', '--')
    default_save = f'save/{model_name_safe}/Seed_{seed}_lr_{args.learning_rate}_epoch_{args.num_epochs}_temp_{args.temperature}_alpha_{args.alpha}_beta_{args.beta}'
    save_dir = args.save_dir or default_save
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_train_eval_dataset(args.dataset_root, args.train_dir, args.eval_dir)
    
    # Print dataset structure
    print("\n=== Dataset Structure ===")
    for split in dataset.keys():
        print(f"Split: {split}")
        print(f"Features: {dataset[split].features}")
        print(f"Num examples: {len(dataset[split])}")
    
    # Train/validation sets (directly use ContrastiveDataset)
    train_dataset = ContrastiveDataset(dataset['train'])
    eval_dataset = ContrastiveDataset(dataset['evaluate'])
    
    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Validation set size: {len(eval_dataset)}")
    
    print('Starting to load model...')
    # Create model (device and mode managed by Trainer, no manual to/device and train here)
    model = DualEncoderModel(args.base_model, temperature=args.temperature, alpha=args.alpha, beta=args.beta)
    print('Model loading completed')
    
    # Parameter statistics
    print("\n=== Parameter Statistics ===")
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"Total parameters: {frozen_params + trainable_params}")
    print(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters ratio: {frozen_params/(frozen_params + trainable_params)*100:.1f}%")
    
    # Single-stage training
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=1/3,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='epoch',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=2,
        report_to='none',
        seed=seed,
        remove_unused_columns=False
    )
    
    # Create callback instance
    plot_callback = TrainingCurvePlotCallback()
    
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn
    )
    
    # Set callback connection
    trainer.set_similarities_callback(plot_callback)
    
    # Add callbacks
    trainer.add_callback(DualEncoderSaveCallback())
    trainer.add_callback(plot_callback)
    
    print("Starting single-stage training (using ContrastiveDataset with gradient accumulation)...")
    trainer.train()
    
    print("Single-stage training completed!")
    print(f"Model saved to: {save_dir}")
    
    # Usage instructions
    print("\n=== Model Usage Instructions ===")
    print(f"from models import DualEncoderModel")
    print(f"model = DualEncoderModel()")
    print(f"model.load('{os.path.join(save_dir, f'model_epoch_{args.num_epochs-1}')}')")
    print(f"similarities = model.forward('your query', ['doc1', 'doc2'])")


if __name__ == '__main__':
    main()


