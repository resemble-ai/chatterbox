import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Import Chatterbox components
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.t3.modules.cond_enc import T3Cond

# Add matplotlib imports for metrics tracking
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime
import threading
import time
from collections import deque

# Hardcoded configuration
AUDIO_DATA_DIR = "./audio_data"
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 2e-5  
WARMUP_STEPS = 500 
MAX_AUDIO_LENGTH = 400.0  
MIN_AUDIO_LENGTH = 1.0
LORA_RANK = 32  
LORA_ALPHA = 64  
LORA_DROPOUT = 0.05  
GRADIENT_ACCUMULATION_STEPS = 8
SAVE_EVERY_N_STEPS = 200
CHECKPOINT_DIR = "checkpoints_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
MAX_TEXT_LENGTH = 1000
VALIDATION_SPLIT = 0.1

# Metrics tracking class
class MetricsTracker:
    def __init__(self, save_path="training_metrics.png", update_interval=2.0):
        self.save_path = save_path
        self.update_interval = update_interval
        self.metrics = {
            'train_loss': deque(maxlen=1000),
            'val_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=1000),
            'steps': deque(maxlen=1000),
            'epochs': deque(maxlen=1000),
            'batch_loss': deque(maxlen=100),  # Recent batch losses
            'gradient_norm': deque(maxlen=1000),
            'loss_variance': deque(maxlen=100),
            'time_per_step': deque(maxlen=100),
        }
        self.start_time = time.time()
        self.last_update = 0
        self.running = True
        self.lock = threading.Lock()
        
        # Initialize plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Chatterbox TTS LoRA Training Metrics', fontsize=16, fontweight='bold')
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Create initial plot
        self._create_initial_plot()
    
    def _create_initial_plot(self):
        """Create the initial plot layout"""
        self.fig.clf()
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_lr = self.fig.add_subplot(gs[1, 0])
        self.ax_grad = self.fig.add_subplot(gs[1, 1])
        self.ax_batch = self.fig.add_subplot(gs[1, 2])
        self.ax_variance = self.fig.add_subplot(gs[2, 0])
        self.ax_time = self.fig.add_subplot(gs[2, 1])
        self.ax_info = self.fig.add_subplot(gs[0, 2])
        self.ax_epoch = self.fig.add_subplot(gs[2, 2])
        
        # Configure info panel
        self.ax_info.axis('off')
        
        # Set titles
        self.ax_loss.set_title('Training & Validation Loss', fontweight='bold')
        self.ax_lr.set_title('Learning Rate', fontweight='bold')
        self.ax_grad.set_title('Gradient Norm', fontweight='bold')
        self.ax_batch.set_title('Recent Batch Losses', fontweight='bold')
        self.ax_variance.set_title('Loss Variance (100 batches)', fontweight='bold')
        self.ax_time.set_title('Time per Step', fontweight='bold')
        self.ax_epoch.set_title('Loss by Epoch', fontweight='bold')
        
        # Set labels
        self.ax_loss.set_xlabel('Steps')
        self.ax_loss.set_ylabel('Loss')
        self.ax_lr.set_xlabel('Steps')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_grad.set_xlabel('Steps')
        self.ax_grad.set_ylabel('Gradient Norm')
        self.ax_batch.set_xlabel('Recent Batches')
        self.ax_batch.set_ylabel('Loss')
        self.ax_variance.set_xlabel('Steps')
        self.ax_variance.set_ylabel('Variance')
        self.ax_time.set_xlabel('Recent Steps')
        self.ax_time.set_ylabel('Seconds')
        self.ax_epoch.set_xlabel('Epoch')
        self.ax_epoch.set_ylabel('Average Loss')
        
        # Enable grids
        for ax in [self.ax_loss, self.ax_lr, self.ax_grad, self.ax_batch, 
                   self.ax_variance, self.ax_time, self.ax_epoch]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
    
    def add_metrics(self, **kwargs):
        """Add metrics to the tracker"""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.metrics and value is not None:
                    self.metrics[key].append(value)
            self.last_update = time.time()
    
    def _update_loop(self):
        """Background thread to update plots"""
        while self.running:
            time.sleep(self.update_interval)
            if time.time() - self.last_update < self.update_interval * 2:
                self._update_plot()
    
    def _update_plot(self):
        """Update the plot with current metrics"""
        with self.lock:
            try:
                # Clear all axes
                for ax in [self.ax_loss, self.ax_lr, self.ax_grad, self.ax_batch, 
                          self.ax_variance, self.ax_time, self.ax_epoch]:
                    ax.clear()
                
                # Plot training loss
                if len(self.metrics['train_loss']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['train_loss']):]
                    self.ax_loss.plot(steps, list(self.metrics['train_loss']), 
                                     'b-', label='Train Loss', linewidth=2)
                    self.ax_loss.set_ylim(bottom=0)
                
                # Plot validation loss
                if len(self.metrics['val_loss']) > 0:
                    val_steps = list(self.metrics['steps'])[-len(self.metrics['val_loss']):]
                    self.ax_loss.plot(val_steps[-len(self.metrics['val_loss']):], 
                                     list(self.metrics['val_loss']), 
                                     'r-o', label='Val Loss', linewidth=2, markersize=8)
                
                self.ax_loss.legend()
                self.ax_loss.set_title('Training & Validation Loss', fontweight='bold')
                self.ax_loss.set_xlabel('Steps')
                self.ax_loss.set_ylabel('Loss')
                self.ax_loss.grid(True, alpha=0.3)
                
                # Plot learning rate
                if len(self.metrics['learning_rate']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['learning_rate']):]
                    self.ax_lr.plot(steps, list(self.metrics['learning_rate']), 
                                   'g-', linewidth=2)
                    self.ax_lr.set_title('Learning Rate', fontweight='bold')
                    self.ax_lr.set_xlabel('Steps')
                    self.ax_lr.set_ylabel('Learning Rate')
                    self.ax_lr.grid(True, alpha=0.3)
                    self.ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
                
                # Plot gradient norm
                if len(self.metrics['gradient_norm']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['gradient_norm']):]
                    self.ax_grad.plot(steps, list(self.metrics['gradient_norm']), 
                                     'm-', linewidth=2)
                    self.ax_grad.set_title('Gradient Norm', fontweight='bold')
                    self.ax_grad.set_xlabel('Steps')
                    self.ax_grad.set_ylabel('Gradient Norm')
                    self.ax_grad.grid(True, alpha=0.3)
                
                # Plot recent batch losses
                if len(self.metrics['batch_loss']) > 0:
                    recent_losses = list(self.metrics['batch_loss'])
                    self.ax_batch.plot(recent_losses, 'c-', linewidth=2)
                    self.ax_batch.axhline(y=np.mean(recent_losses), color='yellow', 
                                         linestyle='--', label=f'Mean: {np.mean(recent_losses):.4f}')
                    self.ax_batch.legend()
                    self.ax_batch.set_title('Recent Batch Losses', fontweight='bold')
                    self.ax_batch.set_xlabel('Recent Batches')
                    self.ax_batch.set_ylabel('Loss')
                    self.ax_batch.grid(True, alpha=0.3)
                
                # Plot loss variance
                if len(self.metrics['loss_variance']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['loss_variance']):]
                    self.ax_variance.plot(steps, list(self.metrics['loss_variance']), 
                                         'orange', linewidth=2)
                    self.ax_variance.set_title('Loss Variance (100 batches)', fontweight='bold')
                    self.ax_variance.set_xlabel('Steps')
                    self.ax_variance.set_ylabel('Variance')
                    self.ax_variance.grid(True, alpha=0.3)
                
                # Plot time per step
                if len(self.metrics['time_per_step']) > 0:
                    self.ax_time.plot(list(self.metrics['time_per_step']), 'lime', linewidth=2)
                    mean_time = np.mean(list(self.metrics['time_per_step']))
                    self.ax_time.axhline(y=mean_time, color='red', linestyle='--', 
                                        label=f'Mean: {mean_time:.2f}s')
                    self.ax_time.legend()
                    self.ax_time.set_title('Time per Step', fontweight='bold')
                    self.ax_time.set_xlabel('Recent Steps')
                    self.ax_time.set_ylabel('Seconds')
                    self.ax_time.grid(True, alpha=0.3)
                
                # Plot epoch-wise loss
                if len(self.metrics['epochs']) > 0 and len(self.metrics['train_loss']) > 0:
                    epoch_losses = {}
                    for epoch, loss in zip(self.metrics['epochs'], self.metrics['train_loss']):
                        if epoch not in epoch_losses:
                            epoch_losses[epoch] = []
                        epoch_losses[epoch].append(loss)
                    
                    epochs = sorted(epoch_losses.keys())
                    avg_losses = [np.mean(epoch_losses[e]) for e in epochs]
                    
                    self.ax_epoch.bar(epochs, avg_losses, color='skyblue', alpha=0.7)
                    self.ax_epoch.set_title('Loss by Epoch', fontweight='bold')
                    self.ax_epoch.set_xlabel('Epoch')
                    self.ax_epoch.set_ylabel('Average Loss')
                    self.ax_epoch.grid(True, alpha=0.3)
                
                # Update info panel
                self.ax_info.clear()
                self.ax_info.axis('off')
                
                info_text = [
                    f"Training Information",
                    f"{'='*25}",
                    f"Device: {DEVICE}",
                    f"Batch Size: {BATCH_SIZE}",
                    f"Grad Accum: {GRADIENT_ACCUMULATION_STEPS}",
                    f"LoRA Rank: {LORA_RANK}",
                    f"LoRA Alpha: {LORA_ALPHA}",
                    f"Learning Rate: {LEARNING_RATE:.2e}",
                    f"",
                    f"Current Stats",
                    f"{'='*25}",
                ]
                
                if len(self.metrics['steps']) > 0:
                    current_step = self.metrics['steps'][-1]
                    info_text.append(f"Step: {current_step}")
                
                if len(self.metrics['epochs']) > 0:
                    current_epoch = self.metrics['epochs'][-1]
                    info_text.append(f"Epoch: {current_epoch}/{EPOCHS}")
                
                if len(self.metrics['train_loss']) > 0:
                    current_loss = self.metrics['train_loss'][-1]
                    info_text.append(f"Current Loss: {current_loss:.4f}")
                
                if len(self.metrics['learning_rate']) > 0:
                    current_lr = self.metrics['learning_rate'][-1]
                    info_text.append(f"Current LR: {current_lr:.2e}")
                
                elapsed_time = time.time() - self.start_time
                info_text.append(f"")
                info_text.append(f"Time Elapsed: {elapsed_time/3600:.2f}h")
                
                if len(self.metrics['steps']) > 1:
                    steps_per_sec = len(self.metrics['steps']) / elapsed_time
                    eta = (len(self.metrics['steps']) / (self.metrics['epochs'][-1] + 1) * EPOCHS - len(self.metrics['steps'])) / steps_per_sec / 3600
                    info_text.append(f"ETA: {eta:.2f}h")
                
                self.ax_info.text(0.05, 0.95, '\n'.join(info_text), 
                                 transform=self.ax_info.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 fontfamily='monospace',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.fig.text(0.99, 0.01, f"Last updated: {timestamp}", 
                             ha='right', va='bottom', fontsize=8, color='gray')
                
                # Save figure
                self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
                
            except Exception as e:
                print(f"Error updating plot: {e}")
    
    def stop(self):
        """Stop the metrics tracker"""
        self.running = False
        self.update_thread.join()
        plt.close(self.fig)


class LoRALayer(nn.Module):
    """LoRA adapter layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Proper initialization
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/np.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.lora_dropout(x)
        result = result @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


def inject_lora_layers(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float):
    lora_layers = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                if min(module.in_features, module.out_features) < rank:
                    continue
                    
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                lora_layer = lora_layer.to(device)
                lora_layers[name] = lora_layer
                
                original_forward = module.forward
                def make_new_forward(orig_forward, lora):
                    def new_forward(x):
                        return orig_forward(x) + lora(x)
                    return new_forward
                
                module.forward = make_new_forward(original_forward, lora_layer)
    
    return lora_layers


@dataclass
class AudioSample:
    """Container for audio sample data"""
    audio_path: Path
    transcript: str
    duration: float
    sample_rate: int


class TTSDataset(Dataset):
    """Dataset handling"""
    def __init__(
        self,
        samples: List[AudioSample],
        tokenizer: EnTokenizer,
        s3_sr: int = S3_SR,
        s3gen_sr: int = S3GEN_SR,
        max_audio_length: float = MAX_AUDIO_LENGTH,
        max_text_length: int = MAX_TEXT_LENGTH,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.s3_sr = s3_sr
        self.s3gen_sr = s3gen_sr
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process audio - keep original length logic
        audio, sr = librosa.load(sample.audio_path, sr=self.s3gen_sr)
        
        audio = librosa.util.normalize(audio)
        
        # Keep original padding/trimming logic but improve it
        max_samples = int(self.max_audio_length * self.s3gen_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            pad_amount = max_samples - len(audio)
            audio = np.pad(audio, (0, pad_amount), mode='constant', constant_values=0)
        
        # Resample for S3 tokenizer
        audio_16k = librosa.resample(audio, orig_sr=self.s3gen_sr, target_sr=self.s3_sr)
        
        # Process text
        text = punc_norm(sample.transcript)
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return {
            'audio': torch.FloatTensor(audio),
            'audio_16k': torch.FloatTensor(audio_16k),
            'text': text,
            'audio_path': str(sample.audio_path),
        }


def prepare_batch_conditionals(
    batch: Dict[str, torch.Tensor],
    model: ChatterboxTTS,
    ve: VoiceEncoder,
    s3gen: S3Gen,
) -> Tuple[T3Cond, List[dict]]:
    B = batch['audio'].size(0)
    device = model.device

    ve_embeds = []
    for i in range(B):
        try:
            wav_16k = batch['audio_16k'][i].numpy()
            
            if len(wav_16k) < S3_SR:  # Less than 1 second
                wav_16k = np.pad(wav_16k, (0, S3_SR - len(wav_16k)), mode='reflect')
            
            utt_embeds = ve.embeds_from_wavs([wav_16k],
                                             sample_rate=S3_SR,
                                             as_spk=False,
                                             batch_size=8,
                                             rate=1.3,
                                             overlap=0.5)

            parts = torch.from_numpy(utt_embeds)
            ref = parts[0].unsqueeze(0)
            sims = F.cosine_similarity(parts, ref, dim=-1)
            voiced = parts[sims > 0.6]
            ve_embed = voiced.mean(0, keepdim=True) if len(voiced) else parts.mean(0, keepdim=True)
            ve_embeds.append(ve_embed)
        except Exception as e:
            print(f"Error in voice embedding {i}: {e}")
            if ve_embeds:
                ve_embed = ve_embeds[-1].clone()
            else:
                ve_embed = torch.zeros(1, 256)  # Typical VE embedding size
            ve_embeds.append(ve_embed)

    ve_embeds = torch.cat(ve_embeds, dim=0).to(device)

    s3gen_refs = []
    for i in range(B):
        try:
            audio = batch['audio'][i].numpy()
            ref_audio = audio[:model.DEC_COND_LEN]
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))
        except Exception as e:
            print(f"Error in S3Gen ref {i}: {e}")
            # Create minimal reference
            ref_audio = np.zeros(model.DEC_COND_LEN)
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))

    t3_tokzr = s3gen.tokenizer
    plen = model.t3.hp.speech_cond_prompt_len
    tok_list = []
    if plen:
        for i in range(B):
            try:
                wav_16k = batch['audio_16k'][i].numpy()
                ref_16k = wav_16k[:model.ENC_COND_LEN]
                
                if len(ref_16k) < S3_SR // 2:  # At least 0.5 seconds
                    ref_16k = np.pad(ref_16k, (0, S3_SR // 2 - len(ref_16k)), mode='reflect')
                
                tokens, _ = t3_tokzr.forward([ref_16k], max_len=plen)
                tok_list.append(torch.atleast_2d(tokens))
            except Exception as e:
                print(f"Error tokenizing speech {i}: {e}")
                # Create dummy tokens
                dummy_tokens = torch.zeros(1, plen, dtype=torch.long)
                tok_list.append(dummy_tokens)
        t3_cond_tokens = torch.cat(tok_list, dim=0).to(device)
    else:
        t3_cond_tokens = torch.empty(B, 0, dtype=torch.long, device=device)

    t3_cond = T3Cond(
        speaker_emb=ve_embeds,
        cond_prompt_speech_tokens=t3_cond_tokens,
        emotion_adv=0.5 * torch.ones(B, 1, 1, device=device),
    )

    return t3_cond, s3gen_refs

def compute_loss(
    model: ChatterboxTTS,
    batch: Dict[str, torch.Tensor],
    t3_cond: T3Cond,
    s3gen_refs: List[dict],
) -> torch.Tensor:
    batch_size = batch['audio'].size(0)
    device = model.device

    # ── text → tokens ────────────────────────────────────────────────────────────
    text_tokens_list = []
    for i in range(batch_size):
        text = batch['text'][i]
        tokens = model.tokenizer.text_to_tokens(text).to(device)
        text_tokens_list.append(tokens)

    # Pad text tokens to same length
    max_text_len = max(t.size(-1) for t in text_tokens_list)
    text_tokens_padded = []
    for t in text_tokens_list:
        pad_amount = max_text_len - t.size(-1)
        if pad_amount > 0:
            padded = F.pad(t, (0, pad_amount), value=model.tokenizer.pad_token_id or 0)
        else:
            padded = t
        text_tokens_padded.append(padded)
    
    text_tokens = torch.cat(text_tokens_padded, dim=0)

    # Add start/stop tokens if needed
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    
    # Check if we need to add start/stop tokens
    if text_tokens.size(1) == 0 or text_tokens[0, 0] != sot:
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    if text_tokens.size(1) == 0 or text_tokens[0, -1] != eot:
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    # ── speech → tokens ─────────────────────────────────────────────────────────
    s3_tokzr = model.s3gen.tokenizer
    MAX_TOKENIZER_SEC = 30
    MAX_TOKENIZER_SAMPLES = MAX_TOKENIZER_SEC * S3_SR

    target_tokens_list = []
    for i in range(batch_size):
        audio_16k = batch['audio_16k'][i].cpu().numpy()
        
        # Truncate if too long
        if len(audio_16k) > MAX_TOKENIZER_SAMPLES:
            audio_16k = audio_16k[:MAX_TOKENIZER_SAMPLES]
        
        # Ensure minimum length
        if len(audio_16k) < S3_SR:  # Less than 1 second
            pad_amount = S3_SR - len(audio_16k)
            audio_16k = np.pad(audio_16k, (0, pad_amount), mode='constant')
        
        # Tokenize speech
        tokens, _ = s3_tokzr.forward([audio_16k])
        
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.from_numpy(tokens)
        
        # Ensure 2D
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            
        target_tokens_list.append(tokens)

    # Pad speech tokens to same length
    max_speech_len = max(t.size(-1) for t in target_tokens_list)
    target_tokens_padded = []
    for t in target_tokens_list:
        pad_amount = max_speech_len - t.size(-1)
        if pad_amount > 0:
            padded = F.pad(t, (0, pad_amount), value=-100)  # Use -100 for ignore_index
        else:
            padded = t
        target_tokens_padded.append(padded)
    
    target_tokens = torch.cat(target_tokens_padded, dim=0).to(device)

    # Print shapes for debugging
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Target tokens shape: {target_tokens.shape}")
    print(f"Batch size: {batch_size}")

    # Classifier-free guidance: double the batch for CFG
    text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
    target_tokens_doubled = torch.cat([target_tokens, target_tokens], dim=0)
    
    # Double the conditioning
    t3_cond_doubled = T3Cond(
        speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
        cond_prompt_speech_tokens=torch.cat(
            [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
        ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
        emotion_adv=torch.cat(
            [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
        ) if t3_cond.emotion_adv is not None else None,
    )

    # ── forward pass ─────────────────────────────────────────────────────────────
    # Use speech tokens for input (teacher forcing), excluding the last token
    input_speech_tokens = target_tokens_doubled[:, :-1]
    
    # Prepare embeddings
    embeds, len_cond = model.t3.prepare_input_embeds(
        t3_cond=t3_cond_doubled,
        text_tokens=text_tokens_doubled,
        speech_tokens=input_speech_tokens,
    )
    
    print(f"Embeds shape: {embeds.shape}")
    print(f"Conditioning length: {len_cond}")

    # Forward through transformer
    if DEVICE == 'cuda':
        with torch.cuda.amp.autocast():
            hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
    else:
        hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]

    print(f"Hidden states shape: {hidden_states.shape}")

    # Extract speech logits
    speech_start = len_cond + text_tokens.size(1)  # Skip conditioning + text
    speech_end = speech_start + target_tokens.size(1) - 1  # -1 because we excluded last token from input
    
    print(f"Speech logits slice: [{speech_start}:{speech_end}]")
    
    if speech_end > hidden_states.size(1):
        print(f"WARNING: speech_end ({speech_end}) > hidden_states length ({hidden_states.size(1)})")
        speech_end = hidden_states.size(1)
    
    if speech_start >= speech_end:
        print(f"ERROR: Invalid speech slice [{speech_start}:{speech_end}]")
        # Return a small loss to continue training
        return torch.tensor(1.0, requires_grad=True, device=device)
    
    speech_hidden = hidden_states[:, speech_start:speech_end]
    speech_logits = model.t3.speech_head(speech_hidden)
    
    print(f"Speech logits shape: {speech_logits.shape}")

    # Target tokens for loss (shifted by 1 for next-token prediction)
    target_shifted = target_tokens_doubled[:, 1:]  # Exclude first token (start token)
    
    print(f"Target shifted shape: {target_shifted.shape}")

    # Ensure shapes match
    min_len = min(speech_logits.size(1), target_shifted.size(1))
    speech_logits = speech_logits[:, :min_len]
    target_shifted = target_shifted[:, :min_len]
    
    print(f"Final shapes - logits: {speech_logits.shape}, targets: {target_shifted.shape}")

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        speech_logits.reshape(-1, speech_logits.size(-1)),  # (batch*seq, vocab)
        target_shifted.reshape(-1),  # (batch*seq,)
        ignore_index=-100,
    )
    
    print(f"Computed loss: {loss.item():.6f}")

    # Sanity checks
    if torch.isnan(loss):
        print("ERROR: NaN loss detected!")
        return torch.tensor(1.0, requires_grad=True, device=device)
    
    if torch.isinf(loss):
        print("ERROR: Infinite loss detected!")
        return torch.tensor(1.0, requires_grad=True, device=device)
    
    if loss.item() == 0.0:
        print("WARNING: Zero loss - check if targets are all ignore_index (-100)")
        print(f"Number of non-ignore targets: {(target_shifted != -100).sum().item()}")
    
    return loss


def main():
    """Main training function"""
    print(f"Starting Chatterbox TTS LoRA fine-tuning")
    print(f"Device: {DEVICE}")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_path="training_metrics.png", update_interval=2.0)
    
    # Load Whisper model
    print("Loading Whisper model...")
    whisper_model = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device="cuda")
    
    # Load audio samples
    samples = load_audio_samples(AUDIO_DATA_DIR, whisper_model)
    if len(samples) == 0:
        raise ValueError(f"No valid audio samples found in {AUDIO_DATA_DIR}")
    
    # Split into train/val
    random.shuffle(samples)
    val_size = int(len(samples) * VALIDATION_SPLIT)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")
    whisper_model.model.cpu()
    # Load Chatterbox model
    print("Loading Chatterbox TTS model...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    # Restart training
    #model = ChatterboxTTS.from_local("./checkpoints_lora/merged_model", DEVICE)

    # Inject LoRA layers
    print("Injecting LoRA layers...")
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_layers = inject_lora_layers(
        model.t3.tfmr,
        target_modules,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )
    print(f"Injected {len(lora_layers)} LoRA layers")
    
    # Create datasets
    train_dataset = TTSDataset(train_samples, model.tokenizer)
    val_dataset = TTSDataset(val_samples, model.tokenizer)
    
    # Set num_workers to 0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Setup optimizer (only LoRA parameters)
    lora_params = []
    for layer in lora_layers.values():
        lora_params.extend([layer.lora_A, layer.lora_B])
    
    optimizer = AdamW(lora_params, lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * EPOCHS,
        eta_min=LEARNING_RATE * 0.1
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None
    
    for epoch in range(EPOCHS):
        # Training
        model.t3.train()
        train_loss = 0.0
        train_steps = 0
        recent_losses = []
        step_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare conditionals
            t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, model.ve, model.s3gen)
            
            # Compute loss
            loss = compute_loss(model, batch, t3_cond, s3gen_refs)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Track batch loss
            batch_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            recent_losses.append(batch_loss)
            if len(recent_losses) > 100:
                recent_losses.pop(0)
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Calculate gradient norm before clipping
                grad_norm = 0.0
                for p in lora_params:
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                train_loss += batch_loss
                train_steps += 1
                
                # Calculate time per step
                step_time = time.time() - step_start_time
                step_start_time = time.time()
                
                # Update metrics
                avg_loss = train_loss / train_steps
                current_lr = scheduler.get_last_lr()[0]
                
                # Calculate loss variance
                loss_variance = np.var(recent_losses) if len(recent_losses) > 1 else 0
                
                # Update metrics tracker
                metrics_tracker.add_metrics(
                    train_loss=avg_loss,
                    learning_rate=current_lr,
                    steps=global_step,
                    epochs=epoch,
                    batch_loss=batch_loss,
                    gradient_norm=grad_norm,
                    loss_variance=loss_variance,
                    time_per_step=step_time
                )
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
                
                # Save checkpoint
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    save_checkpoint(model, lora_layers, optimizer, epoch, global_step, avg_loss, CHECKPOINT_DIR)
        
        # Validation
        model.t3.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, model.ve, model.s3gen)
                loss = compute_loss(model, batch, t3_cond, s3gen_refs)
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/train_steps:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update validation metrics
        metrics_tracker.add_metrics(
            val_loss=avg_val_loss,
            steps=global_step,
            epochs=epoch
        )
        
        # Save epoch checkpoint
        save_checkpoint(model, lora_layers, optimizer, epoch, global_step, avg_val_loss, CHECKPOINT_DIR)
    
    print("Training completed!")
    
    # Stop metrics tracker
    metrics_tracker.stop()
    
    # Save final LoRA adapter
    final_adapter_path = Path(CHECKPOINT_DIR) / "final_lora_adapter.pt"
    save_lora_adapter(lora_layers, str(final_adapter_path))
    
    # Create and save merged model
    print("Creating merged model...")
    
    # Clone the model state for merging
    merged_model = ChatterboxTTS.from_pretrained(DEVICE)
    
    # Re-inject LoRA layers and load final weights
    merged_lora_layers = inject_lora_layers(
        merged_model.t3.tfmr,
        target_modules,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )
    
    # Copy trained weights to merged model's LoRA layers
    for name, layer in lora_layers.items():
        if name in merged_lora_layers:
            merged_lora_layers[name].lora_A.data = layer.lora_A.data.clone()
            merged_lora_layers[name].lora_B.data = layer.lora_B.data.clone()
    
    # Merge LoRA weights into base model
    merged_model = merge_lora_weights(merged_model, merged_lora_layers)
    
    # Save merged model components
    merged_dir = Path(CHECKPOINT_DIR) / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each component
    torch.save(merged_model.ve.state_dict(), merged_dir / "ve.pt")
    torch.save(merged_model.t3.state_dict(), merged_dir / "t3_cfg.pt")
    torch.save(merged_model.s3gen.state_dict(), merged_dir / "s3gen.pt")
    
    # Copy tokenizer
    import shutil
    tokenizer_path = Path(hf_hub_download(repo_id="ResembleAI/chatterbox", filename="tokenizer.json"))
    shutil.copy(tokenizer_path, merged_dir / "tokenizer.json")
    
    # Save conditionals if they exist
    if model.conds:
        model.conds.save(merged_dir / "conds.pt")
    
    print(f"Saved merged model to {merged_dir}")
    print("\nTraining complete! You can now:")
    print(f"1. Use the LoRA adapter: {final_adapter_path}")
    print(f"2. Use the merged model: {merged_dir}")
    print("\nTo load the merged model:")
    print(f"  model = ChatterboxTTS.from_local('{merged_dir}', device='{DEVICE}')")
    print("\nTo load the LoRA adapter:")
    print(f"  lora_layers = load_lora_adapter(model, '{final_adapter_path}')")

def load_audio_samples(audio_dir: str, whisper_model) -> List[AudioSample]:
   """Load audio files and generate transcripts using Whisper"""
   samples = []
   audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
   
   # Cache file for transcripts
   cache_file = Path(audio_dir) / "transcripts_cache.json"
   transcript_cache = {}
   
   # Load existing cache if available
   if cache_file.exists():
       print(f"Loading transcript cache from {cache_file}")
       with open(cache_file, 'r', encoding='utf-8') as f:
           transcript_cache = json.load(f)
   
   print(f"Loading audio files from {audio_dir}...")
   audio_files = []
   for ext in audio_extensions:
       audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
   
   print(f"Found {len(audio_files)} audio files")
   
   # Track if we need to update cache
   cache_updated = False
   
   for audio_path in tqdm(audio_files, desc="Processing audio"):
       try:
           # Load audio for duration check
           audio, sr = librosa.load(audio_path, sr=None)
           duration = len(audio) / sr
           
           # Skip if too short or too long
           if duration < MIN_AUDIO_LENGTH or duration > MAX_AUDIO_LENGTH:
               continue
           
           # Check if we have cached transcript
           audio_path_str = str(audio_path.relative_to(Path(audio_dir)))
           
           if audio_path_str in transcript_cache:
               transcript = transcript_cache[audio_path_str]['transcript']
               print(f"Using cached transcript for {audio_path.name}")
           else:
               # Transcribe with Whisper
               print(f"\nTranscribing {audio_path.name}...")
               result = whisper_model(str(audio_path), return_timestamps=True)
               transcript = result['text'].strip()
               
               # Add to cache
               transcript_cache[audio_path_str] = {
                   'transcript': transcript,
                   'duration': duration,
                   'sample_rate': sr
               }
               cache_updated = True
           
           if transcript:
               samples.append(AudioSample(
                   audio_path=audio_path,
                   transcript=transcript,
                   duration=duration,
                   sample_rate=sr
               ))
       except Exception as e:
           print(f"Error processing {audio_path}: {e}")
           continue
   
   # Save updated cache
   if cache_updated:
       print(f"Saving transcript cache to {cache_file}")
       with open(cache_file, 'w', encoding='utf-8') as f:
           json.dump(transcript_cache, f, ensure_ascii=False, indent=2)
   
   print(f"Successfully loaded {len(samples)} samples")
   return samples


def save_checkpoint(
   model: ChatterboxTTS,
   lora_layers: Dict[str, LoRALayer],
   optimizer: torch.optim.Optimizer,
   epoch: int,
   step: int,
   loss: float,
   checkpoint_dir: str,
   is_best: bool = False,
):
   """Save training checkpoint"""
   checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch}_step{step}.pt"
   if is_best:
       checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
   
   checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
   
   # Extract LoRA weights
   lora_state_dict = {}
   for name, layer in lora_layers.items():
       lora_state_dict[f"{name}.lora_A"] = layer.lora_A
       lora_state_dict[f"{name}.lora_B"] = layer.lora_B
   
   checkpoint = {
       'epoch': epoch,
       'step': step,
       'loss': loss,
       'lora_state_dict': lora_state_dict,
       'optimizer_state_dict': optimizer.state_dict(),
   }
   
   torch.save(checkpoint, checkpoint_path)
   print(f"Saved checkpoint to {checkpoint_path}")


def merge_lora_weights(model: ChatterboxTTS, lora_layers: Dict[str, LoRALayer]):
   """Merge LoRA weights into the base model"""
   with torch.no_grad():
       for name, lora_layer in lora_layers.items():
           # Find the corresponding linear layer in the model
           parts = name.split('.')
           module = model.t3.tfmr
           for part in parts[:-1]:
               module = getattr(module, part)
           linear_layer = getattr(module, parts[-1])
           
           # Compute LoRA update: W' = W + BA * scaling
           lora_update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
           
           # Add to original weights
           linear_layer.weight.data += lora_update
   
   return model


def save_lora_adapter(lora_layers: Dict[str, LoRALayer], filepath: str):
   """Save LoRA adapter weights and configuration"""
   adapter_dict = {
       'lora_config': {
           'rank': LORA_RANK,
           'alpha': LORA_ALPHA,
           'dropout': LORA_DROPOUT,
           'target_modules': list(set(name.split('.')[-1] for name in lora_layers.keys())),
       },
       'lora_weights': {},
   }
   
   for name, layer in lora_layers.items():
       adapter_dict['lora_weights'][name] = {
           'lora_A': layer.lora_A.cpu(),
           'lora_B': layer.lora_B.cpu(),
       }
   
   torch.save(adapter_dict, filepath)
   print(f"Saved LoRA adapter to {filepath}")


def load_lora_adapter(model: ChatterboxTTS, filepath: str, device: str = 'cuda'):
   """Load LoRA adapter weights"""
   adapter_dict = torch.load(filepath, map_location=device)
   config = adapter_dict['lora_config']
   
   # Inject LoRA layers
   lora_layers = inject_lora_layers(
       model.t3.tfmr,
       config['target_modules'],
       rank=config['rank'],
       alpha=config['alpha'],
       dropout=config['dropout']
   )
   
   # Load weights
   for name, weights in adapter_dict['lora_weights'].items():
       if name in lora_layers:
           lora_layers[name].lora_A.data = weights['lora_A'].to(device)
           lora_layers[name].lora_B.data = weights['lora_B'].to(device)
   
   return lora_layers


def collate_fn(samples):
   """Custom collate function for DataLoader"""
   return {
       'audio': torch.stack([s['audio'] for s in samples]),
       'audio_16k': torch.stack([s['audio_16k'] for s in samples]),
       'text': [s['text'] for s in samples],
       'audio_path': [s['audio_path'] for s in samples],
   }


if __name__ == "__main__":
   main()