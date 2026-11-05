"""
Sleep Apnea Detection Model Architecture.
Deep 1D-CNN + Bidirectional LSTM + Multi-Head Self-Attention.
Validated on PhysioNet Apnea-ECG Database (100 Hz Lead-II Electrocardiogram).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Temporal Self-Attention mechanism.
    Learns to assign higher attention weights to critical pathological phases
    such as cardiac vagal pauses, compensatory tachycardia, and R-peak suppression.
    """
    def __init__(self, hidden_dim: int):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, hidden_dim]
        scores = self.projection(x)  # [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(x * weights, dim=1)  # [batch_size, hidden_dim]
        return context, weights


class SleepApneaNet(nn.Module):
    """
    End-to-end Hybrid Deep Neural Network for Obstructive Sleep Apnea (OSA) Detection.
    
    Architecture Pipeline:
      1. 1D-CNN Feature Extractor: Hierarchical temporal convolutions with decaying kernel sizes (15 -> 11 -> 7).
      2. Bidirectional LSTM: Captures long-range forward and backward cardiac autonomic dynamics.
      3. Self-Attention Layer: Aggregates salient temporal epochs while generating explainability weights.
      4. Fully Connected Head: Regularized classification head outputting Apnea vs Normal logits.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(SleepApneaNet, self).__init__()
        
        # 1. 1D-CNN Hierarchical Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.2)
        
        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 3. Temporal Self-Attention
        self.attention = SelfAttention(hidden_dim=64 * 2)
        
        # 4. Dense Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

        # Hook holders for 1D-GradCAM Explainability
        self.gradient = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.activations

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Input shape: [batch_size, 1, seq_len] (e.g. 1000 or 3000 samples @ 100 Hz)
        """
        # Conv Block 1
        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        # Conv Block 2
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        # Conv Block 3
        x = self.pool3(F.gelu(self.bn3(self.conv3(x))))
        x = self.dropout_cnn(x)
        
        # Save activations for GradCAM hook
        self.activations = x
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            
        # Re-arrange for LSTM: [batch, channels, seq] -> [batch, seq, channels]
        x_seq = x.permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x_seq)
        
        # Attention aggregation
        context, attn_weights = self.attention(lstm_out)
        
        # Classification
        logits = self.fc(context)
        
        if return_attention:
            return logits, attn_weights
        return logits
