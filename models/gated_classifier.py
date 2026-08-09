"""门控概念分类器模型定义。

对应论文 2.3 节"门控概念分类器"：
  输入等级特征向量 x = [p^{(3)}; p^{(2)}; d̃] ∈ R^{3l}，
  经矩阵门控自适应加权后，由两层全连接网络映射至标签空间。
"""

import torch
import torch.nn as nn


class GatedConceptClassifier(nn.Module):
    """门控概念分类器（对应论文 2.3 节）。

    输入等级特征向量 x = [p^{(3)}; p^{(2)}; d̃] ∈ R^{3l}，
    经矩阵门控自适应加权后，由两层全连接网络映射至标签空间。

    Args:
        n_concepts: 概念数量 l（v2 词典为 132）
        concept_types: 概念类型列表（保留参数，当前未使用）
        dropout_rate: Dropout 比率
        hidden_features: 隐藏层维度 d_h
        n_summary: 类型级汇总特征维度（保留参数，当前为 0）
        n_main_channels: 主特征通道数（默认 3 路：P3+P2+加权差值）
    """
    def __init__(self, n_concepts, concept_types, dropout_rate=0.5,
                 hidden_features=96, n_summary=0, n_main_channels=3):
        super().__init__()
        self.main_dim = n_concepts * n_main_channels
        self.gate_layer = nn.Linear(self.main_dim, self.main_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.main_dim + n_summary, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        main, summary = x[:, :self.main_dim], x[:, self.main_dim:]
        gated = main * torch.sigmoid(self.gate_layer(main))
        h = self.relu(self.fc1(self.dropout(torch.cat([gated, summary], dim=1))))
        return self.fc2(self.dropout(h))