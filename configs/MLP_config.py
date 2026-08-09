from pathlib import Path


class MLPConfig():
    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent  # 基目录
        self.raw_data_path = self.base_path / "data" / "raw"  # 原始数据根目录
        self.processed_path = self.base_path / "data" / "processed"  # 数据处理保存路径
        self.adjective_path = self.raw_data_path / "adjective" / "toxic_adjectives_v2.csv"  # 形容词词典路径（v2）
        self.models_path = self.base_path / "models"  # LLM模型路径
        self.experiment_path = self.base_path / "experiments"  # 实验结果保存目录

        # ========== MLP 训练超参数 ==========
        self.batch_size = 16  # 批次大小
        self.epochs = 200  # 训练轮数
        
        # OneCycleLR 学习率调度器参数
        self.max_lr = 1e-3  # 峰值学习率
        self.pct_start = 0.2  # Warmup 占总步数的比例
        self.div_factor = 25.0  # 初始学习率 = max_lr / div_factor
        self.final_div_factor = 10000.0  # 最终学习率 = max_lr / final_div_factor
        self.anneal_strategy = 'cos'  # Warmup 后余弦衰减

        # ========== MLP 模型参数 ==========
        self.dropout_rate = 0.5  # Dropout 比率
        self.hidden_features = 96  # 隐藏层特征维度
        self.patience = 20  # 早停耐心值 (验证集F1连续patience个epoch未提升则停止训练)

    def __repr__(self):
        """返回配置对象的字符串表示，包含所有配置项"""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        """返回配置对象的友好字符串表示"""
        return self.__repr__()


if __name__ == '__main__':
    config = MLPConfig()
    print(config)