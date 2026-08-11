# 配置参数
sequence_length = 30
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,
    'd_model': 128,          # 缩小容量，降低过拟合
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'batch_size': 8,
    'num_epochs': 80,
    'learning_rate': 3e-4,
    'dropout': 0.3,
    'feature_num': feature_num,
    'max_grad_norm': 1.0,
    'use_clip': True,
    'warmup_epochs': 5,
    'weight_decay': 1e-3,

    'cross_stock_layers': 2,          # 与 model 中实际层数一致
    'early_stopping_patience': 10,

    'seed': 42,
    # 70% 龙头池最优版：单 seed 复现
    'ensemble_seeds': [42],

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',

    # 龙头股筛选：训练前用训练集近期成交额定池；预测时复用同一池
    'train_leader_ratio': 0.70,
    'leader_lookback_days': 30,
    'predict_use_leader_pool': True,

    # 最终输出权重保持不变
    'top_k_output': 5,
    'top_k_weights': [0.50, 0.47, 0.01, 0.01, 0.01],
    # 损失对齐提交的 top_k_output
    'topk_mle_k': 5,
    # 纯 TopK-ListMLE（与 70% 单种子最优版一致）
    'portfolio_loss_weight': 0.0,
    'portfolio_temperature': 0.35,
    # 选模：按提交权重的相对分数
    'selection_metric': 'weighted_final_score',
}
