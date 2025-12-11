#!/bin/bash

# ============================================================================
# ABIDE双分支GNN完整实验自动化脚本
# ============================================================================

set -e  # 遇到错误立即停止

echo "========================================================================"
echo "ABIDE双分支GNN实验 - 自动化执行"
echo "========================================================================"

# 配置参数
DATA_PATH="./data/gnn_datasets/MDD_DualBranch.pkl"
RESULTS_DIR="./results_baseline/MDD"
mkdir -p $RESULTS_DIR

# ============================================================================
# 步骤1：数据准备
# ============================================================================
echo ""
echo "========================================================================"
echo "步骤1/5: 数据准备（修复NaN）"
echo "========================================================================"

if [ ! -f "$DATA_PATH" ]; then
    echo "正在生成数据集..."
    python prepare_final_data.py \
        --dataset MDD \
        --data_folder ./data \
        --encoder_path ./pretrained_models/node_encoder_best.pth \
        --save_dir ./data/gnn_datasets \
        --fc_k 20 \
        --fc_method ledoit_wolf

    echo "✅ 数据准备完成"
else
    echo "✅ 数据集已存在: $DATA_PATH"
fi

# 验证数据
echo ""
echo "验证数据质量..."
python -c "
import pickle
import torch

with open('$DATA_PATH', 'rb') as f:
    data = pickle.load(f)

sample = data['graph_list'][0]
has_nan = torch.isnan(sample.x).any()
has_inf = torch.isinf(sample.x).any()

print(f'  节点特征: {sample.x.shape}')
print(f'  结构图边数: {sample.edge_index_struct.shape[1]}')
print(f'  功能图边数: {sample.edge_index_func.shape[1]}')
print(f'  是否有NaN: {has_nan}')
print(f'  是否有Inf: {has_inf}')

assert not has_nan and not has_inf, '❌ 数据仍有NaN/Inf！'
print('✅ 数据验证通过')
"

# ============================================================================
# 步骤2：基线实验（无Prompt）
# ============================================================================
echo ""
echo "========================================================================"
echo "步骤2/5: 基线实验（无Prompt）"
echo "========================================================================"

# B1: Structure-Only
echo ""
echo "实验B1: 结构分支（无Prompt）"
echo "预期BACC: 60-62%"
python run_baseline.py \
    --data_path $DATA_PATH \
    --mode struct \
    --hidden_dim 64 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 20 \
    2>&1 | tee $RESULTS_DIR/B1_struct_no_prompt.log

# B2: Functional-Only
echo ""
echo "实验B2: 功能分支（无Prompt）"
echo "预期BACC: 62-65%"
python run_baseline.py \
    --data_path $DATA_PATH \
    --mode func \
    --hidden_dim 64 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 20 \
    2>&1 | tee $RESULTS_DIR/B2_func_no_prompt.log

# ============================================================================
# 步骤3：Prompt增强实验
# ============================================================================
echo ""
echo "========================================================================"
echo "步骤3/5: Prompt增强实验"
echo "========================================================================"

# E1: Structure + NodePrompt+
echo ""
echo "实验E1: 结构分支 + NodePrompt+"
echo "预期提升: 1-2%"
python run_baseline.py \
    --data_path $DATA_PATH \
    --mode struct \
    --use_prompt \
    --num_anchors 5 \
    --hidden_dim 64 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 20 \
    2>&1 | tee $RESULTS_DIR/E1_struct_with_prompt.log

# E2: Functional + EdgePrompt+
echo ""
echo "实验E2: 功能分支 + EdgePrompt+"
echo "预期提升: 1-3%"
python run_baseline.py \
    --data_path $DATA_PATH \
    --mode func \
    --use_prompt \
    --num_anchors 5 \
    --hidden_dim 64 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 20 \
    2>&1 | tee $RESULTS_DIR/E2_func_with_prompt.log

# ============================================================================
# 步骤4：双分支融合实验
# ============================================================================
echo ""
echo "========================================================================"
echo "步骤4/5: 双分支融合实验"
echo "========================================================================"
echo "预期BACC: 68%+"

python run_final.py \
    --dataset MDD \
    --data_path $DATA_PATH \
    --input_dim 308 \
    --hidden_dim 64 \
    --num_layers 3 \
    --pooling_ratio 0.5 \
    --dropout 0.5 \
    --num_anchors 5 \
    --lambda_orth 0.1 \
    --lambda_cons 0.05 \
    --use_consistency \
    --n_folds 5 \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --patience 30 \
    --save_results \
    --result_dir $RESULTS_DIR \
    2>&1 | tee $RESULTS_DIR/Final_dual_branch.log

# ============================================================================
# 步骤5：结果汇总
# ============================================================================
echo ""
echo "========================================================================"
echo "步骤5/5: 结果汇总"
echo "========================================================================"

python -c "
import re
import os

results_dir = '$RESULTS_DIR'
experiments = {
    'B1 (Structure, No Prompt)': 'B1_struct_no_prompt.log',
    'B2 (Functional, No Prompt)': 'B2_func_no_prompt.log',
    'E1 (Structure + NodePrompt+)': 'E1_struct_with_prompt.log',
    'E2 (Functional + EdgePrompt+)': 'E2_func_with_prompt.log',
    'Final (Dual-Branch)': 'Final_dual_branch.log'
}

results = {}

for exp_name, log_file in experiments.items():
    log_path = os.path.join(results_dir, log_file)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            # 提取BACC（假设格式为 'BACC: 0.XXXX'）
            match = re.search(r'BACC:\s+([\d.]+)', content)
            if match:
                results[exp_name] = float(match.group(1))
            else:
                results[exp_name] = None
    else:
        results[exp_name] = None

print('=' * 70)
print('实验结果汇总')
print('=' * 70)

for exp, bacc in results.items():
    if bacc is not None:
        print(f'{exp:40s}: {bacc:.2%}')
    else:
        print(f'{exp:40s}: [未完成或解析失败]')

print()
print('=' * 70)
print('关键发现')
print('=' * 70)

b1 = results.get('B1 (Structure, No Prompt)')
b2 = results.get('B2 (Functional, No Prompt)')
e1 = results.get('E1 (Structure + NodePrompt+)')
e2 = results.get('E2 (Functional + EdgePrompt+)')
final = results.get('Final (Dual-Branch)')

if all(x is not None for x in [b1, b2, e1, e2, final]):
    print(f'1. NodePrompt提升: {(e1-b1):.2%}')
    print(f'2. EdgePrompt提升: {(e2-b2):.2%}')
    print(f'3. 最佳单分支: {max(b1,b2,e1,e2):.2%}')
    print(f'4. 双分支融合提升: {(final-max(b1,b2,e1,e2)):.2%}')
    print(f'5. 总提升（vs最差基线）: {(final-min(b1,b2)):.2%}')
    print()
    if final >= 0.68:
        print('🎉 恭喜！达到目标性能（≥68%）')
    else:
        print(f'⚠️  距离目标还差 {(0.68-final):.2%}，建议调参')
else:
    print('⚠️  部分实验未完成，请检查日志')

print('=' * 70)
"

echo ""
echo "========================================================================"
echo "✅ 所有实验完成！"
echo "========================================================================"
echo "结果保存在: $RESULTS_DIR"
echo "日志文件:"
ls -lh $RESULTS_DIR/*.log

echo ""
echo "下一步建议:"
echo "1. 查看各实验的详细日志"
echo "2. 分析结果汇总（见上方）"
echo "3. 如果未达到预期，调整超参数后重新运行"
echo "4. 绘制学习曲线和混淆矩阵"