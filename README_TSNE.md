# t-SNE 域适应可视化使用说明

## 功能说明

本脚本用于可视化对比**未对齐（原始 SAM）**和**已对齐（MADSAM with DPE）**的特征分布，验证 DPE 模块的域适应效果。

## 预期结果

- **未对齐（原始 SAM）**：训练集（Source Domain）和测试集（Target Domain）的特征在 t-SNE 图上应该分成两堆，说明存在 Domain Gap。
- **已对齐（MADSAM）**：训练集和测试集的特征应该混合在一起，说明 Domain Gap 被缩小，域适应成功。

## 使用方法

### 基本用法

```bash
python visualize_tsne_domain_gap.py \
    --root_path /path/to/train/data \
    --val_path /path/to/test/data \
    --ckpt /path/to/trained/model.pth \
    --list_dir ./lists/lists_khanhha
```

### 完整参数说明

```bash
python visualize_tsne_domain_gap.py \
    --root_path /path/to/train/data \          # 训练数据根目录
    --val_path /path/to/test/data \            # 测试数据根目录
    --list_dir ./lists/lists_khanhha \         # 列表文件目录
    --ckpt ./output/training/xxx/epoch_120.pth \  # 训练后的模型检查点
    --img_size 448 \                           # 图像大小（默认：448）
    --vit_name vit_b \                         # ViT 模型名称（默认：vit_b）
    --batch_size 8 \                           # 批次大小（默认：8）
    --max_samples 500 \                         # 每个域的最大采样数量（默认：500，None表示使用全部）
    --output_dir ./results_visualization \      # 输出目录（默认：./results_visualization）
    --perplexity 30 \                          # t-SNE 困惑度参数（默认：30）
    --n_iter 1000                              # t-SNE 迭代次数（默认：1000）
```

### 参数说明

- `--root_path`: 训练数据（Source Domain）的根目录，应包含 `images/` 和 `masks/` 子目录
- `--val_path`: 测试数据（Target Domain）的根目录，应包含 `images/` 和 `masks/` 子目录
- `--list_dir`: 包含 `train.txt` 和 `test_vol.txt`（或 `val_vol.txt`）的目录
- `--ckpt`: 训练后的模型检查点路径（.pth 文件）
- `--max_samples`: 限制每个域的采样数量，可以加快处理速度。如果数据集很大，建议设置为 200-500
- `--perplexity`: t-SNE 的困惑度参数，通常设置为 5-50。如果样本数较少，应该相应减小
- `--n_iter`: t-SNE 迭代次数，更多迭代可能得到更好的结果，但耗时更长

## 输出文件

脚本会在 `--output_dir` 目录下生成以下文件：

1. **tsne_unaligned.png**: 未对齐（原始 SAM）的 t-SNE 可视化
2. **tsne_aligned.png**: 已对齐（MADSAM with DPE）的 t-SNE 可视化
3. **tsne_comparison.png**: 并排对比图

## 示例

假设你有一个训练好的模型在 `output/training/khanhha_448_pretrain_vit_b_30k_epo140_bs8_lr0.0004_s3407/epoch_120.pth`：

```bash
python visualize_tsne_domain_gap.py \
    --root_path /data/khanhha/train \
    --val_path /data/khanhha/test \
    --list_dir ./lists/lists_khanhha \
    --ckpt ./output/training/khanhha_448_pretrain_vit_b_30k_epo140_bs8_lr0.0004_s3407/epoch_120.pth \
    --max_samples 300 \
    --output_dir ./results_visualization
```

## 注意事项

1. **模型加载**：脚本会自动尝试加载预训练模型（`checkpoints/sam_vit_b_01ec64.pth`），然后加载你提供的训练后权重。如果预训练模型不存在，会直接加载你提供的检查点。

2. **数据格式**：确保数据目录结构正确：
   ```
   root_path/
     images/
       image1.jpg
       image2.jpg
       ...
     masks/
       image1.jpg
       image2.jpg
       ...
   ```

3. **内存使用**：如果数据集很大，建议使用 `--max_samples` 限制采样数量，避免内存不足。

4. **t-SNE 参数调整**：
   - 如果样本数 < 100，建议 `--perplexity` 设置为 5-10
   - 如果样本数 > 1000，可以设置 `--perplexity` 为 30-50
   - 更多迭代（`--n_iter`）可能得到更好的结果，但耗时更长

5. **特征提取**：
   - **未对齐**：直接从 `image_encoder` 提取的 `image_embeddings`，经过全局平均池化
   - **已对齐**：经过 DPE 模块处理后的 `prompt`，经过全局平均池化

## 故障排除

1. **CUDA 内存不足**：减小 `--batch_size` 或 `--max_samples`
2. **找不到数据**：检查 `--root_path` 和 `--val_path` 是否正确
3. **模型加载失败**：确保 `--ckpt` 路径正确，且模型架构匹配
4. **t-SNE 报错**：如果样本数太少，减小 `--perplexity` 参数
