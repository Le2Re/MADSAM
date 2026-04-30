"""
t-SNE 可视化脚本：对比未对齐（原始SAM）和已对齐（MADSAM）的特征分布
用于验证 DPE 模块的域适应效果
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from segment_anything import sam_model_registry
from datasets.dataset_khanhha import Khanhha_dataset


def extract_features(model, dataloader, device, use_dpe=True, max_samples=None):
    """
    提取图像特征
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        use_dpe: 是否使用 DPE 模块（True=已对齐，False=未对齐）
        max_samples: 最大采样数量（None表示使用全部）
    
    Returns:
        features: 特征向量列表 [N, feature_dim]
        labels: 域标签列表 ['source' or 'target']
    """
    model.eval()
    features = []
    labels = []
    sample_count = 0
    
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tqdm(dataloader, desc=f"提取特征 (use_dpe={use_dpe})")):
            if max_samples and sample_count >= max_samples:
                break
                
            image = sampled_batch['image'].to(device)  # [B, C, H, W]
            domain_label = sampled_batch.get('domain', 'source')  # 默认是 source domain
            
            # 预处理图像（确保在 [0, 1] 范围内）
            if image.max() > 1.0:
                image = image / 255.0
            
            # 获取图像嵌入
            image_embeddings = model.image_encoder(image)  # [B, C, H, W]
            
            if use_dpe:
                # 已对齐：使用 DPE 模块处理后的 prompt
                _, prompt, _, _ = model.prompt_generator(image_embeddings)  # prompt: [B, C, H, W]
                # 全局平均池化得到特征向量
                feature = F.adaptive_avg_pool2d(prompt, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            else:
                # 未对齐：直接使用原始 image_embeddings
                feature = F.adaptive_avg_pool2d(image_embeddings, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            
            # 转换为 numpy
            feature_np = feature.cpu().numpy()
            batch_size = feature_np.shape[0]
            
            # 如果超过最大样本数，只取需要的部分
            if max_samples and sample_count + batch_size > max_samples:
                remaining = max_samples - sample_count
                feature_np = feature_np[:remaining]
                batch_size = remaining
            
            features.append(feature_np)
            
            # 添加域标签
            if isinstance(domain_label, str):
                batch_labels = [domain_label] * batch_size
            else:
                batch_labels = domain_label[:batch_size] if len(domain_label) >= batch_size else domain_label
            labels.extend(batch_labels)
            
            sample_count += batch_size
            
            if max_samples and sample_count >= max_samples:
                break
    
    # 合并所有特征
    if features:
        features = np.vstack(features)
    else:
        features = np.array([])
    
    return features, labels


def create_dataloader_with_domain(base_dir, list_dir, split, domain_label, batch_size=8):
    """创建带有域标签的数据加载器"""
    # 确保 base_dir 以 '/' 结尾，以便正确拼接路径
    if base_dir and not base_dir.endswith('/') and not base_dir.endswith('\\'):
        base_dir = base_dir + '/'
    dataset = Khanhha_dataset(base_dir=base_dir, list_dir=list_dir, split=split, transform=None)
    
    # 为每个样本添加域标签
    class DomainDataset:
        def __init__(self, base_dataset, domain_label):
            self.base_dataset = base_dataset
            self.domain_label = domain_label
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            sample = self.base_dataset[idx]
            sample['domain'] = self.domain_label
            return sample
    
    domain_dataset = DomainDataset(dataset, domain_label)
    dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader


def visualize_tsne(features, labels, title, save_path, perplexity=30, n_iter=1000):
    """
    使用 t-SNE 进行降维可视化
    
    Args:
        features: 特征矩阵 [N, feature_dim]
        labels: 域标签列表
        title: 图表标题
        save_path: 保存路径
        perplexity: t-SNE 的困惑度参数
        n_iter: 迭代次数
    """
    print(f"\n开始 t-SNE 降维...")
    print(f"特征形状: {features.shape}")
    print(f"样本数量: {len(labels)}")
    
    # 执行 t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features)-1), 
                max_iter=n_iter, random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    
    # 分离不同域的数据点
    source_indices = [i for i, label in enumerate(labels) if label == 'source']
    target_indices = [i for i, label in enumerate(labels) if label == 'target']
    
    if source_indices:
        plt.scatter(features_2d[source_indices, 0], features_2d[source_indices, 1],
                   c='blue', label='Source Domain (Training Set)', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    if target_indices:
        plt.scatter(features_2d[target_indices, 0], features_2d[target_indices, 1],
                   c='red', label='Target Domain (Test Set)', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='t-SNE 可视化：对比域适应效果')
    parser.add_argument('--root_path', type=str, required=True, help='训练数据根目录')
    parser.add_argument('--val_path', type=str, required=True, help='测试数据根目录')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_khanhha', help='列表文件目录')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--img_size', type=int, default=448, help='图像大小')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='ViT 模型名称')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_samples', type=int, default=500, help='每个域的最大采样数量（None表示使用全部）')
    parser.add_argument('--output_dir', type=str, default='./results_visualization', help='输出目录')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE 困惑度参数')
    parser.add_argument('--n_iter', type=int, default=1000, help='t-SNE 迭代次数')
    
    args = parser.parse_args()
    
    # 设置随机种子（确保可重复性）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # 确保 CUDA 操作是确定性的（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    # 首先加载预训练模型（如果 ckpt 是预训练模型路径）
    # 如果 ckpt 是训练后的模型，需要先加载预训练模型，再加载训练权重
    pretrain_ckpt = 'checkpoints/sam_vit_b_01ec64.pth'  # 预训练模型路径
    if os.path.exists(pretrain_ckpt):
        sam, img_embedding_size = sam_model_registry[args.vit_name](
            image_size=args.img_size,
            num_classes=1,
            checkpoint=pretrain_ckpt,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1]
        )
        # 如果提供了训练后的模型权重，加载它
        if args.ckpt != pretrain_ckpt and os.path.exists(args.ckpt):
            print(f"加载训练后的模型权重: {args.ckpt}")
            weights = torch.load(args.ckpt, map_location=device, weights_only=False)
            sam.load_state_dict(weights, strict=False)
    else:
        # 直接加载模型（可能是训练后的完整模型）
        sam, img_embedding_size = sam_model_registry[args.vit_name](
            image_size=args.img_size,
            num_classes=1,
            checkpoint=args.ckpt if os.path.exists(args.ckpt) else None,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1]
        )
        if os.path.exists(args.ckpt):
            weights = torch.load(args.ckpt, map_location=device, weights_only=False)
            sam.load_state_dict(weights, strict=False)
    
    sam = sam.to(device)
    sam.eval()
    print("模型加载完成")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader = create_dataloader_with_domain(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='train',
        domain_label='source',
        batch_size=args.batch_size
    )
    
    test_loader = create_dataloader_with_domain(
        base_dir=args.val_path,
        list_dir=args.list_dir,
        split='test_vol' if 'test_vol' in os.listdir(args.list_dir) else 'val_vol',
        domain_label='target',
        batch_size=args.batch_size
    )
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 1. 提取未对齐特征（原始 SAM）
    print("\n" + "="*60)
    print("步骤 1: 提取未对齐特征（原始 SAM Image Embedding）")
    print("="*60)
    
    # 分别提取训练集和测试集特征
    print("提取训练集特征（Source Domain）...")
    features_train_unaligned, labels_train_unaligned = extract_features(
        sam, train_loader, device, use_dpe=False, 
        max_samples=args.max_samples if args.max_samples else None
    )
    
    print("提取测试集特征（Target Domain）...")
    features_test_unaligned, labels_test_unaligned = extract_features(
        sam, test_loader, device, use_dpe=False,
        max_samples=args.max_samples if args.max_samples else None
    )
    
    # 合并特征和标签
    features_unaligned = np.vstack([features_train_unaligned, features_test_unaligned])
    labels_unaligned = labels_train_unaligned + labels_test_unaligned
    
    print(f"未对齐特征总数: {len(features_unaligned)} (Source: {len(features_train_unaligned)}, Target: {len(features_test_unaligned)})")
    
    # 2. 提取已对齐特征（MADSAM with DPE）
    print("\n" + "="*60)
    print("步骤 2: 提取已对齐特征（MADSAM with DPE Prompt）")
    print("="*60)
    
    print("提取训练集特征（Source Domain）...")
    features_train_aligned, labels_train_aligned = extract_features(
        sam, train_loader, device, use_dpe=True,
        max_samples=args.max_samples if args.max_samples else None
    )
    
    print("提取测试集特征（Target Domain）...")
    features_test_aligned, labels_test_aligned = extract_features(
        sam, test_loader, device, use_dpe=True,
        max_samples=args.max_samples if args.max_samples else None
    )
    
    # 合并特征和标签
    features_aligned = np.vstack([features_train_aligned, features_test_aligned])
    labels_aligned = labels_train_aligned + labels_test_aligned
    
    print(f"已对齐特征总数: {len(features_aligned)} (Source: {len(features_train_aligned)}, Target: {len(features_test_aligned)})")
    
    # 3. 可视化对比
    print("\n" + "="*60)
    print("步骤 3: 生成 t-SNE 可视化")
    print("="*60)
    
    # 未对齐可视化
    visualize_tsne(
        features_unaligned,
        labels_unaligned,
        title='Unaligned: Original SAM Image Embedding\n(Expected: Source and Target separated, Domain Gap exists)',
        save_path=os.path.join(args.output_dir, 'tsne_unaligned.png'),
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )
    
    # 已对齐可视化
    visualize_tsne(
        features_aligned,
        labels_aligned,
        title='Aligned: MADSAM with DPE Prompt\n(Expected: Source and Target mixed, Domain Gap reduced)',
        save_path=os.path.join(args.output_dir, 'tsne_aligned.png'),
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )
    
    # 4. 生成对比图（并排显示）
    print("\n生成对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 未对齐
    tsne_unaligned = TSNE(n_components=2, perplexity=min(args.perplexity, len(features_unaligned)-1),
                         max_iter=args.n_iter, random_state=42)
    features_2d_unaligned = tsne_unaligned.fit_transform(features_unaligned)
    
    source_indices = [i for i, label in enumerate(labels_unaligned) if label == 'source']
    target_indices = [i for i, label in enumerate(labels_unaligned) if label == 'target']
    
    if source_indices:
        axes[0].scatter(features_2d_unaligned[source_indices, 0], features_2d_unaligned[source_indices, 1],
                       c='blue', label='Source Domain (Training Set)', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    if target_indices:
        axes[0].scatter(features_2d_unaligned[target_indices, 0], features_2d_unaligned[target_indices, 1],
                       c='red', label='Target Domain (Test Set)', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    axes[0].set_title('Unaligned: Original SAM Image Embedding\n(Expected: Source and Target separated, Domain Gap exists)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 已对齐
    tsne_aligned = TSNE(n_components=2, perplexity=min(args.perplexity, len(features_aligned)-1),
                       max_iter=args.n_iter, random_state=42)
    features_2d_aligned = tsne_aligned.fit_transform(features_aligned)
    
    source_indices = [i for i, label in enumerate(labels_aligned) if label == 'source']
    target_indices = [i for i, label in enumerate(labels_aligned) if label == 'target']
    
    if source_indices:
        axes[1].scatter(features_2d_aligned[source_indices, 0], features_2d_aligned[source_indices, 1],
                       c='blue', label='Source Domain (Training Set)', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    if target_indices:
        axes[1].scatter(features_2d_aligned[target_indices, 0], features_2d_aligned[target_indices, 1],
                       c='red', label='Target Domain (Test Set)', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    axes[1].set_title('Aligned: MADSAM with DPE Prompt\n(Expected: Source and Target mixed, Domain Gap reduced)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, 'tsne_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {comparison_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)
    print(f"输出文件:")
    print(f"  1. {os.path.join(args.output_dir, 'tsne_unaligned.png')}")
    print(f"  2. {os.path.join(args.output_dir, 'tsne_aligned.png')}")
    print(f"  3. {comparison_path}")


if __name__ == '__main__':
    main()
