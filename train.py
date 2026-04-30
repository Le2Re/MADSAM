import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry
from trainer import trainer_khanhha

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, help='root dir for training data')
parser.add_argument('--val_path', type=str, help='root dir for validation data')
parser.add_argument('--output', type=str, default='./output/training')
parser.add_argument('--dataset', type=str,
                    default='khanhha', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_khanhha', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations number to train')
parser.add_argument('--max_epochs', type=int,
                    default=140, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=130, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0004,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=448, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=300,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--lr_exp', type=float, default=0.6, help='The learning rate decay expotential')
parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration, but may cause NaN')
parser.add_argument('--save_interval', type=int, default=3, help='Save and validation intervals')

args = parser.parse_args()

if __name__ == "__main__":
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'khanhha': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          :-3] + 'k' 
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) 
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) 
    snapshot_path = snapshot_path + '_s' + str(args.seed) 
   

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    
    net =sam.cuda()

    if args.num_classes > 1: 
        multimask_output = True
    else:
        multimask_output = False # For crack segmentation

    low_res = img_embedding_size * 4  # It's better to use high resolution in crack segmentation

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters:{total_params}")

    total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters:{total_params_train}")

    trainer = {'khanhha': trainer_khanhha}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
