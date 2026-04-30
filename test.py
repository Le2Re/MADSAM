import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_khanhha import Khanhha_dataset
import time

def inference(args, multimask_output, model, test_save_path=None):
    db_test = Khanhha_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    inference_time = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0] # tensor
        metric_i,a= test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        inference_time = inference_time+a
        logging.info('idx %d case %s mean_pr %f mean_re %f mean_f1 %f  mean_iou %f' % (
            i_batch, case_name, metric_i[0], metric_i[1],metric_i[2], metric_i[3]))
    metric_list = metric_list / len(db_test)
    print(len(db_test))
    print(f"Average Inference time: {inference_time/len(db_test)*1000:.4f} ms")
    logging.info('Testing performance in best val model: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f' % (metric_list[0], metric_list[1],metric_list[2], metric_list[3]))
    logging.info("Testing Finished!")
    return 1


def transfromnormal(cx, cy, w, h):
    cx *= 448
    cy *= 448
    w *= 448
    h *= 448
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return [xmin,ymin,xmax,ymax]

def get_prompt_dic(folder_path = "path/to/your/txt_files"):

    bbox_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                bboxes = []
                for line in file:
                    components = line.split()
                    cx, cy, w, h = [float(x) for x in components[1:]]
                    bbox = transfromnormal(cx, cy, w, h)
                    bboxes.append(bbox)
                file_key = os.path.splitext(filename)[0]
                bbox_dict[file_key] = bboxes

    return bbox_dict

def inference_prompt(args, multimask_output, db_config, model, test_save_path=None, prompt_path='/data/x3/annotation_yolo'):#提示的路径
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    inference_time = 0
    bbox_dict = get_prompt_dic(prompt_path)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0] # tensor
        
        metric_i,a= test_single_volume_prompt(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'], prompt_box=bbox_dict[os.path.splitext(case_name)[0]])
        
        metric_list += np.array(metric_i)
        inference_time = inference_time+a
        logging.info('idx %d case %s mean_pr %f mean_re %f mean_f1 %f  mean_iou %f' % (
            i_batch, case_name, metric_i[0], metric_i[1],metric_i[2], metric_i[3]))
    metric_list = metric_list / len(db_test)
    print(len(db_test))
    print(f"Average Inference time: {inference_time/len(db_test)*1000:.4f} ms")
    logging.info('Testing performance in best val model: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f' % (metric_list[0], metric_list[1],metric_list[2], metric_list[3]))
    logging.info("Testing Finished!")
    return 1

    
def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str)
    parser.add_argument('--num_classes', type=int, default=1, help='For crack segmentation, the output class should be 1')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_road420/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='./output_infer')
    parser.add_argument('--img_size', type=int, default=448, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=3407, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--snapshot', type=str, default='checkpoints/epoch_105.pth', help='model snapshot')
    


    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0], 
                                                                     pixel_std=[1, 1, 1])
    net = sam.cuda()
    total_params = sum(p.numel() for p in sam.prompt_generator.parameters())
    print("Total parameters:", total_params)
    
    if args.snapshot is not None:
        weights = torch.load(args.snapshot)
        net.load_state_dict(weights)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False 

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
        os.makedirs(test_save_path+ '/img/', exist_ok=True)
        os.makedirs(test_save_path+ '/pred/', exist_ok=True)
        os.makedirs(test_save_path+ '/gt/', exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, net, test_save_path)
    #prompt_path = "/data/x3/annotation_yolo"
    #inference_prompt(args, multimask_output, dataset_config[dataset_name], net, test_save_path, prompt_path)
	
