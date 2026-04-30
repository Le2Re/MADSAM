import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
import os
class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target) 
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    try: 
        pred.max() <=1 and pred.min()>=0 and gt.max() <=1 and gt.min()>=0
    except:
        print("Please Ensure max(pred) <=1 and min(pred)>=0 and max(gt) <=1 and min(gt)>=0 !!!")

    A = pred.sum() 
    B = gt.sum() 

    if A > 0 and B > 0:

        AinterB = pred&gt
        AunionB = (pred | gt) 
        AinterB = AinterB.sum()
        AunionB = AunionB.sum()
        dice = 2*AinterB/(A+B) 
        iou = AinterB /AunionB
        precision= AinterB/A
        recall = AinterB/B
        
        return precision, recall, dice, iou
    elif A >0 and B ==0.0: # For non-crack images
        return 0.0,0.0,0.0,0.0
    elif A == 0.0 and B == 0.0: # For non-crack images
        return 1.0,1.0,1.0,1.0
    elif A ==0.0 and B >0:
        return 0.0,0.0,0.0,0.0



def test_single_volume(image, label, net, classes, multimask_output, patch_size=[448, 448], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, label = image.cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() # image: 1,c,h,w label: h,w
    if len(image.shape) == 4:
        print(image.shape)
        prediction = np.zeros_like(label)
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1,1,patch_size[0] / x, patch_size[1] / y), order=3)  # ndarray   
        inputs = torch.from_numpy(image).float().cuda() #修改这里，inputs每张图片加入提示，格式看Sam类里的forward有说明
        net.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = net(inputs, multimask_output, patch_size[0]) # inputs 1,c,h,w
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time*1000:.4f} ms")
            output_masks = outputs['masks']  # 1,2,h,w
            
            #out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0) # h,w
            #prediction = out.cpu().detach().numpy()# h,w  
            
            probabilities = torch.softmax(output_masks, dim=1).squeeze(0)  #2,h,w
            prediction = probabilities[1]
            threshold = 0.47
            prediction = (prediction > threshold).int()
            prediction = prediction.cpu().detach().numpy()
            
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0) 

    metric_test=calculate_metric_percase(prediction , label)  # ndarray

    if test_save_path is not None:
        # image: 1,c,h,w  ndarray
        image = image*255
        label = label*255
        prediction = prediction*255
        image = Image.fromarray(np.transpose(image.squeeze(0), (1, 2, 0)).astype(np.uint8)) # 1,c,h,w -> h,w,c
        image.save(test_save_path + '/img/' + case + "_img.jpg")
        # pred h,w   ndarray
        pred = Image.fromarray(prediction.astype(np.uint8)) # h,w 
        pred.save(test_save_path + '/pred/' + case + "_img.jpg")
        # label h,w  ndarray
        label = Image.fromarray(label.astype(np.uint8)) # h,w 
        label.save(test_save_path + '/gt/' + case + "_img.jpg")

    return metric_test,inference_time

def save_single_promptmask(output_folder, prediction):
    os.makedirs(output_folder, exist_ok=True)

    # 保存每个二值图
    for i in range(prediction.size(0)):
        # 将二值图从张量转换为 PIL 图像格式
        binary_image = prediction[i].cpu().numpy().astype("uint8") * 255  # 转换为 0 或 255
        img = Image.fromarray(binary_image)

        # 保存图像
        img.save(os.path.join(output_folder, f"prediction_{i}.jpg"))
	
def test_single_volume_prompt(image, label, net, classes, multimask_output, patch_size=[448, 448], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1, prompt_box=[]):
    image, label = image.cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # image: 1,c,h,w label: h,w
    len_prompt = len(prompt_box)
    if len(image.shape) == 4:
        prediction = np.zeros_like(label)
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1, 1, patch_size[0] / x, patch_size[1] / y), order=3)  # ndarray
        inputs = torch.from_numpy(image).float().cuda()     #input include prompt
        if len_prompt != 0:
            Prompt_box = torch.tensor(prompt_box, dtype=torch.float32, device=inputs.device)
            inputs = inputs.squeeze(0)
            inputs = [{'image':inputs, 'boxes':Prompt_box }]
        
        net.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = net(inputs, multimask_output, patch_size[0])  # inputs 1,c,h,w
            #if len_prompt != 0:
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time*1000:.4f} ms")	 	
            output_masks = outputs[0]['masks']  # n,2,h,w   n为提示的个数
            print(output_masks.size())
            #out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)  # h,w
            #prediction = out.cpu().detach().numpy()  # h,w

            probabilities = torch.softmax(output_masks, dim=1)  #n,2,h,w
            probabilities = probabilities[:, 1, :, :]           #n,h,w
            threshold = 0.47
            prediction = (probabilities > threshold).int()
            save_single_promptmask('/data/x3/CrackSAM/CrackSAM/save_prompt_single_mask', prediction)
            prediction = torch.any(prediction, dim=0).int() 
            prediction = prediction.cpu().detach().numpy()
            
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    metric_test = calculate_metric_percase(prediction, label)  # ndarray

    if test_save_path is not None:
        # image: 1,c,h,w  ndarray
        image = image * 255
        label = label * 255
        prediction = prediction * 255
        image = Image.fromarray(np.transpose(image.squeeze(0), (1, 2, 0)).astype(np.uint8))  # 1,c,h,w -> h,w,c
        image.save(test_save_path + '/img/' + case + "_img.jpg")
        # pred h,w   ndarray
        pred = Image.fromarray(prediction.astype(np.uint8))  # h,w
        pred.save(test_save_path + '/pred/' + case + "_img.jpg")
        # label h,w  ndarray
        label = Image.fromarray(label.astype(np.uint8))  # h,w
        label.save(test_save_path + '/gt/' + case + "_img.jpg")

    return metric_test,inference_time
