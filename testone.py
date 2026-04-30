import os
import sys
from tqdm import tqdm
import numpy as np
import random
import numpy as np
import torch
from segment_anything import sam_model_registry
import cv2
from PIL import Image
sam, img_embedding_size = sam_model_registry['vit_b'](image_size=448,
                                                                    num_classes=1,
                                                                    checkpoint='checkpoints/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0], 
                                                                     pixel_std=[1, 1, 1])
sam = sam.cuda()
weights = torch.load('checkpoints/epoch_105.pth')
sam.load_state_dict(weights)
image = cv2.imread('../Road420/images/2023_10_30_15_44_IMG_5888.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0
image = torch.from_numpy(image).float()
image = image.permute(2, 0, 1)
image = image.unsqueeze(0)
image = image.numpy()
print(image.shape)
inputs = torch.from_numpy(image).float().cuda()
sam.eval()
outputs = sam(inputs, False, 448, (448,448))
output_masks = outputs['masks']
out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)  # h,w
prediction = out.cpu().detach().numpy()
prediction = prediction*255
pred = Image.fromarray(prediction.astype(np.uint8))
pred.save('2023_10_30_15_44_IMG_5888' + "_img.jpg")
