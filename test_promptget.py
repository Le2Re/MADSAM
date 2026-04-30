import os
import sys
from tqdm import tqdm
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
    i=0
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

prompt_path = "/data/x3/DeepCrack"
bbox_dict = get_prompt_dic(prompt_path)
print(bbox_dict)
    