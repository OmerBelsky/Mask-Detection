import os
import argparse
import numpy as np
import pandas as pd
from utills import conv_net, to_gpu
from PIL import Image
import torchvision.transforms as transforms
import torch
from sklearn.metrics import accuracy_score

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
root = args.input_folder
files = os.listdir(root)


#####
image_width = 180
image_height =  180
cnn = conv_net()
cnn.load_state_dict(torch.load('cnn1.pkl'))
cnn = to_gpu(cnn)

test_transform = transforms.Compose([
    transforms.Resize((image_width,image_height)),
    transforms.ToTensor()])

y_pred = []
for f in files:
    im = Image.open(os.path.join(root, f))
    im = to_gpu(test_transform(im))
    outputs = cnn(im.unsqueeze(0))
    _, predicted = torch.max(outputs.data, 1)
    y_pred.append(predicted.cpu().item())


prediction_df = pd.DataFrame(zip(files, y_pred), columns=['id', 'label'])
# ####

# # TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)



