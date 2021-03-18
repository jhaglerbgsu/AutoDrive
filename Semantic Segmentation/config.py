import torch

# Basic configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = "SETR-PUP" 
# net = "SETR-MLA" 
# net = "TransUNet-Base" 
# net = "TransUNet-Large" 
net = "UNet"

# data 
data_dir = "./data/cityscapes"
IMG_DIM = 256
CLASS_NUM = 13

# training 
lrate = 0.01
momentum = 0.9
print_freq = 100
tensorboard_freq = 200
wdecay = 1e-4
fine_tune_ratio = 0.8
early_stop_tolerance = 10 #4
is_continue = False
batch_size = 16
ckpt_src = "./checkpoints/{0}/best_ckpt.pth".format(net)
iteration_num = 80000
epoch_num = 40
# epochs num is determined based on number of iterations and dataloader length.

# inference
best_ckpt_src = "./checkpoints/{0}/best_ckpt_1.pth".format(net)