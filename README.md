# Pytorch--mask-rcnn

For modify the original mask rcnn model.
Please connect the models in the model/modified_mask_rcnn.py

# Make sure the environment is the same !!
## env
```
python=3.7.7
pytorch=1.6.0=py3.7_cuda10.1.243_cudnn7.6.3_0
torchvision=0.7.0=py37_cu101
numpy=1.19.1=py37hbc911f0_0
```
## conda env setting
```Shell
conda env create -f environment.yml -n rcnn
```
## start conda
```Shell
conda activate rcnn
```

# Run the pedestrian dataset to make sure your model works
```
./download_pedestrian.sh
python test_pedestrian.py
```

# Mainly training on PASCAL
When you want to train on pascal voc.
you don't need to run the .sh file because it is a built-in function in voc_utils.
```
python train_voc.py
```

## Problem shooting
1. cannot run .sh file 
```
chmod 777 YOUR_SH_FILE_NAME.sh
```
# For running test_coco.py in distribute system
## run distribute
```Shell
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_coco.py
python -m torch.distributed.launch --nproc_per_node= --use_env train_coco\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_coco\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```
## kill the nvidia zombie threads
```Shell
kill $(ps aux | grep train_coco.py | grep -v grep | awk '{print $2}') 
```

