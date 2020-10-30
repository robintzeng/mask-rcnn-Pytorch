# Pytorch--mask-rcnn

For modify the original mask rcnn model
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

## run distribute
```Shell
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_coco.py
python -m torch.distributed.launch --nproc_per_node= --use_env train_coco.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_coco.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```
## kill the nv zombie threads
```Shell
kill $(ps aux | grep train_coco.py | grep -v grep | awk '{print $2}') 
```

## Using torchvision to finetune mask-rcnn
```
./download_pedestrian.sh
python test_pedestrian.py
```

## TODO:
PASCAL data loader <br>
refine the .sh file in download PASCAL

