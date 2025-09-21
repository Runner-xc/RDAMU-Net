#!/usr/bin/zsh

for model in unet_3plus swin_unet res50_unet vim_unet; do

    python train.py --model $model 

done
