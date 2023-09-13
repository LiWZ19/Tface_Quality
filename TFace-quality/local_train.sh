#!/bin/bash  #注明这是一个bash文件

if [ ! -d "logs" ]; then   #如果不存在“logs”文件夹
    mkdir logs         #就创建
fi
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'            #引进该变量
nohup python -u train.py > logs/$(date +%F-%H-%M-%S).log 2>&1 &          #将train.py训练结果引进log文件里
