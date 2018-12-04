#!/usr/bin/env bash
HERE=`pwd`
source $HERE/conf.sh

export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64:./lib:$LD_LIBRARY_PATH
#export hdfs_dataset=datasets_shift.tar
#export hdfs_dataset2=coco_train2014.zip
export hdfs_dataset3=data_large_in_one.tar
export hdfs_anaconda=anaconda2/anaconda2.tar

# I only need to make this folder
# As I copy the file not the folder
if [ ! -d pretrained_models ]; then
    mkdir pretrained_models
fi

# Download data from hadoop to ./lmdb
sh ./hdfs.sh download ${hdfs_anaconda}   ./
#sh ./hdfs.sh download ${hdfs_dataset}  ./
#sh ./hdfs.sh download ${hdfs_dataset2}  ./
sh ./hdfs.sh download ${hdfs_dataset3}  ./


tar -xvf anaconda2.tar
#tar -xvf datasets_shift.tar
tar -xvf data_large_in_one.tar
#unzip coco_train2014.zip

rm anaconda2.tar
#rm datasets_shift.tar
#rm coco_train2014.zip
rm data_large_in_one.tar


# add path of anaconda
export PATH=${HERE}/anaconda2/bin:$PATH


# coco2014: train2014

python train.py --name='places_skip' --gpu_ids='0' --skip=1 \
                --nThreads=2 \
                --dataroot='./data_large_in_one' \
                --save_epoch_freq=1 \
                --niter=10 \
                --print_freq=500 \
                --which_model_netG='unet_shift_triple' \
                --batchSize=1 2>&1 | tee log/Shift_base.log

