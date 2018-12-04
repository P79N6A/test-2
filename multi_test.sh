#!/bin/bash
name='256C_TC_TS_g1_IN_cos32_gan0.2'

which_model_netG='unet_shift_triple_MostAdvCos' # mind this setting

step=2
final_epoch=40

j=$step
count=1

while [ $j -le $final_epoch ];do
{
echo "Testing epoch : "${j}
j=$(($count * $step))

python test.py --which_model_netG=${which_model_netG} --name=${name} --which_epoch=${j}
let count++
}
done
