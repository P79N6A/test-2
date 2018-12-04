#!/bin/bash
#hup指令后台运行：nohup sh transfer_files.sh > /home/coswadmin/he/scplog/scp.log 2>&1 &
#查询当前后台执行脚本的pid：ps -ef|grep 'transfer_files.sh'
#停止进程：kill -9 pid
 
#解决shell脚本中SCP命令需要输入密码的问题：http://blog.csdn.net/chris_playnow/article/details/22579139
 
#定义变量值
training_model="SH_A_SPData_Main_Pretrained_CSRNet_Refine_L2_upto_L3_ResFusion_refineEndCSRNet_refineW_1"
folder=/home/vis/yanzhaoyi/code/crowd_counting/experiments/crowd_new_idea/crowd_new/output/${training_model}
 
now=$(date '+%Y-%m-%d %H:%M:%S')
 
#log folder
log_dir="/home/vis/yanzhaoyi/scplog"
log_file="$log_dir/log_${training_model}_${now}.log"
#log_file="$log_dir/scp.log"
 
#对账文件备份目录, no backup, as I want to save storage.
#bak_dir='/home/coswadmin/he/checkfile_bak'
 
#--parents,此选项后，可以是一个路径名称。
#若路径中的某些目录尚不存在，系统将自动建立好那些尚不存在的目录。
#即一次可以建立多个目录。
mkdir -p $log_dir
#mkdir -p $bak_dir
 
#进入ftp对账文件目录
cd $folder
 
#统计当前文件夹下对账文件数量，并赋值到fileNum
fileNum=$(ls -l |grep "^-"|wc -l)
 
while true
do
      now=$(date '+%Y-%m-%d %H:%M:%S')
      fileNum=$(ls -l |grep "^-"|wc -l)
      #如果文件数量大于0，则说明存在对账文件，执行文件移动操作，将文件移动到另一台服务器
      if [ $fileNum -gt 2 ] # exclude two 'txt' files.  
      then   
         #遍历当前文件夹，输出其下文件名,下面移动方式会将文件夹一起进行移动
         # !!!!!!!!!!!!!!!!!! Change here to make it only transfer .pth!
         # As loss_log.txt should not remove
         for file_a in $folder/*.pth; do 
             echo -e $now' 开始移动对账文件' >> $log_file         
             temp_file=`basename $file_a` 
             #1、文件名输入到文件        
             echo $temp_file     >> $log_file    
             #2、文件移动到指定服务器scp，
             scp $temp_file vis@szwg-idl-gpu-online11.szwg01.baidu.com:/home/vis/yanzhaoyi/code/crowd_counting/experiments/crowd_new_idea/crowd_new/output/${training_model}/
             #3、文件移动到备份文件夹
             exec rm $temp_file &
             echo -e $now' 对账文件移动结束' >> $log_file  
             echo -e $now' Delete file...' >> $log_file  
         done 
      else  
         echo $now' 当前没有需要移动的对账文件' >> $log_file  
      fi
      #休眠1 minute      
      sleep 60
done
 
echo -e '' >> $log_file
 
 
 
 
 
 


