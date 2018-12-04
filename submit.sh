#! /bin/bash
HERE=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
source $HERE/conf.sh
#export user_name=IDL_Data
#export submitter=zhangchengyue01

export http_proxy=
export https_proxy=

hadoopworkdir=/user/vis-finegrained/$submitter/$jobname/

hdfs_addr=afs://xingtian.afs.baidu.com:9902
export user_name=vis-finegrained
export password=vis-finegrained_passwd421
 

/home/vis/yanzhaoyi/app/.hgcp_p40/software-install/HGCP_client/tools/hadoop-v2/hadoop/bin/hadoop fs -Dfs.default.name=$hdfs_addr -Dhadoop.job.ugi=$user_name,$password -Dfs.afs.impl=org.apache.hadoop.fs.DFileSystem -mkdir -p $hadoopworkdir


#清空删除hadoop工作目录
# hadoop fs -Dfs.default.name=$hdfs_addr -Dhadoop.job.ugi=$user_name,$password -rmr $hadoopworkdir

#本地创建文件夹，避免不必要的麻烦 
if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

#提交训练包
sh -x /home/vis/yanzhaoyi/app/.hgcp_p40/software-install/HGCP_client/bin/submit \
        --hdfs afs://xingtian.afs.baidu.com:9902 \
        --hdfs-user vis-finegrained --hdfs-passwd vis-finegrained_passwd421 \
        --hdfs-path $hadoopworkdir \
        --file-dir ./ \
        --job-name $jobname \
        --queue-name  ${cluster_name} \
        --num-nodes $nodenum --gpu-pnode ${gpu_pnode} \
        --submitter ${submitter} --num-task-pernode ${gpu_pnode} \
	    --time-limit 0 \
     	--job-script ./run_sp.sh
