#! /bin/bash
HERE=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
source $HERE/conf.sh
fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310"
fs_ugi="idl-vrt,VRTData@2016"
hadoop_dir=/app/idl/users/vrt/yanzhaoyi/pth_crowd_counting
function download() {
    remote_db=${hadoop_dir}/$1
    local_db=$2

    ret=1
    for ((i=0; i<5; i++))  # try 5 times
    do
        hadoop fs -Dfs.default.name=${fs_name} -Dhadoop.job.ugi=${fs_ugi} -test -e ${remote_db}
        if [ $? -ne 0 ]; then
            echo -e "${remote_db} not exist in hdfs"
            ret=1
            break
        fi
        hadoop fs -Dfs.default.name=${fs_name} -Dhadoop.job.ugi=${fs_ugi} -get ${remote_db} ${local_db}
        if [ $? -eq 0 ]; then
            echo -e "Download File: ${remote_db}"
            ret=0
            break
        fi
    done

    return $ret
}

function upload() {
    local_db=$1
    remote_db=${hadoop_dir}/$2

    ret=1
    for ((i=0; i<5; i++))  # try 5 times
    do
        hadoop fs -Dfs.default.name=${fs_name} -Dhadoop.job.ugi=${fs_ugi} -test -e ${remote_db}
        if [ $? -eq 0 ]; then
            echo -e "${remote_db} already exit in hdfs"
            ret=0
            break 
        fi
        hadoop fs -Dfs.default.name=${fs_name} -Dhadoop.job.ugi=${fs_ugi} -put ${local_db} ${remote_db}
        if [ $? -eq 0 ]; then
            echo -e "Upload File: ${local_db}"
            ret=0
            break 
        fi
    done

    return $ret
}

option=$1
if [ $option == "download" ]; then
    download $2 $3
    exit $?
elif [ $option == "upload" ]; then
    upload $2 $3
    exit $?
else
    exit 1
fi
