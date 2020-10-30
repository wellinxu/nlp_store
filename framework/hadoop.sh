"""
hadoop的简单使用
"""

remote_file=""     # 远程文件名
local_file=""    # 本地文件名


#查看文件夹
hadoop fs -ls $remote_file

#查看文件
hadoop fs -cat $remote_file | head -5

#创建文件夹
hadoop fs -mkdir $remote_file

#下载数据
hadoop fs -get $remote_file $local_file
#上传数据
hadoop fs -put $local_file $remote_file    #如果远程文件夹不存在，最好先创建文件夹

#删除文件夹
hadoop fs -rm -r -skipTrash $remote_file
#删除文件
hadoop fs -rm -skipTrash $remote_file
