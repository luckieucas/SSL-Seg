#!/bin/bash

# 检查是否提供了需要处理的文件夹路径
if [ -z "$1" ]
then
  echo "请提供需要处理的文件夹路径。"
  exit 1
fi

# 获取输入的文件夹路径，并在末尾添加文件名
output_file="${1%/}/results.txt"

# 定义输入参数数组
args=()

# 遍历指定文件夹中的所有文件
for file in "$1"/model1_iter*.pth
do
  # 判断文件是否是普通文件
  if [ -f "$file" ]
  then
    # 获取文件名和文件路径
    file_name=$(basename "$file")
    
    # 利用正则表达式提取文件名中的第二个数字
    num=$(echo "$file_name" | grep -oE '[0-9]+' | sed -n '2p')
    # 利用正则表达式提取文件名中的第一个数字
    #num=$(echo "$file_name" | grep -oE '[0-9]+' | sed -n '1p')
    
    # 比较数字是否大于10000
    if [ "$num" -gt 25000 ]
    then
      # 判断是否已经存在相同的数字的文件，如果存在则只保留修改时间最新的那个文件
      if [ "${file_map[$num]}" ]
      then
        # 获取已有文件的修改时间和当前文件的修改时间
        existing_file="${file_map[$num]}"
        existing_file_mtime=$(stat -c %Y "$existing_file")
        current_file_mtime=$(stat -c %Y "$file")
        
        # 比较修改时间，保留修改时间最新的文件
        if [ "$current_file_mtime" -gt "$existing_file_mtime" ]
        then
          file_map[$num]="$file"
        fi
      else
        file_map[$num]="$file"
      fi
    fi
  fi
done

# 将修改时间最新的文件名作为输入参数添加到数组中
for file in "${file_map[@]}"
do
  if [ "$file" ]
  then
    args+=("$(realpath "$file")")
  fi
done

# 按顺序运行test.py脚本，并将输入参数传递给脚本
for arg in "${args[@]}"
do
  echo "当前测试的模型是: $arg"
  python test.py --model_path "$arg" --gpu 0 | tee -a "$output_file"
done