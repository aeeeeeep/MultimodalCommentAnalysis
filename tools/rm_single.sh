#!/bin/bash  
  
# 遍历data目录下的所有文件  
for file in users/*; do  
  # 判断是否为文件  
  if [ -f "$file" ]; then  
    # 获取文件行数  
    lines=$(wc -l < "$file")  
    # 如果行数小于等于1行，则删除该文件  
    if [ "$lines" -le 1 ]; then  
      echo "Deleting $file ..."  
      rm -f "$file"  
    fi  
  fi  
done
