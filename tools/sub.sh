#!/bin/bash

input_file="../Books_5.json"
output_file_prefix=""

if [ ! -f "$input_file" ]; then
  echo "输入文件 $input_file 不存在。"
  exit 1
fi

output_dir="../data"
mkdir -p "$output_dir"

total_lines=$(wc -l < "$input_file")
lines_per_file=$((total_lines / 5))  
remainder=$((total_lines % 5))      

split -l $lines_per_file "$input_file" "$output_dir/$output_file_prefix"

file_count=1
for file in "$output_dir/$output_file_prefix"*; do
  if [ $file_count -eq 5 ]; then
    tail -n +$((lines_per_file * (file_count - 1) + 1)) "$input_file" >> "$file"
  fi

  mv "$file" "$output_dir/$output_file_prefix$file_count.json"
  file_count=$((file_count + 1))
done

echo "文件分割完成。"
