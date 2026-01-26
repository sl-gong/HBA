#!/usr/bin/env python3

import os
import re
import struct

def fix_pcd_file(file_path):
    # 读取文件内容
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # 将内容分为头和二进制数据
    data_start = content.find(b'DATA binary')
    if data_start == -1:
        print(f"Error: Could not find 'DATA binary' in {file_path}")
        return False
    
    # 计算头的结束位置
    header_end = data_start + len(b'DATA binary') + 1  # +1 for newline
    
    # 提取头文本
    header_text = content[:header_end].decode('utf-8')
    
    # 提取POINTS值
    points_match = re.search(r'POINTS\s+(\d+)', header_text)
    if not points_match:
        print(f"Error: Could not find POINTS in {file_path}")
        return False
    
    points = points_match.group(1)
    
    # 修改WIDTH和HEIGHT
    new_header = re.sub(r'WIDTH\s+\d+', f'WIDTH {points}', header_text)
    new_header = re.sub(r'HEIGHT\s+\d+', 'HEIGHT 1', new_header)
    
    # 将头转换回字节
    new_header_bytes = new_header.encode('utf-8')
    
    # 组合新文件内容
    new_content = new_header_bytes + content[header_end:]
    
    # 写回文件
    with open(file_path, 'wb') as f:
        f.write(new_content)
    
    print(f"Fixed {file_path}: WIDTH={points}, HEIGHT=1")
    return True

def main():
    pcd_dir = '/Users/gsl/work/slam/HBA/park/pcd'
    
    # 获取所有pcd文件
    pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.pcd')]
    
    print(f"Found {len(pcd_files)} PCD files")
    
    # 处理每个文件
    for pcd_file in pcd_files:
        file_path = os.path.join(pcd_dir, pcd_file)
        fix_pcd_file(file_path)
    
    print("All files processed!")

if __name__ == "__main__":
    main()