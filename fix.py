import pandas as pd
import csv

input_file = 'dataset_cleaned_final.csv'
output_file = 'dataset_ready.csv'

cleaned_lines = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # Nếu dòng bị bọc toàn bộ bởi dấu ngoặc kép
        if line.startswith('"') and line.endswith('"') and len(line) > 1:
            # Xóa ngoặc kép ở 2 đầu
            line = line[1:-1]
            # Thay thế các ngoặc kép kép (escape) thành ngoặc kép đơn chuẩn
            line = line.replace('""', '"')

        cleaned_lines.append(line)

# Ghi lại ra file mới
with open(output_file, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print("✅ Đã dọn dẹp xong! Hãy dùng file 'dataset_ready.csv' cho ứng dụng của bạn.")