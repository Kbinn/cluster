import pandas as pd
import os


input_files = [
    'Acinetobacter baumannii.csv',
    'Enterococcus faecium.csv',
    'Klebsiella pneumoniae.csv',
    'Pseudomonas aeruginosa.csv',
    'Staphylococcus aureus.csv'
]
output_dir = 'all'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"目錄 {output_dir} 已創建")
else:
    print(f"目錄 {output_dir} 已存在")


combined_data = pd.DataFrame()


for file_path in input_files:
    data = pd.read_csv(file_path)
    combined_data = pd.concat([combined_data, data], ignore_index=True)
    print(f"文件 {file_path} 已讀取並合併")


grouped = combined_data.groupby('cluster')
print(f"數據已按 cluster 分組")

for cluster, group in grouped:
    output_file_path = os.path.join(output_dir, f'cluster_{cluster}.csv')
    group.to_csv(output_file_path, index=False)
    print(f'Saved {output_file_path}')
