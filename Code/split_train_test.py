import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
df = pd.read_csv('./Build_error_logs_data/augmented_data2.csv')

# 分割数据集为训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存训练集和测试集到新的 CSV 文件
train_df.to_csv('./Build_error_logs_data/test_set2.csv', index=False)
test_df.to_csv('./Build_error_logs_data/test_set2.csv', index=False)

print("训练集和测试集已成功分割并保存为 train_set.csv 和 test_set.csv")