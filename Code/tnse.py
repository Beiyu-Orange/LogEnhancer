import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# 更改 Matplotlib 后端，解决 'tostring_rgb' 错误
matplotlib.use('TkAgg')  # 或者 'Agg', 'Qt5Agg' 等

# 设置支持中文的字体，避免字体缺失警告
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 支持中文的字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据集
df = pd.read_csv("./Build_error_logs_data/dataset2/train_set - 副本.csv")  # 修改为你的分类日志数据路径

# 确保数据包含 'log' 和 'label' 列
if 'log' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV 文件必须包含 'log' 和 'label' 列")

# 使用 TF-IDF 将日志文本转化为向量特征
vectorizer = TfidfVectorizer(max_features=500)  # 限制最大特征数
X = vectorizer.fit_transform(df['log'])

# 将标签提取为整数标签
y = df['label'].values

# 使用 PCA 先做一下降维（为了让 t-SNE 更高效）
pca = PCA(n_components=50)  # 降到 50 维
X_pca = pca.fit_transform(X.toarray())

# 运行 t-SNE 算法，降低到 2D
tsne = TSNE(n_components=2, random_state=42, n_iter=300)
X_tsne = tsne.fit_transform(X_pca)

# 创建 DataFrame 保存 t-SNE 结果
tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Label'] = y  # 添加标签信息

# 绘制 t-SNE 可视化图
plt.figure(figsize=(12, 10))
scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['Label'], cmap='tab10', s=50)

# 添加标题和标签
plt.title('t-SNE 可视化：日志分类', fontsize=16)
plt.xlabel('TSNE1', fontsize=12)
plt.ylabel('TSNE2', fontsize=12)

# 添加图例
plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(1, 11)], title="标签", loc="best")

# 显示图形
plt.show()

# # 计算分类报告，显示精确度、召回率和 F1 分数
# print(classification_report(y, df['Predicted_Label']))
