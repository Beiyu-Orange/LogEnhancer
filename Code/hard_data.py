import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def evaluate_classification(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')

    # 确保列名正确
    if 'label' not in df.columns or 'Predicted_Label' not in df.columns:
        raise ValueError("CSV 文件必须包含 'label' 和 'Predicted_Label' 列")

    # 只筛选标签6到标签10的数据
    df_filtered = df[df['label'].isin([6, 7, 8, 9, 10])]

    # 获取真实标签和预测标签
    y_true = df_filtered['label']
    y_pred = df_filtered['Predicted_Label']

    # 获取数据集中实际的类别标签
    labels = sorted(y_true.unique())

    # 计算分类报告
    report = classification_report(y_true, y_pred, labels=labels, target_names=[str(i) for i in labels], output_dict=True)

    # 计算总准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 输出分类报告
    print("                     precision    recall  f1-score   support")
    for i in labels:
        print(
            f"{i:>15} {report[str(i)]['precision']:10.2f} {report[str(i)]['recall']:10.2f} {report[str(i)]['f1-score']:10.2f} {int(report[str(i)]['support']):10d}")

    # 输出总准确率，调整对齐方式
    print(f"{'accuracy':>15} {' ':>21} {accuracy:10.2f} {len(y_true):10d}")

    # 输出 Macro Avg 和 Weighted Avg
    print(
        f"{'macro avg':>15} {report['macro avg']['precision']:10.2f} {report['macro avg']['recall']:10.2f} {report['macro avg']['f1-score']:10.2f} {len(y_true):10d}")
    print(
        f"{'weighted avg':>15} {report['weighted avg']['precision']:10.2f} {report['weighted avg']['recall']:10.2f} {report['weighted avg']['f1-score']:10.2f} {len(y_true):10d}")


# 示例调用
if __name__ == "__main__":
    csv_file = './Build_error_logs_data/dataset1/Semantic.csv'
    evaluate_classification(csv_file)
