import pandas as pd
import openai
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# API_KEY =

openai.api_key = API_KEY
# gpt-4
model_id = 'gpt-4o'
# model_id = 'deepseek-ai/DeepSeek-V3'

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=API_KEY,
    base_url="https://api.chatanywhere.tech"  # chatgptfree
    # base_url="https://api.deepseek.com"    #deepseek
    # base_url="https://api.siliconflow.cn/v1"    #硅基流动
)


def classify_logs(file_path, output_path):

    # 读取 CSV 文件
    df = pd.read_csv(file_path, encoding='GBK')

    # 确保数据包含 'log' 列和 'label' 列
    if 'log' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV 文件必须包含 'log' 列和 'label' 列")


    def get_label(log_text):
        prompt = f"""
        根据以下错误日志，将其分类到正确的错误类别 (1-10):

        分类标准如下：
        1) 构建配置问题(E1)：依赖缺失：日志特征通常包含有关“依赖”缺失模块、构建依赖关系失败等的信息。相关版本不匹配：其错误日志签名通常包含“依赖” >= 关于版本的信息。
        2) 测试用例问题(E2)：测试用例执行失败：错误日志特征通常包括测试失败、测试错误、测试失败等。
        3) 不兼容问题(E3)：编译器不兼容：使用的编译器版本与代码不兼容，导致编译错误。架构不兼容：它的日志特征通常包括Arch、架构和其他信息。函数更新：日志通常包含有关已更新、弃用等函数的信息。文件冲突: 它的日志通常包含文件冲突等信息。
        4) 源代码问题(E4)：变量 / 函数等，未定义: 它们的日志通常包含关于未定义变量的信息等。函数滥用: 它的错误日志通常包含与函数错误相关的信息。代码拼写 / 语法失败：错误日志通常包含诸如变量错误之类的信息。
        5) 文件 / 目录缺少问题(E5)：构建过程无法找到相关的文件或目录：错误日志特征通常包含诸如No such file or directory.之类的信息。
        6) 内存问题(E6)：内存溢出，其错误日志特征通常包含与内存、Kill进程等相关的信息。
        7) 超时问题(E7)：程序超时，执行超过时间限制等。
        8) 插件问题(E8)：插件无法编译、插件错误等。
        9) 网络问题(E9)：无法下载依赖，网络相关错误信息。
        10) 其他问题(E10)：无法归类到以上类别的错误。
        
        请根据日志的内容，选择合适的错误类别编号 (1-10)，并**仅返回编号**（整数），例如: 3。
        **注意你的回答只能是一个数字，即分类编号**。

        错误日志内容：
        {log_text}
        """

        response = client.chat.completions.create(
            model=model_id,  # 或者选择其他可用模型
            messages=[{"role": "system", "content": "你是一个精确的日志分类助手。"},
                      {"role": "user", "content": prompt}]
        )
        message_content = response.choices[0].message.content.strip()
        return int(message_content)

    # 步骤二：基于清洗后的日志生成预测标签
    df['Predicted_Label'] = df['log'].apply(get_label)

    # 计算评估指标
    y_true = df['label']
    y_pred = df['Predicted_Label']

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted',
                                                               labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print(f"准确率 (ACC): {accuracy:.4f}")
    print(f"精确度 (P): {precision:.4f}")
    print(f"召回率 (R): {recall:.4f}")
    print(f"F1值 (F1): {f1:.4f}")

    # 保存新的CSV文件
    df.to_csv(output_path, index=False)
    print(f"分类完成，结果已保存到 {output_path}")


# 示例调用
classify_logs("./Build_error_logs_data/dataset1/test_set.csv", "./Build_error_logs_data/dataset1/gpt4o_new.csv")
