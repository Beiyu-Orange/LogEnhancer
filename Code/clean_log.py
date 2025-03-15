import pandas as pd
import openai
from openai import OpenAI

# 设置 OpenAI API 密钥
API_KEY = 'sk-zPhmrBarKkWKWq8ABGJVZZ5B7hVWFAs7W45UbrXpcY47ARci'  # 请替换为你自己的 OpenAI API 密钥
openai.api_key = API_KEY

# 选择模型 ID
model_id = 'gpt-4o'  # 或者选择你希望使用的模型

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.chatanywhere.tech"  # 默认 OpenAI API 地址
)


# 清洗日志的函数
def clean_log_with_few_shot_and_cot(log_text):
    prompt = f"""
    你是一个日志清洗助手，任务是清洗错误日志，去除无关的内容，只保留关键信息。
    你的思考方式如下：
    1. 分析日志中的时间戳、堆栈信息等，识别出冗余或无关的部分。
    2. 提取出错误类型和关键信息。
    3. 输出清洗后的简洁日志。
    
    下面是我的输入示例log，以及你需要输出的结果cleaned_log，且只需要内容而不需要“cleaned_log：”作为前缀。

    示例1:
    log：2023-02-18 10:01:01,015 ERROR [main] - Stack trace: java.lang.NullPointerException at com.example.main.run(Main.java:15)
    cleaned_log：NullPointerException at com.example.main.run(Main.java:15)

    示例2:
    log：2023-02-18 10:05:25,200 WARNING [thread-1] - FileNotFoundException: /path/to/file not found
    cleaned：FileNotFoundException: /path/to/file not found

    错误日志内容：
    {log_text}
    """

    # 获取 API 响应
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "你是一个日志清洗助手。"},
            {"role": "user", "content": prompt}
        ]
    )

    cleaned_log = response.choices[0].message.content.strip()

    # 确保清洗后的日志以正确的编码进行处理
    cleaned_log = cleaned_log.encode('utf-8').decode('utf-8')  # 强制转换为 UTF-8 编码
    return cleaned_log


# 读取 CSV 文件并进行日志清洗
def clean_logs(input_csv_path, output_csv_path):
    # 读取 CSV 文件
    with open(input_csv_path, 'r', encoding='utf-8-sig', errors='ignore') as file:
        df = pd.read_csv(file)

    # 检查输入数据是否包含 'label' 和 'log' 列
    if 'label' not in df.columns or 'log' not in df.columns:
        raise ValueError("CSV 文件必须包含 'label' 列和 'log' 列")

    # 清洗每一条日志
    df['cleaned_log'] = df['log'].apply(clean_log_with_few_shot_and_cot)

    # 保存结果到新的 CSV 文件
    # 保存 CSV 文件时也使用 'utf-8-sig' 编码
    df[['label', 'cleaned_log']].to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"清洗完成，结果已保存到 {output_csv_path}")


# 调用函数进行日志清洗
if __name__ == "__main__":
    input_csv = "./Build_error_logs_data/dataset2/train_set - 副本.csv"  # 输入的 CSV 文件路径
    output_csv = './Build_error_logs_data/dataset2/cleaned_logs.csv'  # 输出的 CSV 文件路径
    clean_logs(input_csv, output_csv)
