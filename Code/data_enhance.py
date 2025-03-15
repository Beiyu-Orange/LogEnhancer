import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# 检查并下载所需的nltk数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("下载nltk数据...")
    nltk.download('punkt')
    nltk.download('wordnet')


# 定义同义词替换函数
def synonym_replacement(text, n=2):
    """
    使用同义词替换文本中的词语。
    :param text: 原始文本
    :param n: 替换的词语数量
    :return: 增强后的文本
    """
    words = word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)


def get_synonyms(word):
    """
    获取一个词语的同义词列表。
    :param word: 目标词语
    :return: 同义词列表
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


# 定义随机删除函数
def random_deletion(text, p=0.2):
    """
    随机删除文本中的词语。
    :param text: 原始文本
    :param p: 删除概率
    :return: 增强后的文本
    """
    words = word_tokenize(text)
    if len(words) == 1:
        return text
    remaining = [word for word in words if random.uniform(0, 1) > p]
    if len(remaining) == 0:
        return random.choice(words)
    return ' '.join(remaining)


# 定义随机交换函数
def random_swap(text, n=2):
    """
    随机交换文本中的词语顺序。
    :param text: 原始文本
    :param n: 交换次数
    :return: 增强后的文本
    """
    words = word_tokenize(text)
    if len(words) <= 1:
        return text
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


# 定义数据增强函数
def augment_data(text, n_augment=5):
    """
    对文本进行数据增强。
    :param text: 原始文本
    :param n_augment: 增强次数
    :return: 增强后的文本列表
    """
    augmented_texts = []
    for _ in range(n_augment):
        # 随机选择一种增强方法
        method = random.choice([synonym_replacement, random_deletion, random_swap])
        augmented_text = method(text)
        augmented_texts.append(augmented_text)
    return augmented_texts


# 加载CSV文件
def load_data(file_path):
    """
    加载CSV文件。
    :param file_path: 文件路径
    :return: DataFrame
    """
    df = pd.read_csv(file_path)
    return df


# 对E6-E10的样本进行数据增强
def augment_e6_e10(df, n_augment=5):
    """
    对E6-E10的样本进行数据增强。
    :param df: 原始数据
    :param n_augment: 每个样本增强的次数
    :return: 增强后的数据
    """
    augmented_data = []
    for index, row in df.iterrows():
        if row['label'] >= 6:  # 只对E6-E10的样本进行增强
            text = row['log']
            augmented_texts = augment_data(text, n_augment=n_augment)
            for augmented_text in augmented_texts:
                augmented_data.append({'label': row['label'], 'log': augmented_text})
    return pd.DataFrame(augmented_data)


# 保存增强后的数据
def save_augmented_data(df, output_file):
    """
    保存增强后的数据到CSV文件。
    :param df: 增强后的数据
    :param output_file: 输出文件路径
    """
    df.to_csv(output_file, index=False)


# 主函数
def main(input_file, output_file, n_augment=5):
    """
    主函数。
    :param input_file: 输入CSV文件路径
    :param output_file: 输出CSV文件路径
    :param n_augment: 每个样本增强的次数
    """
    # 加载数据
    df = load_data(input_file)

    # 对E6-E10的样本进行数据增强
    augmented_df = augment_e6_e10(df, n_augment=n_augment)

    # 将增强后的数据与原始数据合并
    final_df = pd.concat([df, augmented_df], ignore_index=True)

    # 随机打乱数据
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存增强后的数据
    save_augmented_data(final_df, output_file)
    print(f"数据增强完成，增强后的数据已保存到 {output_file}")


# 运行脚本
if __name__ == '__main__':
    input_file = './Build_error_logs_data/data.csv'  # 输入CSV文件路径
    output_file = './Build_error_logs_data/augmented_data.csv'  # 输出CSV文件路径
    n_augment = 5  # 每个样本增强的次数
    main(input_file, output_file, n_augment=n_augment)