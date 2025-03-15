import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # 用于SMOTE过采样
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
from gensim.models import Word2Vec
import os

os.environ['CURL_CA_BUNDLE'] = ''


def bert_tf():
    # 设置启用GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # 读取CSV文件
    df = pd.read_csv('./Build_error_logs_data/data.csv')

    df['log'] = df['log'].astype(str)
    df['label'] = df['label'].astype(int)

    # 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['log'].values, df['label'].values,
                                                                          test_size=0.2, random_state=42)

    # 训练Word2Vec模型
    sentences = [text.split() for text in train_texts]
    word2vec_model = Word2Vec(sentences, vector_size=500, window=5, min_count=1, workers=4)
    word2vec_model.save("word2vec_model2")  # 保存Word2Vec模型

    # 获取训练集和测试集的词向量表示
    def get_word_embeddings(texts):
        embeddings = []
        for text in texts:
            words = text.split()
            avg_embedding = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
            embeddings.append(avg_embedding)
        return np.array(embeddings)

    # 获取训练集和测试集的词向量表示
    train_embeddings = get_word_embeddings(train_texts)
    test_embeddings = get_word_embeddings(test_texts)

    sampling_strategy = {
        1: 485,
        2: 164,
        3: 146,
        4: 146,
        5: 146,
        6: 146,
        7: 146,
        8: 146,
        9: 146,
        10: 146
    }

    # 对标签进行SMOTE过采样
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)  # SMOTE过采样方法
    train_embeddings_resampled, train_labels_resampled = smote.fit_resample(train_embeddings, train_labels)

    # 加载Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 对训练集文本进行tokenize和padding
    encoded_train_texts = tokenizer.batch_encode_plus(
        train_texts,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,  # 添加这一行
        return_attention_mask=True,
        return_tensors='tf'
    )

    # 对测试集文本进行tokenize和padding
    encoded_test_texts = tokenizer.batch_encode_plus(
        test_texts,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,  # 添加这一行
        return_attention_mask=True,
        return_tensors='tf'
    )

    # 将Word2Vec词向量与BERT输入拼接
    train_inputs = [encoded_train_texts['input_ids'], train_embeddings_resampled, encoded_train_texts['attention_mask']]
    test_inputs = [encoded_test_texts['input_ids'], test_embeddings, encoded_test_texts['attention_mask']]

    # 转换训练集标签为one-hot编码
    train_labels_resampled = tf.keras.utils.to_categorical(train_labels_resampled)

    # 转换测试集标签为one-hot编码
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # 加载Bert模型
    bert = TFBertModel.from_pretrained('bert-base-uncased', from_pt=True)

    # 定义模型结构
    input_ids = tf.keras.layers.Input(shape=(128,), dtype='int32')
    embeddings = tf.keras.layers.Input(shape=(word2vec_model.vector_size,), dtype='float32')
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype='int32')
    bert_embedding = bert(input_ids, attention_mask)[1]
    concat_embedding = tf.keras.layers.concatenate([bert_embedding, embeddings])
    dropout = tf.keras.layers.Dropout(0.4)(concat_embedding)
    output = tf.keras.layers.Dense(11, activation='softmax')(dropout)
    model = tf.keras.models.Model(inputs=[input_ids, embeddings, attention_mask], outputs=output)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 定义提前停止训练策略
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 训练模型并添加提前停止策略
    history = model.fit(x=train_inputs,
                        y=train_labels_resampled,
                        batch_size=128,
                        epochs=10,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    model.save("my_model3")

    # 进行预测
    y_pred = np.argmax(model.predict(test_inputs), axis=-1)

    # 输出classification_report
    print(classification_report(np.argmax(test_labels, axis=1), y_pred))


if __name__ == '__main__':
    bert_tf()
