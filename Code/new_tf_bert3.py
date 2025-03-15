import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
from gensim.models import Word2Vec
import os

os.environ['CURL_CA_BUNDLE'] = ''


def get_word_embeddings(texts, word2vec_model):
    embeddings = []
    for text in texts:
        words = text.split()
        valid_words = [word for word in words if word in word2vec_model.wv]

        if len(valid_words) > 0:
            avg_embedding = np.mean([word2vec_model.wv[word] for word in valid_words], axis=0)
        else:
            avg_embedding = np.zeros(word2vec_model.vector_size)

        embeddings.append(avg_embedding)

    return np.array(embeddings)


def bert_tf():
    # 设置启用GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # 读取CSV文件
    # 读取 train_set.csv 文件，使用 open() 跳过解码错误
    with open('./Build_error_logs_data/dataset3/train_set.csv', 'r', encoding='utf-8', errors='ignore') as file:
        train_df = pd.read_csv(file)

    # 读取 test_set.csv 文件，使用 open() 跳过解码错误
    with open('./Build_error_logs_data/dataset1/test_set.csv', 'r', encoding='utf-8', errors='ignore') as file:
        test_df = pd.read_csv(file)

    train_df['log'] = train_df['log'].astype(str)
    train_df['label'] = train_df['label'].astype(int)

    test_df['log'] = test_df['log'].astype(str)
    test_df['label'] = test_df['label'].astype(int)

    # 划分数据集
    train_texts, train_labels = train_df['log'].values, train_df['label'].values
    test_texts, test_labels = test_df['log'].values, test_df['label'].values

    # 训练Word2Vec模型
    sentences = [text.split() for text in train_texts]
    word2vec_model = Word2Vec(sentences, vector_size=500, window=5, min_count=1, workers=4)
    word2vec_model.save("word2vec_model2")  # ??Word2Vec??

    # 获取训练集和测试集的词向量表示
    train_embeddings = get_word_embeddings(train_texts, word2vec_model)
    test_embeddings = get_word_embeddings(test_texts, word2vec_model)

    assert train_embeddings.shape[1] == test_embeddings.shape[1], "Embeddings have different dimensions!"

    # 加载Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 对训练集文本进行tokenize和padding
    encoded_train_texts = tokenizer.batch_encode_plus(
        train_texts,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,  # ?????
        return_attention_mask=True,
        return_tensors='tf'
    )

    # 对测试集文本进行tokenize和padding
    encoded_test_texts = tokenizer.batch_encode_plus(
        test_texts,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,  # ?????
        return_attention_mask=True,
        return_tensors='tf'
    )

    # 将Word2Vec词向量与BERT输入拼接
    train_input_ids = np.array(encoded_train_texts['input_ids'])
    train_embeddings = np.array(train_embeddings)
    train_attention_mask = np.array(encoded_train_texts['attention_mask'])

    test_input_ids = np.array(encoded_test_texts['input_ids'])
    test_embeddings = np.array(test_embeddings)
    test_attention_mask = np.array(encoded_test_texts['attention_mask'])

    # 转换训练集标签为one-hot编码
    train_labels = tf.keras.utils.to_categorical(train_labels)

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
    history = model.fit(
        x=[train_input_ids, train_embeddings, train_attention_mask],
        y=train_labels,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        callbacks=[early_stopping]
    )

    model.save("my_model3")

    test_input_ids = tf.convert_to_tensor(test_input_ids, dtype=tf.int32)
    test_embeddings = tf.convert_to_tensor(test_embeddings, dtype=tf.float32)
    test_attention_mask = tf.convert_to_tensor(test_attention_mask, dtype=tf.int32)

    y_pred = np.argmax(model.predict([test_input_ids, test_embeddings, test_attention_mask]), axis=-1)

    # 输出classification_report
    print(classification_report(np.argmax(test_labels, axis=1), y_pred))

    test_df['predict_label'] = y_pred

    test_df.to_csv('result.csv', index=False)
    print("Prediction results saved to result.csv")


if __name__ == '__main__':
    bert_tf()
