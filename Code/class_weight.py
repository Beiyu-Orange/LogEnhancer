import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
from gensim.models import Word2Vec
import os

os.environ['CURL_CA_BUNDLE'] = ''


def bert_tf():
    # 启用 GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # 读取数据
    df = pd.read_csv('./Build_error_logs_data/data.csv')
    df['log'] = df['log'].astype(str)
    df['label'] = df['label'].astype(int)

    # 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['log'].values, df['label'].values, test_size=0.2, random_state=42
    )

    # 计算 class_weight 之前，将标签转换为 0-9
    train_labels_for_weight = train_labels - 1

    # 计算 class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_for_weight),
        y=train_labels_for_weight
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    # 训练 Word2Vec 模型
    sentences = [text.split() for text in train_texts]
    word2vec_model = Word2Vec(sentences, vector_size=500, window=5, min_count=1, workers=4)
    word2vec_model.save("word2vec_model2")

    # 获取词向量
    def get_word_embeddings(texts):
        embeddings = []
        for text in texts:
            words = text.split()
            avg_embedding = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
            embeddings.append(avg_embedding)
        return np.array(embeddings)

    train_embeddings = get_word_embeddings(train_texts)
    test_embeddings = get_word_embeddings(test_texts)

    # 加载 BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize 训练集
    encoded_train_texts = tokenizer.batch_encode_plus(
        train_texts, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf'
    )

    # Tokenize 测试集
    encoded_test_texts = tokenizer.batch_encode_plus(
        test_texts, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf'
    )

    # 组合 Word2Vec 词向量和 BERT 输入
    train_inputs = [encoded_train_texts['input_ids'], train_embeddings, encoded_train_texts['attention_mask']]
    test_inputs = [encoded_test_texts['input_ids'], test_embeddings, encoded_test_texts['attention_mask']]

    # **转换标签为 one-hot（注意仍然使用 0-9 作为索引）**
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels - 1, num_classes=10)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels - 1, num_classes=10)

    # 加载 BERT 模型
    bert = TFBertModel.from_pretrained('bert-base-uncased', from_pt=True)

    # 定义模型
    input_ids = Input(shape=(128,), dtype='int32', name="input_ids")
    embeddings = Input(shape=(word2vec_model.vector_size,), dtype='float32', name="word2vec_embeddings")
    attention_mask = Input(shape=(128,), dtype='int32', name="attention_mask")

    bert_embedding = bert(input_ids, attention_mask)[1]
    concat_embedding = concatenate([bert_embedding, embeddings])
    dropout = Dropout(0.4)(concat_embedding)
    output = Dense(10, activation='softmax')(dropout)  # 10 类分类

    model = Model(inputs=[input_ids, embeddings, attention_mask], outputs=output)

    # 编译模型
    model.compile(
        optimizer=Adam(3e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 训练模型
    history = model.fit(
        x=train_inputs, y=train_labels_one_hot,
        batch_size=128, epochs=10, validation_split=0.1,
        class_weight=class_weights_dict  # **应用 class_weight**
    )

    model.save("my_model_class_weight")

    # 进行预测
    y_pred = np.argmax(model.predict(test_inputs), axis=-1)

    # **将预测的 0-9 转回 1-10**
    y_pred += 1
    true_labels = np.argmax(test_labels_one_hot, axis=1) + 1

    # 输出 classification_report
    print(classification_report(true_labels, y_pred))


if __name__ == '__main__':
    bert_tf()
