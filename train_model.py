# Import các thư viện và module cần thiết
from pyvi import ViTokenizer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle
from Stopwords import stop_words_vi


# Đọc dữ liệu intent từ file JSON
intents = json.loads(open('data/intents.json', encoding='utf-8').read())
# Khởi tạo các danh sách để lưu trữ từ, lớp, và tài liệu
words = []
classes = []
documents = []


# Xử lý các mẫu câu và xây dựng danh sách từ và tài liệu
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = ViTokenizer.tokenize(pattern)
        words.extend(w.split())
        documents.append((w.split(), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Tiền xử lý danh sách từ: chuyển thành chữ thường và loại bỏ từ dừng
words = [w.lower() for w in words if w not in stop_words_vi]
words = sorted(list(set(words)))

# Sắp xếp danh sách lớp và chuẩn bị các danh sách training và output
classes = sorted(list(set(classes)))

# Tạo danh sách mẫu câu và vector đầu ra tương ứng
training = []
output = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [w.lower() for w in word_patterns if w not in stop_words_vi]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Chuyển đổi danh sách training và output thành dạng numpy array
training_x = np.array([x[0] for x in training])
training_y = np.array([x[1] for x in training])

# Đặt lại đồ thị mặc định của TensorFlow
tf.compat.v1.reset_default_graph()

# Xây dựng mô hình mạng nơ-ron
net = tflearn.input_data(shape=[None, len(training_x[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(training_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# Khởi tạo đối tượng mô hình DNN của tflearn và huấn luyện mô hình
model = tflearn.DNN(net, tensorboard_dir='data/tflearn_logs')
model.fit(training_x, training_y, n_epoch=1000, batch_size=8, show_metric=True)

print("Training model done!")

# Lưu mô hình đã huấn luyện
model.save('data/model.tflearn')

# Lưu thông tin về dữ liệu training
training_data = {
    "words": words,
    "classes": classes,
    "training_x": training_x,
    "training_y": training_y
}

# Lưu dữ liệu training vào file pickle
with open("data/training_data.pkl", "wb") as data_file:
    pickle.dump(training_data, data_file)

# Đọc dữ liệu training từ file pickle
with open("data/training_data.pkl", "rb") as data_file:
    training_data = pickle.load(data_file)

words = training_data["words"]
classes = training_data["classes"]
training_x = training_data["training_x"]
training_y = training_data["training_y"]


# Kiểm tra độ chính xác của mô hình đã huấn luyện
# Dự đoán intent cho một câu
def predict_intent(sentence):
    sentence_bag = [0] * len(words)
    w = ViTokenizer.tokenize(sentence)
    w = [word.lower() for word in w.split() if word.lower() not in stop_words_vi]
    for word in w:
        for i, w in enumerate(words):
            if w == word:
                sentence_bag[i] = 1

    # Dự đoán intent cho câu với mô hình đã huấn luyện
    results = model.predict([np.array(sentence_bag)])
    # Lấy nhãn có xác suất dự đoán cao nhất
    results_index = np.argmax(results)
    predicted_intent = classes[results_index]
    return predicted_intent

# Tính toán accuracy trên dữ liệu huấn luyện
correct = 0
for i, sentence in enumerate(training_x):
    predicted = predict_intent(' '.join([words[j] for j in range(len(words)) if sentence[j] == 1]))
    true_label = ' '.join([classes[j] for j in range(len(classes)) if training_y[i][j] == 1])
    if predicted == true_label:
        correct += 1

accuracy = correct / len(training_x)
print(f"Accuracy on training data: {accuracy * 100:.2f}%")


