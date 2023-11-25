# Import các thư viện và module cần thiết
from pyvi import ViTokenizer, ViPosTagger
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import datetime
import webbrowser
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

# Xây dựng mô hình mạng nơ-ron
net = tflearn.input_data(shape=[None, len(training_x[0])])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, len(training_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# Đọc lại mô hình đã lưu trước đó
model = tflearn.DNN(net, tensorboard_dir='data/tflearn_logs')
model.load('data/model.tflearn')

# Đọc dữ liệu training từ file pickle
with open("data/training_data.pkl", "rb") as data_file:
    training_data = pickle.load(data_file)

words = training_data["words"]
classes = training_data["classes"]
training_x = training_data["training_x"]
training_y = training_data["training_y"]


# Các hàm tiện ích cho việc xử lý câu hỏi và tạo vector BoW
def clean_up_sentence(sentence):
     # Sử dụng ViTokenizer để tách từ trong câu và lưu vào biến sentence_words
    sentence_words = ViTokenizer.tokenize(sentence)
    # Chuyển đổi các từ thành chữ thường và loại bỏ các từ dừng
    sentence_words = [w.lower() for w in sentence_words.split() if w not in stop_words_vi]
    # Trả về danh sách các từ đã được tiền xử lý
    return sentence_words
    
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Thiết lập ngưỡng lỗi cho việc phân loại
ERROR_THRESHOLD = 0.25

# Hàm classify đã thay đổi để sử dụng mô hình đã tải
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

# Hàm xử lý câu hỏi và trả về câu trả lời tương ứng
def response(sentence):
    if 'time' in sentence:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Thời gian hiện tại là: {current_time}"
    elif any(word in sentence for word in ['google', 'youtube', 'wikipedia', 'facebook', 'twitter', 'instagram', 'github', 'stackoverflow', 'gmail']):
        website = {
            'google': 'https://www.google.com',
            'youtube': 'https://www.youtube.com',
            'wikipedia': 'https://www.wikipedia.org',
            'facebook': 'https://www.facebook.com',
            'twitter': 'https://www.twitter.com',
            'instagram': 'https://www.instagram.com',
            'github': 'https://www.github.com',
            'stackoverflow': 'https://www.stackoverflow.com',
            'gmail': 'https://mail.google.com'
        }
        webbrowser.open(website[next(word for word in website if word in sentence)])
        return f"Đã mở trình duyệt và điều hướng đến {next(word for word in website if word in sentence).capitalize()}."
    else:
        results = classify(sentence)
        if results:
            intent, confidence = results[0]
            print(f"Predicted Intent: {intent}, Confidence: {confidence}")
            for i in intents['intents']:
                if i['tag'] == intent:
                    responses = i['responses']
                    return random.choice(responses)
        else:
            return "Xin lỗi, tôi không hiểu câu hỏi của bạn."

# Tên của chatbot
bot_name = "Sam"

# Bắt đầu vòng lặp để chạy chatbot
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        # Gọi hàm response để nhận câu trả lời và in ra màn hình
        resp = response(sentence)
        print(resp)
