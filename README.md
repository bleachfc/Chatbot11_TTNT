# Xây dựng hệ thống chatbot tư vấn tập gym.

Phiên bản Python đề nghị: 3.11.5
Các thư viện cần cài đặt: pyvi, tensorflow, tflearn, numpy.
kích hoạt venv bằng lệnh: python -m venv myvenv
Cài bằng lệnh: pip install -r requirements.txt
File Stopwords là file chứa các từ dừng, có thể thêm các từ mới vào file.
Kích hoạt venv và tải các thư viện cần thiết

# Chatbot sử dụng phần UI của tác giả: patrickloeber
Link trên Github: https://github.com/patrickloeber/chatbot-deployment

# Để chạy Chatbot cùng giao diện: 
Chạy file: app.py

# Để training thêm dữ liệu cho chatbot:
Thêm dữ liệu vào file: intents.json
Chạy file: train_model.json để training sau khi thêm dữ liệu

# Để chạy file chatbot riêng không kèm giao diện:
Copy file chatbot.py trong thư mục module ra ngoài thư mục chính.
Sau đó chạy lệnh:  python chatbot.py

# Có thể chạy riêng file ChatbotGym.ipynb trên Google Colab

# Một số lưu ý:

1.  Train Model:
> Trước khi chạy chatbot cần training model, trường hợp đã có dữ liệu training trước đó thì có thể bỏ qua phần huấn luyện mô hình.

2.  Import và cài đặt các thư viện cần thiết:
>   pyvi, tflearn và tensorflow.
>   Có lỗi liên quan đến import thư viện tflearn sẽ gặp phải trong phần này.
>   Thay thế đoạn is_sequence thành is_sequence_or_composite

![](images_dir/image_path.jpg)

3. File Stopwords:
>  File Stopwords cần upload để có thể Import.

4. Đường dẫn file:
> Cần thay thế đường dẫn đến các file và thư mục như: **intents.json, tflearn_logs, model.tflearn, training_data.pkl** trước khi chạy.