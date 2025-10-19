# Image Classification Web App 🧠📷

Ứng dụng web cho phép người dùng **tải ảnh lên** và **phân loại hình ảnh** bằng mô hình học sâu (Deep Learning).

## 🚀 Cấu trúc dự án

<img width="460" height="266" alt="image" src="https://github.com/user-attachments/assets/598d39bf-2d19-4659-9055-f79fdbeef083" />

## ⚙️ Yêu cầu hệ thống

- Python 3.10+
- pip (trình quản lý gói Python)

## 🚀 Demo Test

### 🔹 Online:
Truy cập trực tiếp:  
👉 **[https://cat-dog-classification-beryl.vercel.app/](https://cat-dog-classification-beryl.vercel.app/)**  
để tiến hành **test mô hình trực tuyến**.

> ⚠️ Nếu hệ thống online không hoạt động, bạn có thể chạy **offline** theo hướng dẫn dưới đây.

---

## 🧩 Cài đặt và chạy offline

### 1️⃣ Clone repository
Mở terminal và chạy:
```bash
git clone https://github.com/Lecongquochuy/CatDogClassification.git
cd CatDogClassification
```

### 2️⃣ Tạo môi trường ảo
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# hoặc
source venv/bin/activate     # macOS/Linux
```


3️⃣ Cài đặt các thư viện cần thiết
```bash
pip install -r BE/requirements.txt
```

4️⃣ Chạy server Flask
```bash
cd BE
python app.py
```

5️⃣ Mở giao diện web

Tại thư mục gốc dự án, chạy:

```bash
cd FE
python -m http.server 3000
```

sau đó truy cập **[http://127.0.0.1:3000](http://127.0.0.1:3000)**


## 🧪 Đánh giá độ chính xác của mô hình (Test model)

Nếu bạn muốn **đánh giá độ chính xác của model bằng dữ liệu cá nhân**, có thể làm theo các bước sau:

### 1️⃣ Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

2️⃣ Chạy đánh giá mô hình
```bash
python main.py
```

💡 Lưu ý:

Mở file main.py và chọn mô hình bạn muốn kiểm tra (ví dụ: model.pth).

Cập nhật đường dẫn đến tập dữ liệu cá nhân mà bạn muốn đánh giá.

Kết quả đánh giá (accuracy) sẽ được in trực tiếp ra terminal.

## 📦 Tải trọng số mô hình và quá trình huấn luyện

Bạn có thể truy cập đường dẫn sau để tải **trọng số mô hình (model weights)** và **quá trình huấn luyện (training logs)**:

👉 [Google Drive - CatDogClassification Weights & Training](https://drive.google.com/drive/folders/1TzAB7TjuIqCj7YetzLqbSfcoWrlhBCJZ?usp=drive_link)

> 💡 **Lưu ý:**  
> - Sau khi tải trọng số về, đặt file `.pth` vào đúng thư mục mà `model.py` hoặc `main.py` sử dụng.  
> - Đảm bảo tên file và đường dẫn trong code trùng khớp để tránh lỗi `FileNotFoundError`.  
> - Bạn có thể xem lại lịch sử huấn luyện (loss, accuracy, epoch, v.v.) trong thư mục log để tham khảo quá trình training.

