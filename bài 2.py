# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import os
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Tải dữ liệu từ thư mục
def load_images_from_folder(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label = label_folder
        for filename in os.listdir(os.path.join(folder, label_folder)):
            img = cv2.imread(os.path.join(folder, label_folder, filename))
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images_from_folder('C:/Users/wiburach/Capture/hoavameo')

# Tiền xử lý dữ liệu
images = images / 255.0  # Chuẩn hóa hình ảnh
le = LabelEncoder()
labels = le.fit_transform(labels)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Chuyển đổi định dạng hình ảnh
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Định nghĩa các mô hình
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Lưu trữ kết quả
results = {}

# Huấn luyện và đánh giá mô hình
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_flattened, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test_flattened)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    results[name] = {
        'Training Time': training_time,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# Hiển thị kết quả
results_df = pd.DataFrame(results).T
print(results_df)
