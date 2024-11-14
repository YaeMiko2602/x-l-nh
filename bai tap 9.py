# Import các thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Bước 1: Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Chuẩn hóa ảnh
    rotation_range=20,  # Xoay ảnh ngẫu nhiên 20 độ
    width_shift_range=0.2,  # Dịch chuyển ngang ngẫu nhiên 20%
    height_shift_range=0.2,  # Dịch chuyển dọc ngẫu nhiên 20%
    shear_range=0.2,  # Biến dạng ảnh
    zoom_range=0.2,  # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,  # Lật ảnh ngẫu nhiên theo chiều ngang
    fill_mode='nearest'  # Phương thức điền màu cho pixel trống khi dịch chuyển ảnh
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\anhtr\\Downloads\\bai tap 9\\dataset\\Train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'C:\\Users\\anhtr\\Downloads\\bai tap 9\\dataset\\Validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Bước 2: Cải thiện mô hình CNN với BatchNormalization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile mô hình với Adam Optimizer có learning rate nhỏ
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Bước 3: Huấn luyện mô hình với Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]
)


# Bước 4: Sử dụng mô hình để dự đoán ảnh mới
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    plt.imshow(image.load_img(img_path))
    plt.axis('off')
    if prediction[0] > 0.5:
        plt.title("Dự đoán: Chó")
    else:
        plt.title("Dự đoán: Mèo")
    plt.show()


# Ví dụ sử dụng dự đoán ảnh
# Thay 'path_to_your_image.jpg' bằng đường dẫn thực tế của ảnh cần dự đoán
predict_image(model, 'anhmeotes.jpg')

# Bước 5: Đánh giá kết quả
# Vẽ biểu đồ độ chính xác và độ mất mát
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Tính Confusion Matrix
validation_steps = len(validation_generator)
Y_pred = model.predict(validation_generator, steps=validation_steps)
y_pred = [1 if x > 0.5 else 0 for x in Y_pred]
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
classes = ['Mèo', 'Chó']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Thêm số liệu trong ô của Confusion Matrix
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
