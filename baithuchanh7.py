import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

# Đọc ảnh (thay đường dẫn chính xác tới ảnh của bạn)
image = cv2.imread('D:/image/image.png', cv2.IMREAD_GRAYSCALE)

# Kiểm tra xem ảnh có được đọc thành công không
if image is None:
    print("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")
else:
    # Áp dụng Gaussian Blur để làm mờ ảnh, giảm nhiễu
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Phân đoạn với các toán tử biên
    # 1. Sobel
    sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.absolute(sobel))  # Chuyển về dạng uint8 để hiển thị

    # 2. Prewitt
    prewitt_x = filters.prewitt_h(gaussian_blur)
    prewitt_y = filters.prewitt_v(gaussian_blur)
    prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt = (prewitt * 255).astype(np.uint8)  # Chuyển về dạng uint8 để hiển thị

    # 3. Robert
    roberts = filters.roberts(gaussian_blur)
    roberts = (roberts * 255).astype(np.uint8)  # Chuyển về dạng uint8 để hiển thị

    # 4. Canny
    canny = cv2.Canny(gaussian_blur, 100, 200)

    # Hiển thị kết quả
    titles = ['Original Image', 'Gaussian Blur', 'Sobel', 'Prewitt', 'Roberts', 'Canny']
    images = [image, gaussian_blur, sobel, prewitt, roberts, canny]

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
