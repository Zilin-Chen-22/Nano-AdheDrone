import cv2
import numpy as np

# 选择tag family和id
TAG_FAMILY = "tag36h11"
TAG_ID = 0

# 输出尺寸（像素）
IMG_SIZE = 400

def generate_apriltag(tag_id=0, size=400):
    # OpenCV 自带 AprilTag 生成（需要较新版本）
    dict_apriltag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    
    img = np.zeros((size, size), dtype=np.uint8)
    img = cv2.aruco.generateImageMarker(dict_apriltag, tag_id, size)

    return img

if __name__ == "__main__":
    tag = generate_apriltag(TAG_ID, IMG_SIZE)

    filename = f"apriltag_{TAG_ID}.png"
    cv2.imwrite(filename, tag)

    print(f"Saved: {filename}")