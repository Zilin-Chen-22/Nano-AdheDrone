import cv2
import numpy as np

# ===== 参数（必须改）=====
TAG_SIZE = 0.08  # 实际边长（单位：米，比如 5cm = 0.05）

# 加载相机畸变矫正参数
camera_params = np.load("./camera_params.npz")
camera_matrix = camera_params['camera_matrix'].astype(np.float32)
dist_coeffs = camera_params['dist_coeffs'].astype(np.float32)

# =========================

# AprilTag dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(aruco_dict)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 应用镜头畸变矫正
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            c = corners[i]

            # 估计位姿
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                c, TAG_SIZE, camera_matrix, dist_coeffs
            )

            # 距离（Z轴）
            distance = tvec[0][0][2]

            # 画框
            cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)

            # 画坐标轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, TAG_SIZE)

            # 显示距离
            x, y = int(c[0][0][0]), int(c[0][0][1])
            cv2.putText(frame,
                        f"{distance:.3f} m",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

    cv2.imshow("AprilTag Distance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()