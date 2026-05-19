# Nano AdheDrone - 本科毕业设计

VSERF驱动栖息无人机控制研究

## 文件结构

### aptiltag
生成对应tag图片

### calibration_images
镜头畸变校准，有图形化界面简单操作，需要标准棋盘格

### 其余文件
- camera_params.npz：摄像头畸变参数，无牙仔镜头
- fpv_mouse_crsf.py：已弃用
- remote_controller.py：核心主程序，包含控制器、crsf无限发送、图传接收等功能，有图形化界面
- test.py：验证语法用，无实际意义