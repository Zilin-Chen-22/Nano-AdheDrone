import pygame
import serial
import time
import cv2

SERIAL_PORT = "COM9"
BAUDRATE = 115200

CRSF_SYNC = 0xC8
CRSF_FRAMETYPE_RC_CHANNELS = 0x16

CHANNEL_MIN = 172
CHANNEL_MID = 992
CHANNEL_MAX = 1811

CRSF_LOW = 172
CRSF_MID = 992

SEND_HZ = 50
THROTTLE_ARM_THRESHOLD = 200


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def map_norm_to_channel(v):
    # 1️⃣ 先转成标准 RC（1000~2000）
    us = 1500 + v * 500   # v ∈ [-1,1]

    # 2️⃣ 再映射到 CRSF（172~1811）
    crsf = (us - 1000) * (CHANNEL_MAX - CHANNEL_MIN) / 1000 + CHANNEL_MIN

    # 3️⃣ 用 round 而不是 int（关键！）
    return int(round(crsf))


def crsf_crc(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc << 1) ^ 0xD5 if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


def pack_channels(ch):
    buf = bytearray(22)
    bitpos = 0
    for v in ch:
        v = clamp(v, 0, 0x7FF)
        for i in range(11):
            if v & (1 << i):
                buf[(bitpos + i) // 8] |= 1 << ((bitpos + i) % 8)
        bitpos += 11
    return buf


def build_crsf_frame(channels):
    payload = pack_channels(channels)
    frame = bytearray()
    frame.append(CRSF_SYNC)
    frame.append(len(payload) + 2)
    frame.append(CRSF_FRAMETYPE_RC_CHANNELS)
    frame.extend(payload)
    frame.append(crsf_crc(frame[2:]))
    return frame


# ========= GUI =========
pygame.init()

SCREEN_W = 1000
SCREEN_H = 520
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("FPV Slider RC + Camera")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# ========= 布局 =========
CONTROL_X_START = 100
CONTROL_SPACING = 80

BUTTON_X = 520

CAM_X = 620
CAM_Y = 40
CAM_W = 320
CAM_H = 240

# ========= 摄像头 =========
cam_index = 0
cap = cv2.VideoCapture(cam_index)


def switch_camera(idx):
    global cap, cam_index
    if cap:
        cap.release()
    cam_index = idx
    cap = cv2.VideoCapture(cam_index)


# ========= 滑杆 =========
SLIDER_HEIGHT = 400
SLIDER_WIDTH = 30
SLIDER_TOP = 60

SLIDER_X = [
    CONTROL_X_START + i * CONTROL_SPACING
    for i in range(4)
]

SLIDER_LABELS = ["ROLL", "PITCH", "THRO", "YAW"]

slider_vals = [0.0, 0.0, -1.0, 0.0]
dragging = [False] * 4

# ========= 按钮 =========
arm_request = False
angle_on = False

arm_rect = pygame.Rect(BUTTON_X, 60, 80, 40)
angle_rect = pygame.Rect(BUTTON_X, 110, 80, 40)

# ========= 串口 =========
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0)

armed = False
last_send = 0
running = True

while running:
    screen.fill((30, 30, 30))

    # ===== 事件 =====
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                switch_camera(0)
            elif event.key == pygame.K_2:
                switch_camera(1)
            elif event.key == pygame.K_3:
                switch_camera(2)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            if arm_rect.collidepoint(mx, my):
                if armed:
                    arm_request = False
                else:
                    if throttle_low:
                        arm_request = True

            if angle_rect.collidepoint(mx, my):
                angle_on = not angle_on

            for i in range(4):
                rect = pygame.Rect(
                    SLIDER_X[i] - SLIDER_WIDTH // 2,
                    SLIDER_TOP,
                    SLIDER_WIDTH,
                    SLIDER_HEIGHT
                )
                if rect.collidepoint(mx, my):
                    dragging[i] = True

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = [False] * 4

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            for i in range(4):
                if dragging[i]:
                    y = clamp(my, SLIDER_TOP, SLIDER_TOP + SLIDER_HEIGHT)
                    val = 1 - (y - SLIDER_TOP) / SLIDER_HEIGHT * 2
                    # 增加细分（关键优化）
                    val = round(val, 4)
                    slider_vals[i] = clamp(val, -1, 1)

    # ===== 自动回中 =====
    for i in [0, 1, 3]:
        if not dragging[i]:
            slider_vals[i] *= 0.85
            if abs(slider_vals[i]) < 0.01:
                slider_vals[i] = 0.0

    # ===== 通道 =====
    ch = [CHANNEL_MID] * 16
    ch[0] = map_norm_to_channel(slider_vals[0])
    ch[1] = map_norm_to_channel(slider_vals[1])
    ch[2] = map_norm_to_channel(slider_vals[2])
    ch[3] = map_norm_to_channel(slider_vals[3])

    throttle = ch[2]
    throttle_low = throttle <= THROTTLE_ARM_THRESHOLD

    if arm_request and throttle_low:
        armed = True
    if not arm_request:
        armed = False

    ch[5] = CRSF_MID if armed else CRSF_LOW
    ch[6] = CRSF_MID if angle_on else CRSF_LOW

    # ===== 发送 =====
    now = time.time()
    if now - last_send > 1.0 / SEND_HZ:
        ser.write(build_crsf_frame(ch))
        last_send = now

    # ===== 摄像头 =====
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0, 1)
        surface = pygame.surfarray.make_surface(frame)
        screen.blit(surface, (CAM_X, CAM_Y))

    # ===== 滑杆 =====
    for i in range(4):
        x = SLIDER_X[i]
        y_top = SLIDER_TOP
        y_bottom = SLIDER_TOP + SLIDER_HEIGHT

        pygame.draw.line(screen, (100, 100, 100), (x, y_top), (x, y_bottom), 4)

        y = int(y_top + (1 - slider_vals[i]) / 2 * SLIDER_HEIGHT)
        pygame.draw.circle(screen, (0, 200, 255), (x, y), 8)

        label = font.render(SLIDER_LABELS[i], True, (255, 255, 255))
        screen.blit(label, (x - 35, y_bottom + 10))

    # ===== 按钮 =====
    if armed:
        arm_color = (0, 200, 0)
    elif arm_request:
        arm_color = (200, 150, 0)
    else:
        arm_color = (120, 0, 0)

    pygame.draw.rect(screen, arm_color, arm_rect)
    pygame.draw.rect(screen, (0, 200, 0) if angle_on else (120, 0, 0), angle_rect)

    screen.blit(font.render("ARM", True, (255, 255, 255)), (arm_rect.x + 20, arm_rect.y + 10))
    screen.blit(font.render("ANGLE", True, (255, 255, 255)), (angle_rect.x + 10, angle_rect.y + 10))

    # ===== 状态 =====
    if armed:
        msg = "ARMED"
        color = (50, 255, 50)
    else:
        if throttle_low:
            msg = "SAFE TO ARM"
            color = (200, 200, 50)
        else:
            msg = "THROTTLE HIGH - CANNOT ARM"
            color = (255, 50, 50)

    screen.blit(font.render(msg, True, color), (200, 20))

    screen.blit(font.render(f"CAM: {cam_index + 1} (1/2/3)", True, (200, 200, 200)), (CAM_X, 5))

    value_x = 500
    value_y = 440
    line_gap = 22
    col_gap = 200

    # 第一行：ROLL 和 PITCH
    roll_text = font.render(f"ROLL: {ch[0]}", True, (255, 255, 255))
    screen.blit(roll_text, (value_x, value_y))

    pitch_text = font.render(f"PITCH: {ch[1]}", True, (255, 255, 255))
    screen.blit(pitch_text, (value_x + col_gap, value_y))

    # 第二行：YAW 和 THROTTLE
    yaw_text = font.render(f"YAW: {ch[3]}", True, (255, 255, 255))
    screen.blit(yaw_text, (value_x, value_y + line_gap))

    throttle_text = font.render(f"THROTTLE: {ch[2]}", True, (255, 255, 255))
    screen.blit(throttle_text, (value_x + col_gap, value_y + line_gap))

    # # 第三行：YAW Angle
    # yaw_angle_text = font.render(f"YAW Angle: {slider_vals[3] * 180:.1f}°", True, (255, 255, 255))
    # screen.blit(yaw_angle_text, (value_x, value_y + 2 * line_gap))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
cap.release()
ser.close()