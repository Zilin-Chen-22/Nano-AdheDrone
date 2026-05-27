import pygame
import serial
import time
import cv2
import numpy as np

# ========= 串口配置 =========
SERIAL_PORT = "COM9"
BAUDRATE = 115200

# ========= CRSF 协议常量 =========
CRSF_SYNC = 0xC8
CRSF_FRAMETYPE_RC_CHANNELS = 0x16

CHANNEL_MIN = 172
CHANNEL_MID = 992
CHANNEL_MAX = 1811

CRSF_LOW = 172
CRSF_MID = 992

SEND_HZ = 50
THROTTLE_ARM_THRESHOLD = 200

# ========= 摇杆手感 =========
STICK_RETURN_RATE = 0.85
STICK_DEADZONE = 0.01

# ========= 视觉伺服参数（调参区）=========
TARGET_DIST = 0.3
TAG_HOLD_DURATION = 0.05

# Tag 丢失后的安全油门（归一化值，对应 RC 1400）
TAG_LOST_THROTTLE = -0.2

# 自动降落触发条件
LAND_DIST_MIN   = 0.3
LAND_DIST_MAX   = 0.45
LAND_ERR_MAX    = 0.15
LAND_THRUST_DUR = 0.5   # 冲顶时间
LAND_STICK_ON = 0.2     # 延迟打开电流变液时间

# 各轴 PID 参数（运行时可在界面调整）
pid_params = {
    "THRO":  [0.45,  0.120,  0.11],
    "PITCH": [0.30,  0.000,  0.20],
    "ROLL":  [0.30,  0.000,  0.20],
}

# 各轴 PID 输出限幅（归一化 [-1, 1]）
THROTTLE_OUT_LIMIT = 1.0
PITCH_OUT_LIMIT    = 0.5
ROLL_OUT_LIMIT     = 0.5

# Apriltag 尺寸
TAG_SIZE = 0.04


# ========= PID 控制器 =========
class PID:
    def __init__(self, kp, ki, kd, out_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update(self, error, now):
        if self.last_time is None:
            dt = 0.0
        else:
            dt = now - self.last_time
        self.last_time = now

        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.out_limit, min(self.out_limit, output))

    def set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


# ========= 工具函数 =========
def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def map_norm_to_channel(v):
    us = 1500 + v * 500
    crsf = (us - 1000) * (CHANNEL_MAX - CHANNEL_MIN) / 1000 + CHANNEL_MIN
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


# ========= 摄像头 =========
def switch_camera(idx):
    global cap, cam_index
    if cap:
        cap.release()
    cam_index = idx
    cap = cv2.VideoCapture(cam_index)
    for _ in range(3):
        cap.read()


cam_index = 0
cap = cv2.VideoCapture(cam_index)
for _ in range(3):
    cap.read()

camera_params = np.load("camera_params.npz")
camera_matrix = camera_params['camera_matrix']
dist_coeffs = camera_params['dist_coeffs']

# ===== AprilTag =====
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ========= GUI =========
pygame.init()

SCREEN_W = 1000
SCREEN_H = 520
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("FPV Slider RC + Camera + PID")

clock = pygame.time.Clock()
font       = pygame.font.SysFont(None, 24)
font_small = pygame.font.SysFont(None, 20)
font_mono  = pygame.font.SysFont("Courier New", 19)

# ========= 布局 =========
CONTROL_X_START = 100
CONTROL_SPACING = 80
BUTTON_X = 520

CAM_X = 620
CAM_Y = 40
CAM_W = 320
CAM_H = 240

# ========= 滑杆 =========
SLIDER_HEIGHT = 400
SLIDER_WIDTH = 30
SLIDER_TOP = 60

SLIDER_X = [CONTROL_X_START + i * CONTROL_SPACING for i in range(4)]
SLIDER_LABELS = ["ROLL", "PITCH", "THRO", "YAW"]

slider_vals = [0.0, 0.0, -1.0, 0.0]
dragging = [False] * 4

# ========= 按钮 =========
arm_request = False
angle_on    = False
auto_on     = False
stick_on    = False   # AUX4 控制

arm_rect   = pygame.Rect(BUTTON_X, 60,  80, 40)
angle_rect = pygame.Rect(BUTTON_X, 110, 80, 40)
auto_rect  = pygame.Rect(BUTTON_X, 160, 80, 40)
stick_rect = pygame.Rect(BUTTON_X, 210, 80, 40)   # 新增 STICK 按钮

# ========= 串口 =========
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0)

# ========= 状态 =========
armed = False
last_send = 0
throttle_low = True
running = True

# ========= PID 控制器实例 =========
pid_throttle = PID(*pid_params["THRO"],  THROTTLE_OUT_LIMIT)
pid_pitch    = PID(*pid_params["PITCH"], PITCH_OUT_LIMIT)
pid_roll     = PID(*pid_params["ROLL"],  ROLL_OUT_LIMIT)

# ========= 视觉伺服状态 =========
last_tag_time = None
last_pid_out = [0.0, 0.0, -1.0]   # [roll, pitch, throttle]
err_roll_disp     = 0.0
err_pitch_disp    = 0.0
err_throttle_disp = 0.0
current_dist      = 0.0

# ========= 自动降落序列状态 =========
land_state        = None
land_thrust_start = None


def abort_land():
    global land_state, land_thrust_start
    land_state = None
    land_thrust_start = None


def do_disarm():
    global armed, arm_request, auto_on, land_state, land_thrust_start
    arm_request = False
    armed = False
    auto_on = False
    pid_throttle.reset()
    pid_pitch.reset()
    pid_roll.reset()
    last_pid_out[0] = 0.0
    last_pid_out[1] = 0.0
    last_pid_out[2] = -1.0
    abort_land()


# ========= PID 编辑器状态 =========
selected_cell = None
edit_buffer   = ""

PID_EDITOR_X  = 510
PID_EDITOR_Y  = 290
PID_ROW_H     = 26
PID_COL_W     = 68
PID_ROWS      = ["THRO", "PITCH", "ROLL"]
PID_COLS      = ["Kp", "Ki", "Kd"]


def get_pid_cell_rect(row, col):
    x = PID_EDITOR_X + 52 + col * PID_COL_W
    y = PID_EDITOR_Y + 22 + row * PID_ROW_H
    return pygame.Rect(x, y, PID_COL_W - 4, PID_ROW_H - 2)


def apply_pid_params():
    pid_throttle.set_gains(*pid_params["THRO"])
    pid_pitch.set_gains(*pid_params["PITCH"])
    pid_roll.set_gains(*pid_params["ROLL"])


# ========= 主循环 =========
while running:
    now = time.time()
    screen.fill((30, 30, 30))

    # ===== 事件 =====
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if selected_cell is not None and not armed:
                if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    try:
                        val = float(edit_buffer)
                        row, col = selected_cell
                        pid_params[PID_ROWS[row]][col] = val
                        apply_pid_params()
                    except ValueError:
                        pass
                    selected_cell = None
                    edit_buffer = ""
                elif event.key == pygame.K_ESCAPE:
                    selected_cell = None
                    edit_buffer = ""
                elif event.key == pygame.K_BACKSPACE:
                    edit_buffer = edit_buffer[:-1]
                else:
                    if event.unicode in "0123456789.-":
                        edit_buffer += event.unicode
            else:
                if event.key == pygame.K_1:
                    switch_camera(0)
                elif event.key == pygame.K_2:
                    switch_camera(1)
                elif event.key == pygame.K_3:
                    switch_camera(2)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            # PID 编辑器点击（仅 disarm）
            if not armed:
                clicked_cell = False
                for r in range(len(PID_ROWS)):
                    for c in range(len(PID_COLS)):
                        rect = get_pid_cell_rect(r, c)
                        if rect.collidepoint(mx, my):
                            selected_cell = (r, c)
                            edit_buffer = str(pid_params[PID_ROWS[r]][c])
                            clicked_cell = True
                            break
                    if clicked_cell:
                        break
                if not clicked_cell:
                    selected_cell = None
                    edit_buffer = ""

            # ARM
            if arm_rect.collidepoint(mx, my):
                if armed:
                    do_disarm()
                else:
                    if throttle_low:
                        arm_request = True

            # AUTO
            if auto_rect.collidepoint(mx, my):
                if armed:
                    if auto_on:
                        auto_on = False
                        abort_land()
                        pid_throttle.reset()
                        pid_pitch.reset()
                        pid_roll.reset()
                        last_pid_out[0] = 0.0
                        last_pid_out[1] = 0.0
                        last_pid_out[2] = -1.0
                    else:
                        auto_on = True

            # ANGLE
            if angle_rect.collidepoint(mx, my):
                angle_on = not angle_on

            # STICK（随时可切换）
            if stick_rect.collidepoint(mx, my):
                stick_on = not stick_on

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
                    val = round(val, 4)
                    slider_vals[i] = clamp(val, -1, 1)

    # ===== 自动回中 =====
    for i in [0, 1, 3]:
        if not dragging[i]:
            slider_vals[i] *= STICK_RETURN_RATE
            if abs(slider_vals[i]) < STICK_DEADZONE:
                slider_vals[i] = 0.0

    # ===== 通道基础值 =====
    ch = [CHANNEL_MID] * 16

    # ===== 摄像头 & AprilTag =====
    ret, frame = cap.read()
    tag_detected = False

    if ret:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        h, w = frame.shape[:2]
        angle = 45
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        cx_r = new_w / 2.0
        cy_r = new_h / 2.0

        if ids is not None and len(ids) > 0:
            c = corners[0]
            img_pts = c.reshape(4, 2).astype(np.float32)

            tag_cx = img_pts[:, 0].mean()
            tag_cy = img_pts[:, 1].mean()

            half = TAG_SIZE / 2
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0]
            ], dtype=np.float32)

            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, None)

            if ok:
                tag_detected = True
                last_tag_time = now

                dist = tvec[2][0]
                current_dist = dist

                tag_pt = np.array([[[tag_cx, tag_cy]]], dtype=np.float32)
                tag_pt_rotated = cv2.transform(tag_pt, M)
                tag_cx_r = tag_pt_rotated[0][0][0]
                tag_cy_r = tag_pt_rotated[0][0][1]
                tag_cy_r = new_h - tag_cy_r

                err_throttle = dist - TARGET_DIST
                err_pitch    = -((tag_cy_r - cy_r) / cy_r)
                err_roll     =  (tag_cx_r - cx_r) / cx_r

                err_throttle_disp = err_throttle
                err_pitch_disp    = err_pitch
                err_roll_disp     = err_roll

                if auto_on and land_state is None:
                    out_throttle = pid_throttle.update(err_throttle, now)
                    out_pitch    = pid_pitch.update(err_pitch, now)
                    out_roll     = pid_roll.update(err_roll, now)
                    last_pid_out[0] = out_roll
                    last_pid_out[1] = out_pitch
                    last_pid_out[2] = out_throttle

                    if (LAND_DIST_MIN <= dist <= LAND_DIST_MAX
                            and abs(err_roll)  <= LAND_ERR_MAX
                            and abs(err_pitch) <= LAND_ERR_MAX):
                        land_state = "thrusting"
                        land_thrust_start = now

                cv2.polylines(frame, [c.astype(int)], True, (0, 255, 0), 2)
                cv2.drawFrameAxes(frame, camera_matrix, None, rvec, tvec, TAG_SIZE)
                cv2.drawMarker(frame, (int(tag_cx), int(tag_cy)),
                               (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

        if not tag_detected and auto_on and land_state is None:
            if last_tag_time is not None:
                if now - last_tag_time > TAG_HOLD_DURATION:
                    last_pid_out[0] = 0.0
                    last_pid_out[1] = 0.0
                    last_pid_out[2] = TAG_LOST_THROTTLE
            else:
                last_pid_out[0] = 0.0
                last_pid_out[1] = 0.0
                last_pid_out[2] = TAG_LOST_THROTTLE

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rotated = cv2.warpAffine(frame, M, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        rotated = cv2.flip(rotated, 0)

        scale = min(CAM_W / new_w, CAM_H / new_h)
        resized_w = int(new_w * scale)
        resized_h = int(new_h * scale)
        resized = cv2.resize(rotated, (resized_w, resized_h))

        canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        x_offset = (CAM_W - resized_w) // 2
        y_offset = (CAM_H - resized_h) // 2
        canvas[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized
        canvas = canvas.swapaxes(0, 1)
        surface = pygame.surfarray.make_surface(canvas)
        screen.blit(surface, (CAM_X, CAM_Y))

    else:
        no_signal_bg = pygame.Surface((CAM_W, CAM_H))
        no_signal_bg.fill((20, 20, 20))
        screen.blit(no_signal_bg, (CAM_X, CAM_Y))
        no_signal_text = font.render("NO CAMERA SIGNAL", True, (255, 60, 60))
        screen.blit(no_signal_text, (CAM_X + 60, CAM_Y + 110))

    # ===== 自动降落序列执行 =====
    if land_state == "thrusting":
        elapsed = now - land_thrust_start
        if elapsed >= LAND_STICK_ON:
            stick_on = True          # 0.2s 后打开 stick
        if elapsed < LAND_THRUST_DUR:
            last_pid_out[0] = 0.0
            last_pid_out[1] = 0.0
            last_pid_out[2] = 1.0
        else:
            do_disarm()              # 0.5s 后 disarm，stick_on 不动

    # ===== 通道赋值 =====
    if auto_on:
        ch[0] = map_norm_to_channel(clamp(last_pid_out[0], -1, 1))
        ch[1] = map_norm_to_channel(clamp(last_pid_out[1], -1, 1))
        ch[2] = map_norm_to_channel(clamp(last_pid_out[2], -1, 1))
        ch[3] = map_norm_to_channel(slider_vals[3])
    else:
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

    ch[5] = CRSF_MID if armed   else CRSF_LOW   # AUX1 - ARM
    ch[6] = CRSF_MID if angle_on else CRSF_LOW   # AUX2 - ANGLE
    ch[7] = CRSF_MID if stick_on else CRSF_LOW   # AUX4 - STICK（1500 开，172 关）

    # ===== 发送 =====
    if now - last_send > 1.0 / SEND_HZ:
        ser.write(build_crsf_frame(ch))
        last_send = now

    # ===== 滑杆绘制 =====
    for i in range(4):
        x = SLIDER_X[i]
        y_top = SLIDER_TOP
        y_bottom = SLIDER_TOP + SLIDER_HEIGHT

        pygame.draw.line(screen, (100, 100, 100), (x, y_top), (x, y_bottom), 4)

        if auto_on and i in [0, 1, 2]:
            color = (60, 100, 120)
        else:
            color = (0, 200, 255)

        y = int(y_top + (1 - slider_vals[i]) / 2 * SLIDER_HEIGHT)
        pygame.draw.circle(screen, color, (x, y), 8)

        label = font.render(SLIDER_LABELS[i], True, (255, 255, 255))
        screen.blit(label, (x - 35, y_bottom + 10))

    # ===== 按钮绘制 =====
    # ARM
    if armed:
        arm_color = (0, 200, 0)
    elif arm_request:
        arm_color = (200, 150, 0)
    else:
        arm_color = (120, 0, 0)
    pygame.draw.rect(screen, arm_color, arm_rect)
    screen.blit(font.render("ARM", True, (255, 255, 255)), (arm_rect.x + 20, arm_rect.y + 10))

    # ANGLE
    pygame.draw.rect(screen, (0, 200, 0) if angle_on else (120, 0, 0), angle_rect)
    screen.blit(font.render("ANGLE", True, (255, 255, 255)), (angle_rect.x + 10, angle_rect.y + 10))

    # AUTO
    if not armed:
        auto_color = (60, 60, 60)
    elif auto_on:
        auto_color = (0, 200, 100)
    else:
        auto_color = (120, 0, 0)
    pygame.draw.rect(screen, auto_color, auto_rect)
    screen.blit(font.render("AUTO", True, (255, 255, 255)), (auto_rect.x + 18, auto_rect.y + 10))

    # STICK（随时可切换，颜色同 ANGLE）
    pygame.draw.rect(screen, (0, 200, 0) if stick_on else (120, 0, 0), stick_rect)
    screen.blit(font.render("STICK", True, (255, 255, 255)), (stick_rect.x + 10, stick_rect.y + 10))

    # ===== 状态文字 =====
    if land_state == "thrusting":
        remaining = LAND_THRUST_DUR - (now - land_thrust_start)
        msg   = f"LANDING... {remaining:.1f}s"
        color = (255, 200, 0)
    elif armed:
        msg   = "ARMED"
        color = (50, 255, 50)
    else:
        msg   = "SAFE TO ARM" if throttle_low else "THROTTLE HIGH - CANNOT ARM"
        color = (200, 200, 50) if throttle_low else (255, 50, 50)
    screen.blit(font.render(msg, True, color), (200, 20))

    screen.blit(font.render(f"CAM: {cam_index + 1} (1/2/3)", True, (200, 200, 200)), (CAM_X, 5))

    # ===== 通道数值 =====
    value_x = 500
    value_y = 460
    line_gap = 22
    col_gap  = 200

    screen.blit(font.render(f"ROLL: {ch[0]}",     True, (255, 255, 255)), (value_x, value_y))
    screen.blit(font.render(f"PITCH: {ch[1]}",    True, (255, 255, 255)), (value_x + col_gap, value_y))
    screen.blit(font.render(f"YAW: {ch[3]}",      True, (255, 255, 255)), (value_x, value_y + line_gap))
    screen.blit(font.render(f"THROTTLE: {ch[2]}", True, (255, 255, 255)), (value_x + col_gap, value_y + line_gap))

    # ===== 距离显示（始终显示）=====
    if last_tag_time is not None:
        dist_color = (0, 255, 200) if tag_detected else (150, 150, 150)
        screen.blit(font.render(f"DIST: {current_dist:.2f} m", True, dist_color), (510, 440))

    # ===== 右侧信息区 =====
    pid_x = PID_EDITOR_X
    pid_y = PID_EDITOR_Y

    if not armed:
        screen.blit(font.render("PID PARAMS  (click to edit)", True, (200, 200, 200)), (pid_x, pid_y))

        for ci, col_name in enumerate(PID_COLS):
            hx = pid_x + 52 + ci * PID_COL_W + PID_COL_W // 2 - 10
            screen.blit(font_small.render(col_name, True, (180, 180, 180)), (hx, pid_y + 6))

        for ri, row_name in enumerate(PID_ROWS):
            ry = pid_y + 22 + ri * PID_ROW_H
            screen.blit(font_mono.render(row_name, True, (200, 220, 255)), (pid_x, ry + 3))

            for ci in range(len(PID_COLS)):
                rect = get_pid_cell_rect(ri, ci)
                is_selected = (selected_cell == (ri, ci))

                bg_color = (60, 80, 120) if is_selected else (45, 45, 55)
                pygame.draw.rect(screen, bg_color, rect, border_radius=3)
                pygame.draw.rect(screen, (120, 160, 255) if is_selected else (80, 80, 100),
                                 rect, 1, border_radius=3)

                if is_selected:
                    display_str = edit_buffer + "|"
                    txt_color = (255, 255, 100)
                else:
                    display_str = f"{pid_params[PID_ROWS[ri]][ci]:.3f}"
                    txt_color = (220, 220, 220)

                txt_surf = font_mono.render(display_str, True, txt_color)
                screen.blit(txt_surf, (rect.x + 3, rect.y + 3))

        screen.blit(font_small.render("Enter=confirm  Esc=cancel", True, (120, 120, 120)),
                    (pid_x, pid_y + 22 + len(PID_ROWS) * PID_ROW_H + 4))

    elif auto_on:
        if land_state == "thrusting":
            tag_status       = "LANDING SEQUENCE"
            tag_status_color = (255, 200, 0)
        elif tag_detected:
            tag_status       = "TAG: LOCKED"
            tag_status_color = (50, 255, 50)
        elif last_tag_time is not None and (now - last_tag_time) <= TAG_HOLD_DURATION:
            tag_status_color = (255, 200, 0)
            tag_status       = f"TAG: HOLD ({TAG_HOLD_DURATION - (now - last_tag_time):.1f}s)"
        else:
            tag_status       = "TAG: LOST"
            tag_status_color = (255, 60, 60)

        screen.blit(font.render(tag_status, True, tag_status_color),                        (pid_x, pid_y))
        screen.blit(font_small.render(f"ERR THRO:  {err_throttle_disp:+.3f} m", True, (180, 220, 255)), (pid_x, pid_y + 24))
        screen.blit(font_small.render(f"ERR PITCH: {err_pitch_disp:+.3f}",      True, (180, 220, 255)), (pid_x, pid_y + 42))
        screen.blit(font_small.render(f"ERR ROLL:  {err_roll_disp:+.3f}",       True, (180, 220, 255)), (pid_x, pid_y + 60))
        screen.blit(font_small.render(f"PID THRO:  {last_pid_out[2]:+.3f}",     True, (255, 220, 100)), (pid_x, pid_y + 82))
        screen.blit(font_small.render(f"PID PITCH: {last_pid_out[1]:+.3f}",     True, (255, 220, 100)), (pid_x, pid_y + 100))
        screen.blit(font_small.render(f"PID ROLL:  {last_pid_out[0]:+.3f}",     True, (255, 220, 100)), (pid_x, pid_y + 118))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
cap.release()
ser.close()