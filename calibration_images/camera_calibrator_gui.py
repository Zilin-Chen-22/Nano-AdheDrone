"""
Camera Calibrator GUI
--------------------
A Tkinter-based GUI that guides you through camera calibration using a chessboard.

Features:
- Live camera preview
- Capture chessboard images (saved to ./calibration_images/)
- Detect and visualize chessboard corners for each capture
- Run calibration on collected images and show reprojection error
- Save / Load calibration parameters (camera_params.npz)
- Stream undistorted live preview using loaded params

Dependencies:
- OpenCV (opencv-python or opencv-contrib-python)
- numpy
- Pillow

Run:
    python camera_calibrator_gui.py

Default chessboard size entries are pre-filled with 12 x 9 (you can change them).
"""

import os
import threading
import time
import glob
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# -------------------------- Configuration --------------------------
CAPTURE_FOLDER = "calibration_images"
PARAMS_FILE = "camera_params.npz"

os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# -------------------------- Helper functions -----------------------

def ensure_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# -------------------------- GUI App -------------------------------
class CameraCalibratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Camera Calibrator")

        # State
        self.cap = None
        self.running = False
        self.frame = None
        self.undistort_mode = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None

        # Default chessboard (pre-filled per user request)
        self.chess_w_var = tk.IntVar(value=12)  # columns
        self.chess_h_var = tk.IntVar(value=9)   # rows

        # UI layout
        self._build_controls()
        self._build_video_panel()
        self._build_status_panel()

        # Periodic updater
        self.update_job = None

    def _build_controls(self):
        frm = ttk.Frame(self.master)
        frm.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        # Camera controls
        self.open_btn = ttk.Button(frm, text="Open Camera", command=self.open_camera)
        self.open_btn.grid(row=0, column=0, padx=4)

        self.close_btn = ttk.Button(frm, text="Close Camera", command=self.close_camera, state=tk.DISABLED)
        self.close_btn.grid(row=0, column=1, padx=4)

        ttk.Label(frm, text="Camera ID:").grid(row=0, column=2, padx=(10,0))
        self.camera_id = tk.IntVar(value=0)
        self.cam_id_spin = ttk.Spinbox(frm, from_=0, to=10, width=4, textvariable=self.camera_id)
        self.cam_id_spin.grid(row=0, column=3)

        # Chessboard size
        ttk.Label(frm, text="Chessboard W:").grid(row=1, column=0)
        self.chess_w = ttk.Entry(frm, width=5, textvariable=self.chess_w_var)
        self.chess_w.grid(row=1, column=1)
        ttk.Label(frm, text="H:").grid(row=1, column=2)
        self.chess_h = ttk.Entry(frm, width=5, textvariable=self.chess_h_var)
        self.chess_h.grid(row=1, column=3)

        # Capture and process buttons
        self.capture_btn = ttk.Button(frm, text="Capture Frame (s)", command=self.capture_frame, state=tk.DISABLED)
        self.capture_btn.grid(row=2, column=0, pady=6)

        self.detect_btn = ttk.Button(frm, text="Detect Corners (d)", command=self.detect_corners_in_last, state=tk.DISABLED)
        self.detect_btn.grid(row=2, column=1)

        self.calibrate_btn = ttk.Button(frm, text="Calibrate", command=self.calibrate_camera, state=tk.DISABLED)
        self.calibrate_btn.grid(row=2, column=2)

        self.save_btn = ttk.Button(frm, text="Save Params", command=self.save_params, state=tk.DISABLED)
        self.save_btn.grid(row=2, column=3)

        self.load_btn = ttk.Button(frm, text="Load Params", command=self.load_params)
        self.load_btn.grid(row=0, column=4, padx=10)

        self.undistort_btn = ttk.Button(frm, text="Toggle Undistort Preview", command=self.toggle_undistort, state=tk.DISABLED)
        self.undistort_btn.grid(row=1, column=4)

        # Key bindings
        self.master.bind('<s>', lambda e: self.capture_frame())
        self.master.bind('<d>', lambda e: self.detect_corners_in_last())

    def _build_video_panel(self):
        vfrm = ttk.Frame(self.master)
        vfrm.pack(side=tk.LEFT, padx=6, pady=6)

        self.video_label = ttk.Label(vfrm)
        self.video_label.pack()

        # thumbnails / image list
        rfrm = ttk.Frame(self.master)
        rfrm.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Label(rfrm, text="Captured Images:").pack(anchor=tk.W)
        self.img_listbox = tk.Listbox(rfrm, width=40, height=20)
        self.img_listbox.pack(fill=tk.BOTH, expand=True)
        self.img_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.refresh_captured_list()

    def _build_status_panel(self):
        sfrm = ttk.Frame(self.master)
        sfrm.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(sfrm, textvariable=self.status_var).pack(anchor=tk.W)

    # ---------------- Camera control ----------------
    def open_camera(self):
        cam_id = int(self.camera_id.get())
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cam_id}")
            return
        self.running = True
        self.open_btn.config(state=tk.DISABLED)
        self.close_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.NORMAL)
        self.detect_btn.config(state=tk.NORMAL)
        self.calibrate_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.undistort_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Camera {cam_id} opened")
        threading.Thread(target=self._camera_loop, daemon=True).start()
        self.schedule_update()

    def close_camera(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.open_btn.config(state=tk.NORMAL)
        self.close_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        self.calibrate_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.undistort_btn.config(state=tk.DISABLED)
        self.status_var.set("Camera closed")
        if self.update_job:
            self.master.after_cancel(self.update_job)
            self.update_job = None

    def _camera_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.frame = frame.copy()
            time.sleep(0.01)

    def schedule_update(self):
        self._update_frame()
        self.update_job = self.master.after(30, self.schedule_update)

    def _update_frame(self):
        if self.frame is None:
            return
        display = self.frame
        if self.undistort_mode and self.camera_matrix is not None and self.dist_coeffs is not None:
            try:
                display = cv2.undistort(self.frame, self.camera_matrix, self.dist_coeffs)
            except Exception as e:
                pass
        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((800, 600))
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ---------------- Capture & file list ----------------
    def capture_frame(self):
        if self.frame is None:
            messagebox.showwarning("Warning", "No frame to capture")
            return
        idx = len(glob.glob(os.path.join(CAPTURE_FOLDER, "*.png")))
        path = os.path.join(CAPTURE_FOLDER, f"img_{idx:03d}.png")
        cv2.imwrite(path, self.frame)
        self.status_var.set(f"Saved {path}")
        self.refresh_captured_list()

    def refresh_captured_list(self):
        self.img_listbox.delete(0, tk.END)
        files = sorted(glob.glob(os.path.join(CAPTURE_FOLDER, "*.png")))
        for f in files:
            self.img_listbox.insert(tk.END, os.path.basename(f))

    def on_image_select(self, evt):
        sel = self.img_listbox.curselection()
        if not sel:
            return
        name = self.img_listbox.get(sel[0])
        path = os.path.join(CAPTURE_FOLDER, name)
        img = cv2.imread(path)
        if img is None:
            return
        # show in a popup
        top = tk.Toplevel(self.master)
        top.title(name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(pil)
        lbl = ttk.Label(top, image=imgtk)
        lbl.image = imgtk
        lbl.pack()

    # ---------------- Corner detection & calibration ----------------
    def detect_corners_in_last(self):
        files = sorted(glob.glob(os.path.join(CAPTURE_FOLDER, "*.png")))
        if not files:
            messagebox.showinfo("Info", "No captured images found")
            return
        last = files[-1]
        img = cv2.imread(last)
        gray = ensure_gray(img)
        cb_w = int(self.chess_w_var.get())
        cb_h = int(self.chess_h_var.get())
        pattern = (cb_w, cb_h)
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            messagebox.showwarning("Warning", f"Chessboard not found in {os.path.basename(last)} with pattern {pattern}")
            return
        # refine
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern, corners2, found)
        # save visualization
        vis_path = last.replace('.png', '_corners.png')
        cv2.imwrite(vis_path, vis)
        self.status_var.set(f"Corners detected and saved visualization: {os.path.basename(vis_path)}")
        self.refresh_captured_list()

    def calibrate_camera(self):
        files = sorted(glob.glob(os.path.join(CAPTURE_FOLDER, "*.png")))
        if not files:
            messagebox.showinfo("Info", "No captured images to calibrate")
            return
        cb_w = int(self.chess_w_var.get())
        cb_h = int(self.chess_h_var.get())
        pattern = (cb_w, cb_h)

        objp = np.zeros((pattern[0]*pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        good_files = []

        for fname in files:
            img = cv2.imread(fname)
            gray = ensure_gray(img)
            found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                objpoints.append(objp)
                good_files.append(fname)

        if len(objpoints) < 3:
            messagebox.showwarning("Warning", f"Not enough valid images found for calibration (found {len(objpoints)}). Need at least 3.")
            return

        image_shape = ensure_gray(cv2.imread(good_files[0])).shape[::-1]
        self.image_size = image_shape

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
        mean_error = self._compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        msg = f"Calibration done. RMS={ret:.4f}, Reprojection Error={mean_error:.4f}\nFound valid images: {len(objpoints)}"
        messagebox.showinfo("Calibration Result", msg)
        self.status_var.set(msg)

    def _compute_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
            total_error += error**2
            total_points += len(objpoints[i])
        return np.sqrt(total_error / total_points)

    def save_params(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            messagebox.showwarning("Warning", "No calibration parameters to save")
            return
        np.savez(PARAMS_FILE, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs, image_size=self.image_size)
        messagebox.showinfo("Saved", f"Parameters saved to {PARAMS_FILE}")
        self.status_var.set(f"Saved params to {PARAMS_FILE}")

    def load_params(self):
        path = filedialog.askopenfilename(title="Select params .npz file", filetypes=[("NPZ files","*.npz")], initialdir='.')
        if not path:
            return
        try:
            data = np.load(path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.image_size = tuple(data['image_size']) if 'image_size' in data else None
            messagebox.showinfo("Loaded", f"Loaded parameters from {os.path.basename(path)}")
            self.status_var.set(f"Loaded params from {os.path.basename(path)}")
            self.undistort_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load params: {e}")

    def toggle_undistort(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            messagebox.showwarning("Warning", "Load or compute camera parameters first")
            return
        self.undistort_mode = not self.undistort_mode
        self.status_var.set("Undistort: ON" if self.undistort_mode else "Undistort: OFF")

# -------------------------- Main ---------------------------------

def main():
    root = tk.Tk()
    app = CameraCalibratorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close_camera(), root.destroy()))
    root.mainloop()

if __name__ == '__main__':
    main()
