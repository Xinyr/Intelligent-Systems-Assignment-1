import collections
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# PATHS - update these to match your trained model locations
# ============================================================
PERSON_MODEL = r"../YOLOV8-PersonDetect/runs/detect/person_detector/weights/best.pt"
FALL_MODEL = r"../YOLOV8-FallDetect/runs/detect/fall_detection/weights/best.pt"
ONE_STEP_MODEL = FALL_MODEL

# Output folders - created automatically beside this script
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "detections"
IMAGE_SUBDIR = "images"
VIDEO_SUBDIR = "videos"
PRE_ROLL_SEC = 2
POST_ROLL_SEC = 4
SNAPSHOT_MIN_FRAMES = 2
FALL_EVENT_MIN_CONF = 0.40
FALL_EVENT_TRIGGER_FRAMES = 2
FALL_EVENT_CLEAR_FRAMES = 8
CLIP_COOLDOWN_SEC = 6
CLIP_FPS = 10.0
CLIP_SIZE = (640, 360)
PERSON_VALIDATION_CONF = 0.25
FALL_PERSON_OVERLAP_THRESHOLD = 0.30
TWO_STEP_PERSON_DETECT_CONF = 0.35
TWO_STEP_PERSON_MIN_CONF = 0.45
ALERT_HOLD_SEC = 2.0
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_ARG = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES_DISPLAY = {
    "fall": "FALL",
    "fall detected": "FALL",
    "walk": "Walk",
    "sit": "Sit",
}

CLASS_COLORS_BGR = {
    "fall": (0, 60, 255),
    "walk": (0, 220, 120),
    "sit": (0, 180, 255),
}

# Colour palette
BG_DEEP = "#050a0f"
BG_PANEL = "#0a1520"
BG_CARD = "#0d1d2e"
BG_ROW = "#0f2133"
ACCENT = "#00aaff"
ACCENT2 = "#00ffcc"
DANGER = "#ff3c3c"
WARNING = "#ffaa00"
SUCCESS = "#00dc78"
MUTED = "#4a7a9b"
TEXT_PRI = "#c8e8ff"
TEXT_SEC = "#5a8aaa"
BORDER = "#0f3355"
GRID_LINE = "#0c2035"


def make_section(parent, title):
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="x", padx=8, pady=4)
    inner = tk.Frame(outer, bg=BG_CARD)
    inner.pack(fill="both", expand=True)

    tk.Label(
        inner,
        text=f"  {title}",
        font=("Courier", 9, "bold"),
        fg=ACCENT,
        bg=BG_CARD,
        anchor="w",
    ).pack(fill="x", pady=(4, 0))

    tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)
    body = tk.Frame(inner, bg=BG_CARD)
    body.pack(fill="both", expand=True, padx=8, pady=6)
    return body


def cyber_button(parent, text, color, command):
    """Flat button with a left colour accent bar."""
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", pady=2)

    tk.Frame(row, bg=color, width=4).pack(side="left", fill="y")

    btn = tk.Button(
        row,
        text=text,
        font=("Courier", 9, "bold"),
        bg=BG_ROW,
        fg=color,
        relief="flat",
        bd=0,
        activebackground=BG_PANEL,
        activeforeground=TEXT_PRI,
        cursor="hand2",
        anchor="w",
        padx=10,
        pady=6,
        command=command,
    )
    btn.pack(side="left", fill="x", expand=True)
    return btn


def stat_row(parent, label, var, color):
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", pady=1)

    tk.Label(
        row,
        text=label,
        font=("Courier", 9),
        fg=TEXT_SEC,
        bg=BG_CARD,
        width=10,
        anchor="w",
    ).pack(side="left")

    bar = tk.Frame(row, bg=GRID_LINE, height=18)
    bar.pack(side="left", fill="x", expand=True, padx=(4, 0))

    tk.Label(
        bar,
        textvariable=var,
        font=("Courier", 9, "bold"),
        fg=color,
        bg=GRID_LINE,
        anchor="e",
        padx=6,
    ).pack(fill="both")


def resolve_model_path(model_path: str) -> Path:
    """Resolve relative model paths from the script folder first."""
    path = Path(model_path)
    if path.is_absolute():
        return path

    script_path = SCRIPT_DIR / path
    if script_path.exists():
        return script_path

    return path


class FallDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.paused = False
        self.cap = None
        self.alert_active = False
        self.fps = 0
        self.frame_count = 0
        self.last_fps_t = time.time()
        self.last_fall_time = 0
        self.current_mode = None
        self.stat_counts = {"falls": 0, "walks": 0, "sits": 0, "frames": 0}
        self.blink_state = False
        self._last_alert_time = 0.0

        self.last_frame = None
        self.last_fall_state = False
        self.last_fall_conf = 0.0
        self._last_save_time = 0.0
        self._auto_save_count = 0
        self._clip_count = 0
        self._consecutive_falls = 0
        self._clip_fall_streak = 0
        self._clip_clear_streak = FALL_EVENT_CLEAR_FRAMES
        self._fall_event_armed = True
        self._last_false_fall_log = 0.0

        self._pre_roll_fps = 25.0
        self._frame_buffer = collections.deque(maxlen=max(1, 25 * PRE_ROLL_SEC))
        self._post_roll_frames = 0
        self._clip_end_time = 0.0
        self._clip_writer = None
        self._clip_path = None
        self._missing_model_warnings = set()

        self._ensure_output_dirs()

        self._load_models()
        self._build_ui()
        self._tick_clock()

    # Models

    def _load_models(self):
        one_step_path = resolve_model_path(ONE_STEP_MODEL)
        person_path = resolve_model_path(PERSON_MODEL)
        fall_path = resolve_model_path(FALL_MODEL)

        one_step_ok = one_step_path.exists()
        person_ok = person_path.exists()
        fall_ok = fall_path.exists()

        self.yolo_one = YOLO(str(one_step_path)) if one_step_ok else None
        self.person_model = YOLO(str(person_path)) if person_ok else None
        self.fall_model = YOLO(str(fall_path)) if fall_ok else None
        self._one_step_ready = one_step_ok
        self._two_step_ready = person_ok and fall_ok

    def _ensure_output_dirs(self):
        for mode in ("one_step", "two_step"):
            (OUTPUT_DIR / mode / IMAGE_SUBDIR).mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / mode / VIDEO_SUBDIR).mkdir(parents=True, exist_ok=True)

    def _mode_output_root(self):
        return OUTPUT_DIR / self.det_mode.get()

    def _image_output_dir(self):
        path = self._mode_output_root() / IMAGE_SUBDIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _video_output_dir(self):
        path = self._mode_output_root() / VIDEO_SUBDIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _relative_output_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(SCRIPT_DIR))
        except ValueError:
            return str(path)

    def _mode_title(self):
        return "ONE-STEP" if self.det_mode.get() == "one_step" else "TWO-STEP"

    def _warn_missing_model_once(self, key: str, message: str):
        if key in self._missing_model_warnings:
            return
        self._missing_model_warnings.add(key)
        self.log(message, "warning")

    # UI

    def _build_ui(self):
        self.root.title("FALL DETECTION SYSTEM  //  COS30018")
        self.root.geometry("1280x820")
        self.root.configure(bg=BG_DEEP)
        self.root.resizable(True, True)

        self._build_topbar()

        body = tk.Frame(self.root, bg=BG_DEEP)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self._build_display(body)
        self._build_sidebar(body)
        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL)
        bar.pack(fill="x")
        tk.Frame(bar, bg=ACCENT, height=2).pack(fill="x")

        inner = tk.Frame(bar, bg=BG_PANEL, pady=8)
        inner.pack(fill="x", padx=12)

        left = tk.Frame(inner, bg=BG_PANEL)
        left.pack(side="left")

        tk.Label(
            left,
            text="[ FALL DETECTION SYSTEM ]",
            font=("Courier", 16, "bold"),
            fg=ACCENT,
            bg=BG_PANEL,
        ).pack(side="left")

        tk.Label(
            left,
            text="  //  COS30018 Intelligent Systems",
            font=("Courier", 10),
            fg=MUTED,
            bg=BG_PANEL,
        ).pack(side="left")

        right = tk.Frame(inner, bg=BG_PANEL)
        right.pack(side="right")

        dev_str = "GPU ONLINE" if torch.cuda.is_available() else "CPU MODE"
        dev_col = SUCCESS if torch.cuda.is_available() else WARNING

        tk.Label(
            right,
            text=f"DEVICE: {dev_str}",
            font=("Courier", 9, "bold"),
            fg=dev_col,
            bg=BG_PANEL,
        ).pack(side="right", padx=(12, 0))

        self.clock_var = tk.StringVar(value="00:00:00")
        tk.Label(
            right,
            textvariable=self.clock_var,
            font=("Courier", 10, "bold"),
            fg=TEXT_SEC,
            bg=BG_PANEL,
        ).pack(side="right", padx=(12, 0))

        tk.Label(
            right,
            text="SYS:",
            font=("Courier", 9),
            fg=MUTED,
            bg=BG_PANEL,
        ).pack(side="right")

    def _build_display(self, parent):
        left = tk.Frame(parent, bg=BG_DEEP)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.alert_banner = tk.Label(
            left,
            text="",
            font=("Courier", 13, "bold"),
            fg=BG_DEEP,
            bg=BG_DEEP,
            height=2,
        )
        self.alert_banner.pack(fill="x")

        cf = tk.Frame(left, bg=BORDER, padx=1, pady=1)
        cf.pack(fill="both", expand=True)

        ci = tk.Frame(cf, bg=BG_DEEP)
        ci.pack(fill="both", expand=True)

        self.canvas = tk.Label(
            ci,
            bg=BG_DEEP,
            text="SELECT INPUT SOURCE  >>",
            font=("Courier", 14, "bold"),
            fg=MUTED,
        )
        self.canvas.pack(fill="both", expand=True)

        info_bar = tk.Frame(left, bg=BG_PANEL, pady=3)
        info_bar.pack(fill="x")
        tk.Frame(info_bar, bg=BORDER, height=1).pack(fill="x")

        ib = tk.Frame(info_bar, bg=BG_PANEL)
        ib.pack(fill="x", padx=8, pady=2)

        self.mode_var = tk.StringVar(value="MODE: STANDBY")
        tk.Label(
            ib,
            textvariable=self.mode_var,
            font=("Courier", 9, "bold"),
            fg=ACCENT,
            bg=BG_PANEL,
        ).pack(side="left")

        self.fps_var = tk.StringVar(value="FPS: --")
        tk.Label(
            ib,
            textvariable=self.fps_var,
            font=("Courier", 9, "bold"),
            fg=ACCENT2,
            bg=BG_PANEL,
        ).pack(side="right")

        self.det_var = tk.StringVar(value="DETECTIONS: 0")
        tk.Label(
            ib,
            textvariable=self.det_var,
            font=("Courier", 9),
            fg=TEXT_SEC,
            bg=BG_PANEL,
        ).pack(side="right", padx=16)

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG_PANEL, width=274)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Frame(side, bg=ACCENT, height=2).pack(fill="x")
        tk.Label(
            side,
            text="  CONTROL PANEL",
            font=("Courier", 10, "bold"),
            fg=ACCENT,
            bg=BG_PANEL,
            anchor="w",
        ).pack(fill="x", pady=(6, 2))

        self._build_input_section(side)
        self._build_mode_section(side)
        self._build_conf_section(side)
        self._build_save_section(side)
        self._build_stats_section(side)
        self._build_log_section(side)

    def _build_input_section(self, parent):
        body = make_section(parent, ">> INPUT SOURCE")
        cyber_button(body, "  LOAD IMAGE", ACCENT, self.load_image)
        cyber_button(body, "  LOAD VIDEO", SUCCESS, self.load_video)
        cyber_button(body, "  LIVE WEBCAM", ACCENT2, self.start_webcam)

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=4)

        btn_row = tk.Frame(body, bg=BG_CARD)
        btn_row.pack(fill="x")

        self.pause_btn = tk.Button(
            btn_row,
            text="PAUSE",
            font=("Courier", 8, "bold"),
            bg=BG_ROW,
            fg=WARNING,
            relief="flat",
            bd=0,
            activebackground=BG_PANEL,
            cursor="hand2",
            state="disabled",
            padx=8,
            pady=5,
            command=self.toggle_pause,
        )
        self.pause_btn.pack(side="left", padx=(0, 2), fill="x", expand=True)

        self.stop_btn = tk.Button(
            btn_row,
            text="STOP",
            font=("Courier", 8, "bold"),
            bg=BG_ROW,
            fg=DANGER,
            relief="flat",
            bd=0,
            activebackground=BG_PANEL,
            cursor="hand2",
            state="disabled",
            padx=8,
            pady=5,
            command=self.stop,
        )
        self.stop_btn.pack(side="left", padx=(2, 0), fill="x", expand=True)

        tk.Button(
            body,
            text="CLEAR ALERT",
            font=("Courier", 8, "bold"),
            bg=BG_ROW,
            fg=MUTED,
            relief="flat",
            bd=0,
            activebackground=BG_PANEL,
            cursor="hand2",
            pady=4,
            command=self.clear_alert,
        ).pack(fill="x", pady=(4, 0))

    def _build_mode_section(self, parent):
        body = make_section(parent, ">> DETECTION MODE")
        self.det_mode = tk.StringVar(value="one_step")

        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x")

        for txt, val in [("ONE-STEP", "one_step"), ("TWO-STEP", "two_step")]:
            tk.Radiobutton(
                row,
                text=txt,
                variable=self.det_mode,
                value=val,
                font=("Courier", 9, "bold"),
                fg=ACCENT,
                bg=BG_CARD,
                selectcolor=BG_DEEP,
                activebackground=BG_CARD,
                activeforeground=TEXT_PRI,
                command=self._on_detection_mode_change,
            ).pack(side="left", padx=6)

        one_col = SUCCESS if self._one_step_ready else DANGER
        one_text = "ONE-STEP: READY" if self._one_step_ready else "ONE-STEP: MODEL NOT FOUND"
        ts_col = SUCCESS if self._two_step_ready else WARNING
        ts_text = "TWO-STEP: READY" if self._two_step_ready else "TWO-STEP: MODELS NOT FOUND"

        tk.Label(
            body,
            text=one_text,
            font=("Courier", 7),
            fg=one_col,
            bg=BG_CARD,
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

        tk.Label(
            body,
            text=ts_text,
            font=("Courier", 7),
            fg=ts_col,
            bg=BG_CARD,
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _build_conf_section(self, parent):
        body = make_section(parent, ">> CONFIDENCE THRESHOLD")
        self.conf_var = tk.DoubleVar(value=0.65)
        self.conf_label = tk.Label(
            body,
            text="CONF: 0.40",
            font=("Courier", 11, "bold"),
            fg=ACCENT2,
            bg=BG_CARD,
        )
        self.conf_label.pack()

        def _upd(v):
            self.conf_label.config(text=f"CONF: {float(v):.2f}")

        tk.Scale(
            body,
            from_=0.10,
            to=0.90,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_var,
            bg=BG_CARD,
            fg=ACCENT,
            troughcolor=BG_DEEP,
            highlightthickness=0,
            showvalue=False,
            command=_upd,
        ).pack(fill="x", padx=4)

    def _build_save_section(self, parent):
        body = make_section(parent, ">> AUTO SAVE OUTPUT")

        folder_row = tk.Frame(body, bg=BG_CARD)
        folder_row.pack(fill="x", pady=(0, 4))

        tk.Label(
            folder_row,
            text="DIR:",
            font=("Courier", 7),
            fg=TEXT_SEC,
            bg=BG_CARD,
        ).pack(side="left")

        self.save_dir_var = tk.StringVar(value="detections/one_step")
        tk.Label(
            folder_row,
            textvariable=self.save_dir_var,
            font=("Courier", 7, "bold"),
            fg=ACCENT2,
            bg=BG_CARD,
        ).pack(side="left", padx=4)

        self.auto_save_var = tk.StringVar(value="FALL SNAPSHOTS: 0")
        tk.Label(
            body,
            textvariable=self.auto_save_var,
            font=("Courier", 8, "bold"),
            fg=DANGER,
            bg=BG_CARD,
            anchor="w",
        ).pack(fill="x", pady=(0, 2))

        self.clip_count_var = tk.StringVar(value="FALL CLIPS: 0")
        tk.Label(
            body,
            textvariable=self.clip_count_var,
            font=("Courier", 8, "bold"),
            fg=WARNING,
            bg=BG_CARD,
            anchor="w",
        ).pack(fill="x", pady=(0, 2))

        self.rec_status_var = tk.StringVar(value="Video: IDLE")
        tk.Label(
            body,
            textvariable=self.rec_status_var,
            font=("Courier", 7, "bold"),
            fg=MUTED,
            bg=BG_CARD,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        tk.Label(
            body,
            text=f"Video event clips only:\n{PRE_ROLL_SEC}s before + fall + {POST_ROLL_SEC}s after",
            font=("Courier", 7),
            fg=MUTED,
            bg=BG_CARD,
            anchor="w",
            justify="left",
        ).pack(fill="x")

    def _build_stats_section(self, parent):
        body = make_section(parent, ">> DETECTION STATS")
        self.stat_vars = {}

        for label, key, color in [
            ("FALL", "falls", DANGER),
            ("WALK", "walks", SUCCESS),
            ("SIT", "sits", ACCENT),
            ("FRAMES", "frames", TEXT_SEC),
        ]:
            var = tk.StringVar(value="0")
            stat_row(body, label, var, color)
            self.stat_vars[key] = var

        tk.Button(
            body,
            text="RESET STATS",
            font=("Courier", 8),
            bg=BG_DEEP,
            fg=MUTED,
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=2,
            command=self._reset_stats,
        ).pack(fill="x", pady=(4, 0))

    def _build_log_section(self, parent):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both", expand=True, padx=8, pady=4)

        inner = tk.Frame(outer, bg=BG_CARD)
        inner.pack(fill="both", expand=True)

        tk.Label(
            inner,
            text="  >> EVENT LOG",
            font=("Courier", 9, "bold"),
            fg=ACCENT,
            bg=BG_CARD,
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)

        self.log_box = tk.Text(
            inner,
            bg=BG_DEEP,
            fg=TEXT_SEC,
            font=("Courier", 8),
            state="disabled",
            relief="flat",
            wrap="word",
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_box.tag_config("fall", foreground=DANGER)
        self.log_box.tag_config("ok", foreground=SUCCESS)
        self.log_box.tag_config("info", foreground=ACCENT)
        self.log_box.tag_config("warning", foreground=WARNING)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL)
        bar.pack(fill="x", side="bottom")
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x")

        inner = tk.Frame(bar, bg=BG_PANEL)
        inner.pack(fill="x", padx=10, pady=3)

        self.status_var = tk.StringVar(value="SYSTEM READY  //  SELECT INPUT SOURCE TO BEGIN")
        tk.Label(
            inner,
            textvariable=self.status_var,
            font=("Courier", 8),
            fg=MUTED,
            bg=BG_PANEL,
            anchor="w",
        ).pack(side="left")

        if self._one_step_ready and self._two_step_ready:
            model_str, model_col = "MODELS: ONE-STEP + TWO-STEP LOADED", SUCCESS
        elif self._one_step_ready:
            model_str, model_col = "MODEL: ONE-STEP ONLY", WARNING
        elif self._two_step_ready:
            model_str, model_col = "MODEL: TWO-STEP ONLY", WARNING
        else:
            model_str, model_col = "MODELS: REQUIRED WEIGHTS MISSING", DANGER

        tk.Label(
            inner,
            text=model_str,
            font=("Courier", 8, "bold"),
            fg=model_col,
            bg=BG_PANEL,
        ).pack(side="right")

    # Clock

    def _tick_clock(self):
        self.clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # Thread-safe GUI helpers

    def _run_on_ui(self, func, *args):
        if threading.current_thread() is threading.main_thread():
            func(*args)
        else:
            try:
                self.root.after(0, func, *args)
            except tk.TclError:
                pass

    def _set_stringvar(self, var, value):
        self._run_on_ui(var.set, value)

    def log(self, msg, tag="info"):
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self.log, msg, tag)
            except tk.TclError:
                pass
            return

        ts = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _reset_stats(self):
        self.stat_counts = {"falls": 0, "walks": 0, "sits": 0, "frames": 0}
        self._auto_save_count = 0
        self._clip_count = 0
        self._consecutive_falls = 0
        self._clip_fall_streak = 0
        self._clip_clear_streak = FALL_EVENT_CLEAR_FRAMES
        self._fall_event_armed = True
        self.auto_save_var.set("FALL SNAPSHOTS: 0")
        self.clip_count_var.set("FALL CLIPS: 0")

        for var in self.stat_vars.values():
            var.set("0")

    def clear_alert(self):
        self.alert_active = False
        self.alert_banner.config(text="", bg=BG_DEEP)
        self.log("ALERT CLEARED", "ok")

    def _on_detection_mode_change(self):
        mode_name = self._mode_title()

        if hasattr(self, "save_dir_var"):
            self.save_dir_var.set(f"detections/{self.det_mode.get()}")

        if self.current_mode and self.current_mode != "image":
            self._flush_clip()
            self.status_var.set(f"RUNNING  //  {self.current_mode.upper()}  //  {mode_name}")
        elif self.current_mode == "image":
            self.status_var.set(f"IMAGE  //  {mode_name}")
        else:
            self.status_var.set(f"SYSTEM READY  //  {mode_name} MODE SELECTED")

        self.mode_var.set(f"MODE: {mode_name}")
        self.log(f"DETECTION MODE -> {mode_name}", "info")

    def _set_running_state(self, running):
        state = "normal" if running else "disabled"
        self.stop_btn.config(state=state)
        self.pause_btn.config(state=state)

    def stop(self):
        self.running = False
        self.paused = False
        self._flush_clip()
        self._consecutive_falls = 0
        self._clip_fall_streak = 0
        self._clip_clear_streak = FALL_EVENT_CLEAR_FRAMES
        self._fall_event_armed = True
        self.alert_active = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self._set_running_state(False)
        self.pause_btn.config(text="PAUSE")
        self.status_var.set("STOPPED  //  SELECT INPUT SOURCE TO BEGIN")
        self.mode_var.set("MODE: STANDBY")
        self.fps_var.set("FPS: --")
        self.rec_status_var.set("Video: IDLE")
        self.alert_banner.config(text="", bg=BG_DEEP)
        self.log("SYSTEM STOPPED", "warning")

    def toggle_pause(self):
        if self.current_mode == "image":
            return

        self.paused = not self.paused

        if self.paused:
            self.pause_btn.config(text="RESUME")
            self.status_var.set("PAUSED")
            self.log("PAUSED", "warning")
        else:
            self.pause_btn.config(text="PAUSE")
            self.status_var.set(f"RUNNING  //  {self.current_mode.upper()}  //  {self._mode_title()}")
            self.log("RESUMED", "ok")

    # Save helpers

    def _draw_alert_overlay(self, frame: np.ndarray, fall: bool, conf: float = 0.0):
        h, w = frame.shape[:2]

        if fall:
            text = f"!! FALL DETECTED ({conf:.2f}) - ALERT !!"
            color = (60, 60, 255)
            bg = (0, 0, 140)
        else:
            text = "OK  MONITORING - NO FALL DETECTED"
            color = (80, 220, 80)
            bg = (10, 60, 10)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        banner_h = th + 16

        cv2.rectangle(frame, (0, 0), (w, banner_h), bg, -1)
        cv2.putText(
            frame,
            text,
            ((w - tw) // 2, th + 6),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _auto_save_fall(self, frame: np.ndarray, conf: float):
        if conf < self.conf_var.get():
            return
        if self.current_mode != "image" and self._consecutive_falls < SNAPSHOT_MIN_FRAMES:
            return

        now = time.time()
        if now - self._last_save_time < 2.0:
            return

        self._last_save_time = now

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._image_output_dir() / f"fall_{self.det_mode.get()}_{ts}_conf{conf:.2f}.jpg"
        out = frame.copy()
        self._draw_alert_overlay(out, True, conf)
        cv2.imwrite(str(path), out)

        self._auto_save_count += 1
        self.auto_save_var.set(f"FALL SNAPSHOTS: {self._auto_save_count}")
        self.log(f"AUTO-SAVED -> {self._relative_output_path(path)}", "fall")

    def _init_buffer(self, fps: float):
        self._pre_roll_fps = fps if fps and fps > 0 else 25.0
        maxlen = max(1, int(self._pre_roll_fps * PRE_ROLL_SEC))
        self._frame_buffer = collections.deque(maxlen=maxlen)
        self._post_roll_frames = 0
        self._clip_end_time = 0.0
        self._clip_writer = None
        self._clip_path = None
        self._clip_fall_streak = 0
        self._clip_clear_streak = FALL_EVENT_CLEAR_FRAMES
        self._fall_event_armed = True
        self.last_fall_time = 0
        if hasattr(self, "rec_status_var"):
            self._set_stringvar(self.rec_status_var, "Video: MONITORING")

    def _is_clip_fall_candidate(self, fall: bool, conf: float) -> bool:
        return fall and conf >= max(self.conf_var.get(), FALL_EVENT_MIN_CONF)

    def _trim_frame_buffer(self, now: float):
        while self._frame_buffer and now - self._frame_buffer[0][0] > PRE_ROLL_SEC:
            self._frame_buffer.popleft()

    def _estimate_clip_fps(self, buffered_frames) -> float:
        if len(buffered_frames) >= 2:
            elapsed = buffered_frames[-1][0] - buffered_frames[0][0]
            if elapsed > 0:
                measured_fps = (len(buffered_frames) - 1) / elapsed
                return max(5.0, min(CLIP_FPS, measured_fps))
        return CLIP_FPS

    def _push_frame(self, frame: np.ndarray, fall: bool, conf: float):
        """
        Store a Video-style rolling buffer and save only fall-event clips.

        All frames are resized to CLIP_SIZE before entering the buffer/writer.
        This prevents broken clips caused by writing mixed frame dimensions.
        """
        annotated = frame.copy()
        self._draw_alert_overlay(annotated, fall, conf)
        clip_frame = cv2.resize(annotated, CLIP_SIZE)

        now = time.time()
        self._frame_buffer.append((now, clip_frame))
        self._trim_frame_buffer(now)

        fall_candidate = self._is_clip_fall_candidate(fall, conf)

        if fall_candidate:
            self._clip_fall_streak += 1
            self._clip_clear_streak = 0
        else:
            self._clip_fall_streak = 0
            self._clip_clear_streak += 1
            if self._clip_clear_streak >= FALL_EVENT_CLEAR_FRAMES:
                self._fall_event_armed = True

        if self._clip_writer is None:
            if not self._fall_event_armed:
                return

            if self._clip_fall_streak < FALL_EVENT_TRIGGER_FRAMES:
                return

            if now - self.last_fall_time < CLIP_COOLDOWN_SEC:
                return

            self._fall_event_armed = False
            self.last_fall_time = now
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self._video_output_dir() / f"fall_clip_{self.det_mode.get()}_{ts}.avi"
            buffered_frames = list(self._frame_buffer)
            clip_fps = self._estimate_clip_fps(buffered_frames)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(path), fourcc, clip_fps, CLIP_SIZE)

            if not writer.isOpened():
                writer.release()
                self.log("ERROR: Failed to create video writer", "warning")
                return

            for _, buffered in buffered_frames:
                writer.write(buffered)

            self._frame_buffer.clear()

            self._clip_writer = writer
            self._clip_path = path
            self._clip_end_time = now + POST_ROLL_SEC

            self._clip_count += 1
            self._set_stringvar(self.clip_count_var, f"FALL CLIPS: {self._clip_count}")
            self._set_stringvar(self.rec_status_var, f"Video: SAVING {path.name}")
            self.log(f"CLIP STARTED -> {self._relative_output_path(path)}", "fall")
            return

        if self._clip_writer is not None:
            self._clip_writer.write(clip_frame)

            if now >= self._clip_end_time:
                self._close_clip()

    def _close_clip(self):
        if self._clip_writer is not None:
            self._clip_writer.release()
            if self._clip_path is not None:
                self.log(f"CLIP SAVED -> {self._relative_output_path(self._clip_path)}", "ok")
            self._clip_writer = None
            self._clip_path = None

        self._post_roll_frames = 0
        self._clip_end_time = 0.0
        self._set_stringvar(self.rec_status_var, "Video: MONITORING")

    def _flush_clip(self):
        if self._clip_writer is not None:
            self._close_clip()
        self._clip_fall_streak = 0
        self._clip_clear_streak = FALL_EVENT_CLEAR_FRAMES
        self._fall_event_armed = True

    # Detection

    def _normalise_class(self, raw: str) -> str:
        name = raw.lower().strip()

        if name in ("fall", "fall detected"):
            return "fall"
        if name in ("walk", "walking"):
            return "walk"
        if name in ("sit", "sitting"):
            return "sit"
        if name in ("person", "people", "human"):
            return "person"

        return name

    def _bbox_area(self, box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _bbox_intersection(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _bbox_center_inside(self, inner, outer):
        x1, y1, x2, y2 = inner
        ox1, oy1, ox2, oy2 = outer
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

    def _detect_person_boxes_for_validation(self, frame):
        if self.person_model is None:
            return []

        h, w = frame.shape[:2]
        results = self.person_model.predict(
            frame,
            conf=PERSON_VALIDATION_CONF,
            device=DEVICE_ARG,
            verbose=False,
        )
        boxes = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

        return boxes

    def _fall_has_person_context(self, fall_box, person_boxes):
        fall_area = self._bbox_area(fall_box)
        if fall_area <= 0:
            return False

        for person_box in person_boxes:
            if self._bbox_center_inside(fall_box, person_box):
                return True

            overlap = self._bbox_intersection(fall_box, person_box) / fall_area
            if overlap >= FALL_PERSON_OVERLAP_THRESHOLD:
                return True

        return False

    def _filter_one_step_false_falls(self, frame, dets):
        if not any(d["class_key"] == "fall" for d in dets):
            return dets

        person_boxes = [
            d["bbox"]
            for d in dets
            if d["class_key"] in ("walk", "sit")
        ]
        person_boxes.extend(self._detect_person_boxes_for_validation(frame))

        if not person_boxes:
            return dets

        filtered = []
        rejected = 0

        for det in dets:
            if det["class_key"] != "fall":
                filtered.append(det)
                continue

            if self._fall_has_person_context(det["bbox"], person_boxes):
                filtered.append(det)
            else:
                rejected += 1

        if rejected:
            now = time.time()
            if now - self._last_false_fall_log >= 3.0:
                self._last_false_fall_log = now
                self.log(f"IGNORED FALL BOX WITHOUT PERSON ({rejected})", "warning")

        return filtered

    def _detect_one_step(self, frame):
        if self.yolo_one is None:
            self._warn_missing_model_once(
                "one_step",
                "ONE-STEP model missing - cannot run direct fall/action detection",
            )
            return []

        conf = self.conf_var.get()
        h, w = frame.shape[:2]
        results = self.yolo_one.predict(frame, conf=conf, device=DEVICE_ARG, verbose=False)
        dets = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                raw = results[0].names[int(box.cls[0])]
                cls_key = self._normalise_class(raw)

                dets.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "class_key": cls_key,
                        "class_name": CLASS_NAMES_DISPLAY.get(cls_key, raw.upper()),
                        "confidence": float(box.conf[0]),
                    }
                )

        return self._filter_one_step_false_falls(frame, dets)

    def _detect_two_step(self, frame):
        if not self._two_step_ready:
            self._warn_missing_model_once(
                "two_step",
                "TWO-STEP models missing - person crop + fall classification pipeline unavailable",
            )
            return []

        conf = self.conf_var.get()
        h, w = frame.shape[:2]
        dets = []

        person_results = self.person_model.predict(
            frame,
            conf=TWO_STEP_PERSON_DETECT_CONF,
            device=DEVICE_ARG,
            verbose=False,
        )
        if person_results[0].boxes is None or len(person_results[0].boxes) == 0:
            return dets

        for box in person_results[0].boxes:
            person_conf = float(box.conf[0])
            if person_conf < TWO_STEP_PERSON_MIN_CONF:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            fall_results = self.fall_model.predict(crop, conf=conf, device=DEVICE_ARG, verbose=False)
            cls_key = "unknown"
            best_c = 0.0

            if fall_results[0].boxes is not None and len(fall_results[0].boxes) > 0:
                best = max(fall_results[0].boxes, key=lambda b: float(b.conf[0]))
                cls_key = self._normalise_class(fall_results[0].names[int(best.cls[0])])
                best_c = float(best.conf[0])

            if cls_key == "unknown":
                continue

            dets.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_key": cls_key,
                    "class_name": CLASS_NAMES_DISPLAY.get(cls_key, cls_key.upper()),
                    "confidence": best_c,
                    "person_confidence": person_conf,
                    "mode": "two_step",
                }
            )

        return dets

    def _detect(self, frame):
        if self.det_mode.get() == "two_step":
            return self._detect_two_step(frame)
        return self._detect_one_step(frame)

    # Drawing

    def _draw_detections(self, frame, dets):
        fall_detected = False
        fall_conf = 0.0

        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cls_key = d["class_key"]
            name = d["class_name"]
            conf = d["confidence"]
            color = CLASS_COLORS_BGR.get(cls_key, (255, 255, 255))
            if d.get("mode") == "two_step":
                label = f"Person, {name} ({conf:.2f})"
            else:
                label = f"{name} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            tl, sz = 8, 3
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1),
                (x2, y1, -1, 1),
                (x1, y2, 1, -1),
                (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * tl, cy), color, sz)
                cv2.line(frame, (cx, cy), (cx, cy + dy * tl), color, sz)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            if cls_key == "fall":
                fall_detected = True
                fall_conf = max(fall_conf, conf)
                self.stat_counts["falls"] += 1
            elif cls_key == "walk":
                self.stat_counts["walks"] += 1
            elif cls_key == "sit":
                self.stat_counts["sits"] += 1

        self._set_stringvar(self.det_var, f"DETECTIONS: {len(dets)}")
        return frame, fall_detected, fall_conf

    def _show_frame(self, frame, fall=False, fall_conf=0.0):
        self.last_frame = frame
        self.last_fall_state = fall
        self.last_fall_conf = fall_conf

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()

        if cw > 10 and ch > 10:
            fw, fh = img.size
            scale = min(cw / fw, ch / fh)
            img = img.resize((max(1, int(fw * scale)), max(1, int(fh * scale))), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=imgtk, text="")
        self.canvas.image = imgtk

        if fall:
            self._consecutive_falls += 1
            self.alert_active = True
            self._last_alert_time = time.time()
            self.blink_state = not self.blink_state
            bg = DANGER if self.blink_state else "#aa2222"
            self.alert_banner.config(
                text=f"  !!  FALL DETECTED ({fall_conf:.2f}) - ALERT  !!",
                fg=BG_DEEP,
                bg=bg,
            )
            self.log(f"FALL DETECTED! conf={fall_conf:.2f}", "fall")
            self._auto_save_fall(frame, fall_conf)
        else:
            self._consecutive_falls = 0
            if self.alert_active and self.current_mode != "image":
                if time.time() - self._last_alert_time >= ALERT_HOLD_SEC:
                    self.alert_active = False

            if not self.alert_active:
                self.alert_banner.config(
                    text="  OK  MONITORING - NO FALL DETECTED",
                    fg=BG_DEEP,
                    bg="#0a3d1f",
                )

        fps_text = f"FPS: {self.fps}" if self.current_mode != "image" else "FPS: IMAGE"
        self.fps_var.set(fps_text)

        for key, var in self.stat_vars.items():
            var.set(str(self.stat_counts.get(key, 0)))

    # Image mode

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        self.stop()
        self.current_mode = "image"
        self._reset_stats()
        self.log(f"IMAGE LOADED: {Path(path).name}", "info")
        self.status_var.set(f"IMAGE  //  {Path(path).name}  //  {self._mode_title()}")
        self.mode_var.set(f"MODE: IMAGE  //  {self._mode_title()}")

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image file!")
            return

        dets = self._detect(frame)
        frame, fall, fconf = self._draw_detections(frame, dets)

        self.stat_counts["frames"] += 1
        self.last_frame = frame
        self.last_fall_state = fall
        self.last_fall_conf = fconf

        self.root.after(100, self._show_frame, frame, fall, fconf)
        self._set_running_state(False)

        for d in dets:
            tag = "fall" if d["class_key"] == "fall" else "ok"
            self.log(f"  DET: {d['class_name']} ({d['confidence']:.2f})", tag)

        if not dets:
            self.log("  NO DETECTIONS", "warning")

    # Video mode

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        self.stop()
        self._reset_stats()
        self.current_mode = "video"
        self.running = True

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file!")
            self.running = False
            return

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_s = self.cap.get(cv2.CAP_PROP_FPS)
        src_fps = fps_s if fps_s and fps_s > 0 else 25.0

        self._init_buffer(src_fps)
        self._set_running_state(True)
        self.mode_var.set(f"MODE: VIDEO  //  {self._mode_title()}")
        self.status_var.set(
            f"VIDEO  //  {Path(path).name}  [{total} frames  {src_fps:.0f}fps]  //  {self._mode_title()}"
        )
        self.log(f"VIDEO: {Path(path).name}  {total}fr  {src_fps:.0f}fps", "info")

        threading.Thread(target=self._video_loop, daemon=True).start()

    def _video_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.log("VIDEO FINISHED", "ok")
                self.root.after(0, self.stop)
                break

            dets = self._detect(frame)
            frame, fall, fconf = self._draw_detections(frame, dets)

            self.stat_counts["frames"] += 1
            self.frame_count += 1

            now = time.time()
            if now - self.last_fps_t >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_t = now

            self.root.after(0, self._show_frame, frame, fall, fconf)
            self._push_frame(frame, fall, fconf)
            time.sleep(0.01)

    # Webcam mode

    def start_webcam(self):
        self.stop()
        self._reset_stats()
        self.current_mode = "webcam"
        self.running = True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Error",
                "Webcam not found!\nMake sure your camera is connected.",
            )
            self.running = False
            return

        self._set_running_state(True)
        self.mode_var.set(f"MODE: WEBCAM LIVE  //  {self._mode_title()}")
        self.status_var.set(f"LIVE  //  WEBCAM STREAM ACTIVE  //  {self._mode_title()}")
        self.log("WEBCAM STREAM STARTED", "ok")

        webcam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._init_buffer(webcam_fps if webcam_fps and webcam_fps > 0 else 25.0)

        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def _webcam_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.log("WEBCAM SIGNAL LOST", "warning")
                self.root.after(0, self.stop)
                break

            dets = self._detect(frame)
            frame, fall, fconf = self._draw_detections(frame, dets)

            self.stat_counts["frames"] += 1
            self.frame_count += 1

            now = time.time()
            if now - self.last_fps_t >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_t = now

            self.root.after(0, self._show_frame, frame, fall, fconf)
            self._push_frame(frame, fall, fconf)


if __name__ == "__main__":
    root = tk.Tk()
    app = FallDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()
