import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# PATHS — update these after training
# ============================================================
ONE_STEP_MODEL  = r"runs\one_step\fall_detection\weights\best.pt"
CLASSIFIER_PATH = r"classifier_model\best_classifier.pth"
# ============================================================

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES      = {0: "FALL", 1: "Walk", 2: "Sit"}
CLASS_COLORS_BGR = {0: (0, 60, 255), 1: (0, 220, 120), 2: (0, 180, 255)}
CLASS_COLORS_HEX = {0: "#ff3c3c", 1: "#00dc78", 2: "#00b4ff"}

# ── Robotic colour palette ────────────────────────────────────
BG_DEEP    = "#050a0f"   # near-black background
BG_PANEL   = "#0a1520"   # panel background
BG_CARD    = "#0d1d2e"   # card / section background
BG_ROW     = "#0f2133"   # alternating row
ACCENT     = "#00aaff"   # electric blue
ACCENT2    = "#00ffcc"   # cyan-green for positive
DANGER     = "#ff3c3c"   # red alert
WARNING    = "#ffaa00"   # amber warning
SUCCESS    = "#00dc78"   # green ok
MUTED      = "#4a7a9b"   # muted text
TEXT_PRI   = "#c8e8ff"   # primary text
TEXT_SEC   = "#5a8aaa"   # secondary text
BORDER     = "#0f3355"   # border lines
GRID_LINE  = "#0c2035"   # subtle grid

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_classifier():
    m = models.resnet18(pretrained=False)
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)
    )
    return m.to(DEVICE)


# ── Custom styled widgets ─────────────────────────────────────

def make_section(parent, title):
    """Labelled card section with robotic styling."""
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="x", padx=8, pady=4)
    inner = tk.Frame(outer, bg=BG_CARD)
    inner.pack(fill="both", expand=True)
    tk.Label(inner,
             text=f"  {title}",
             font=("Courier", 9, "bold"),
             fg=ACCENT, bg=BG_CARD,
             anchor="w").pack(fill="x", pady=(4, 0))
    tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)
    body = tk.Frame(inner, bg=BG_CARD)
    body.pack(fill="both", expand=True, padx=8, pady=4)
    return body

def cyber_button(parent, text, color, command, width=22):
    """Flat button with left-side accent bar."""
    frame = tk.Frame(parent, bg=color, width=3)
    frame.pack(fill="x", pady=2)
    frame.pack_propagate(False)
    btn = tk.Button(
        frame, text=text,
        font=("Courier", 10, "bold"),
        bg=BG_ROW, fg=color,
        relief="flat", bd=0,
        activebackground=BG_PANEL,
        activeforeground=TEXT_PRI,
        cursor="hand2",
        width=width,
        command=command,
        anchor="w", padx=12
    )
    btn.pack(side="right", fill="both", expand=True)
    return btn

def stat_row(parent, label, var, color):
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", pady=1)
    tk.Label(row, text=label, font=("Courier", 9),
             fg=TEXT_SEC, bg=BG_CARD, width=10, anchor="w").pack(side="left")
    # Progress-bar-style value display
    bar_bg = tk.Frame(row, bg=GRID_LINE, height=18)
    bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 0))
    tk.Label(bar_bg, textvariable=var,
             font=("Courier", 9, "bold"),
             fg=color, bg=GRID_LINE,
             anchor="e", padx=6).pack(fill="both")
    return row


class FallDetectionGUI:
    def __init__(self, root):
        self.root          = root
        self.running       = False
        self.paused        = False
        self.cap           = None
        self.alert_active  = False
        self.fps           = 0
        self.frame_count   = 0
        self.last_fps_t    = time.time()
        self.current_mode  = None
        self.stat_counts   = {"falls": 0, "walks": 0, "sits": 0, "frames": 0}
        self.blink_state   = False

        self._load_models()
        self._build_ui()
        self._tick_clock()

    # ── Model loading ─────────────────────────────────────────

    def _load_models(self):
        if Path(ONE_STEP_MODEL).exists():
            self.yolo = YOLO(ONE_STEP_MODEL)
        else:
            self.yolo = YOLO("yolov8n.pt")

        self.clf = None
        if Path(CLASSIFIER_PATH).exists():
            self.clf = build_classifier()
            self.clf.load_state_dict(
                torch.load(CLASSIFIER_PATH, map_location=DEVICE))
            self.clf.eval()

    # ── UI build ──────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("FALL DETECTION SYSTEM  //  COS30018")
        self.root.geometry("1280x780")
        self.root.configure(bg=BG_DEEP)
        self.root.resizable(True, True)

        self._build_topbar()

        body = tk.Frame(self.root, bg=BG_DEEP)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self._build_display(body)
        self._build_sidebar(body)
        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, pady=0)
        bar.pack(fill="x")

        # Top accent line
        tk.Frame(bar, bg=ACCENT, height=2).pack(fill="x")

        inner = tk.Frame(bar, bg=BG_PANEL, pady=8)
        inner.pack(fill="x", padx=12)

        # Left — title block
        left = tk.Frame(inner, bg=BG_PANEL)
        left.pack(side="left")

        tk.Label(left,
                 text="[ FALL DETECTION SYSTEM ]",
                 font=("Courier", 16, "bold"),
                 fg=ACCENT, bg=BG_PANEL).pack(side="left")
        tk.Label(left,
                 text="  //  COS30018 Intelligent Systems",
                 font=("Courier", 10),
                 fg=MUTED, bg=BG_PANEL).pack(side="left")

        # Right — system info
        right = tk.Frame(inner, bg=BG_PANEL)
        right.pack(side="right")

        dev_str = "GPU ONLINE" if torch.cuda.is_available() else "CPU MODE"
        dev_col = SUCCESS if torch.cuda.is_available() else WARNING
        tk.Label(right, text=f"DEVICE: {dev_str}",
                 font=("Courier", 9, "bold"),
                 fg=dev_col, bg=BG_PANEL).pack(side="right", padx=(12, 0))

        self.clock_var = tk.StringVar(value="00:00:00")
        tk.Label(right, textvariable=self.clock_var,
                 font=("Courier", 10, "bold"),
                 fg=TEXT_SEC, bg=BG_PANEL).pack(side="right", padx=(12, 0))

        tk.Label(right, text="SYS:",
                 font=("Courier", 9),
                 fg=MUTED, bg=BG_PANEL).pack(side="right")

    def _build_display(self, parent):
        left = tk.Frame(parent, bg=BG_DEEP)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Alert banner
        self.alert_banner = tk.Label(
            left,
            text="",
            font=("Courier", 13, "bold"),
            fg=BG_DEEP, bg=BG_DEEP,
            height=2
        )
        self.alert_banner.pack(fill="x")

        # Video canvas with border
        canvas_frame = tk.Frame(left, bg=BORDER, padx=1, pady=1)
        canvas_frame.pack(fill="both", expand=True)

        canvas_inner = tk.Frame(canvas_frame, bg=BG_DEEP)
        canvas_inner.pack(fill="both", expand=True)

        self.canvas = tk.Label(
            canvas_inner,
            bg=BG_DEEP,
            text="SELECT INPUT SOURCE  >>",
            font=("Courier", 14, "bold"),
            fg=MUTED
        )
        self.canvas.pack(fill="both", expand=True)

        # Bottom info bar under canvas
        info_bar = tk.Frame(left, bg=BG_PANEL, pady=3)
        info_bar.pack(fill="x")
        tk.Frame(info_bar, bg=BORDER, height=1).pack(fill="x")
        ib = tk.Frame(info_bar, bg=BG_PANEL)
        ib.pack(fill="x", padx=8, pady=2)

        self.mode_var = tk.StringVar(value="MODE: STANDBY")
        tk.Label(ib, textvariable=self.mode_var,
                 font=("Courier", 9, "bold"),
                 fg=ACCENT, bg=BG_PANEL).pack(side="left")

        self.fps_var = tk.StringVar(value="FPS: --")
        tk.Label(ib, textvariable=self.fps_var,
                 font=("Courier", 9, "bold"),
                 fg=ACCENT2, bg=BG_PANEL).pack(side="right")

        self.det_var = tk.StringVar(value="DETECTIONS: 0")
        tk.Label(ib, textvariable=self.det_var,
                 font=("Courier", 9),
                 fg=TEXT_SEC, bg=BG_PANEL).pack(side="right", padx=16)

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG_PANEL, width=268)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        # Top accent
        tk.Frame(side, bg=ACCENT, height=2).pack(fill="x")

        tk.Label(side, text="  CONTROL PANEL",
                 font=("Courier", 10, "bold"),
                 fg=ACCENT, bg=BG_PANEL,
                 anchor="w").pack(fill="x", pady=(6, 4))

        self._build_input_section(side)
        self._build_mode_section(side)
        self._build_conf_section(side)
        self._build_stats_section(side)
        self._build_log_section(side)

    def _build_input_section(self, parent):
        body = make_section(parent, ">> INPUT SOURCE")

        cyber_button(body, "[ LOAD IMAGE  ]", ACCENT,   self.load_image)
        cyber_button(body, "[ LOAD VIDEO  ]", SUCCESS,  self.load_video)
        cyber_button(body, "[ LIVE WEBCAM ]", ACCENT2,  self.start_webcam)

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=4)

        btn_row = tk.Frame(body, bg=BG_CARD)
        btn_row.pack(fill="x")

        self.pause_btn = tk.Button(
            btn_row, text="PAUSE",
            font=("Courier", 8, "bold"),
            bg=BG_ROW, fg=WARNING,
            relief="flat", bd=0,
            activebackground=BG_PANEL,
            cursor="hand2", state="disabled",
            padx=8, pady=3,
            command=self.toggle_pause
        )
        self.pause_btn.pack(side="left", padx=(0, 2), fill="x", expand=True)

        self.stop_btn = tk.Button(
            btn_row, text="STOP",
            font=("Courier", 8, "bold"),
            bg=BG_ROW, fg=DANGER,
            relief="flat", bd=0,
            activebackground=BG_PANEL,
            cursor="hand2", state="disabled",
            padx=8, pady=3,
            command=self.stop
        )
        self.stop_btn.pack(side="left", padx=(2, 0), fill="x", expand=True)

        tk.Button(body, text="CLEAR ALERT",
                  font=("Courier", 8, "bold"),
                  bg=BG_ROW, fg=MUTED,
                  relief="flat", bd=0,
                  activebackground=BG_PANEL,
                  cursor="hand2", pady=3,
                  command=self.clear_alert
                  ).pack(fill="x", pady=(4, 0))

    def _build_mode_section(self, parent):
        body = make_section(parent, ">> DETECTION MODE")
        self.det_mode = tk.StringVar(value="one_step")
        row = tk.Frame(body, bg=BG_CARD)
        row.pack(fill="x")
        for txt, val in [("ONE-STEP", "one_step"), ("TWO-STEP", "two_step")]:
            tk.Radiobutton(
                row, text=txt, variable=self.det_mode, value=val,
                font=("Courier", 9, "bold"),
                fg=ACCENT, bg=BG_CARD,
                selectcolor=BG_DEEP,
                activebackground=BG_CARD,
                activeforeground=TEXT_PRI,
                indicatoron=True
            ).pack(side="left", padx=6)

    def _build_conf_section(self, parent):
        body = make_section(parent, ">> CONFIDENCE THRESHOLD")
        self.conf_var   = tk.DoubleVar(value=0.40)
        self.conf_label = tk.Label(
            body,
            text="CONF: 0.40",
            font=("Courier", 11, "bold"),
            fg=ACCENT2, bg=BG_CARD
        )
        self.conf_label.pack()

        def _update_conf(v):
            self.conf_label.config(text=f"CONF: {float(v):.2f}")

        tk.Scale(
            body,
            from_=0.10, to=0.90, resolution=0.05,
            orient="horizontal",
            variable=self.conf_var,
            bg=BG_CARD, fg=ACCENT,
            troughcolor=BG_DEEP,
            highlightthickness=0,
            showvalue=False,
            command=_update_conf
        ).pack(fill="x", padx=4)

    def _build_stats_section(self, parent):
        body = make_section(parent, ">> DETECTION STATS")
        self.stat_vars = {}
        rows = [
            ("FALL",   "falls",  DANGER),
            ("WALK",   "walks",  SUCCESS),
            ("SIT",    "sits",   ACCENT),
            ("FRAMES", "frames", TEXT_SEC),
        ]
        for label, key, color in rows:
            var = tk.StringVar(value="0")
            stat_row(body, label, var, color)
            self.stat_vars[key] = var

        tk.Button(
            body, text="RESET STATS",
            font=("Courier", 8),
            bg=BG_DEEP, fg=MUTED,
            relief="flat", bd=0,
            cursor="hand2", pady=2,
            command=self._reset_stats
        ).pack(fill="x", pady=(4, 0))

    def _build_log_section(self, parent):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="both", expand=True, padx=8, pady=4)
        inner = tk.Frame(outer, bg=BG_CARD)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text="  >> EVENT LOG",
                 font=("Courier", 9, "bold"),
                 fg=ACCENT, bg=BG_CARD, anchor="w").pack(fill="x", pady=(4, 0))
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=6, pady=2)

        self.log_box = tk.Text(
            inner,
            bg=BG_DEEP, fg=TEXT_SEC,
            font=("Courier", 8),
            state="disabled",
            relief="flat",
            wrap="word",
            insertbackground=ACCENT
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        # Tag colours for log
        self.log_box.tag_config("fall",    foreground=DANGER)
        self.log_box.tag_config("ok",      foreground=SUCCESS)
        self.log_box.tag_config("info",    foreground=ACCENT)
        self.log_box.tag_config("warning", foreground=WARNING)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, pady=0)
        bar.pack(fill="x", side="bottom")
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x")
        inner = tk.Frame(bar, bg=BG_PANEL)
        inner.pack(fill="x", padx=10, pady=3)

        self.status_var = tk.StringVar(value="SYSTEM READY  //  SELECT INPUT SOURCE TO BEGIN")
        tk.Label(inner, textvariable=self.status_var,
                 font=("Courier", 8),
                 fg=MUTED, bg=BG_PANEL,
                 anchor="w").pack(side="left")

        model_str = "MODEL: " + ("TRAINED" if Path(ONE_STEP_MODEL).exists() else "DEFAULT (yolov8n)")
        model_col = SUCCESS if Path(ONE_STEP_MODEL).exists() else WARNING
        tk.Label(inner, text=model_str,
                 font=("Courier", 8, "bold"),
                 fg=model_col, bg=BG_PANEL).pack(side="right")

    # ── Clock tick ────────────────────────────────────────────

    def _tick_clock(self):
        self.clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # ── Helpers ───────────────────────────────────────────────

    def log(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _reset_stats(self):
        self.stat_counts = {"falls": 0, "walks": 0, "sits": 0, "frames": 0}
        for v in self.stat_vars.values():
            v.set("0")

    def clear_alert(self):
        self.alert_active = False
        self.alert_banner.config(text="", bg=BG_DEEP)
        self.log("ALERT CLEARED", "ok")

    def _set_running_state(self, running):
        s = "normal" if running else "disabled"
        self.stop_btn.config(state=s)
        self.pause_btn.config(state=s)

    def stop(self):
        self.running = False
        self.paused  = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._set_running_state(False)
        self.pause_btn.config(text="PAUSE")
        self.status_var.set("STOPPED  //  SELECT INPUT SOURCE TO BEGIN")
        self.mode_var.set("MODE: STANDBY")
        self.fps_var.set("FPS: --")
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
            self.status_var.set(f"RUNNING  //  {self.current_mode.upper()}")
            self.log("RESUMED", "ok")

    # ── Detection ─────────────────────────────────────────────

    def _detect(self, frame):
        conf = self.conf_var.get()
        mode = self.det_mode.get()
        h, w = frame.shape[:2]
        res  = self.yolo.predict(frame, conf=conf, verbose=False)
        dets = []

        if res[0].boxes is not None:
            for box in res[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if mode == "two_step" and self.clf is not None:
                    crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    t = val_tf(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(self.clf(t), dim=1)[0]
                        cls   = int(probs.argmax())
                        c     = float(probs[cls])
                else:
                    cls = int(box.cls[0])
                    c   = float(box.conf[0])

                dets.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": cls,
                    "class_name": CLASS_NAMES.get(cls, "?"),
                    "confidence": c
                })
        return dets

    def _draw_detections(self, frame, dets):
        fall = False
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cls   = d["class"]
            name  = d["class_name"]
            color = CLASS_COLORS_BGR.get(cls, (255, 255, 255))
            label = f"{name}  {d['confidence']:.2f}"

            # Outer box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents (robotic look)
            tl, sz = 8, 3
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (cx, cy), (cx + dx*tl, cy), color, sz)
                cv2.line(frame, (cx, cy), (cx, cy + dy*tl), color, sz)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame,
                          (x1, y1 - th - 10),
                          (x1 + tw + 8, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)

            if name == "FALL":
                fall = True
                self.stat_counts["falls"] += 1
            elif name == "Walk":
                self.stat_counts["walks"] += 1
            elif name == "Sit":
                self.stat_counts["sits"] += 1

        self.det_var.set(f"DETECTIONS: {len(dets)}")
        return frame, fall

    def _show_frame(self, frame, fall=False):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cw  = self.canvas.winfo_width()
        ch  = self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            fw, fh = img.size
            scale  = min(cw / fw, ch / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            img    = img.resize((nw, nh), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=imgtk, text="")
        self.canvas.image = imgtk

        # Alert banner
        if fall:
            self.alert_active = True
            self.blink_state  = not self.blink_state
            bg = DANGER if self.blink_state else "#aa2222"
            self.alert_banner.config(
                text="  !!  FALL DETECTED — ALERT  !!",
                fg=BG_DEEP, bg=bg)
            self.log("FALL DETECTED!", "fall")
        elif not self.alert_active:
            self.alert_banner.config(
                text="  OK  MONITORING — NO FALL DETECTED",
                fg=BG_DEEP, bg="#0a3d1f")

        self.fps_var.set(
            f"FPS: {self.fps}" if self.current_mode != "image" else "FPS: IMAGE")
        for k, v in self.stat_vars.items():
            v.set(str(self.stat_counts.get(k, 0)))

    # ── Mode 1: Image ─────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self.stop()
        self.current_mode = "image"
        self._reset_stats()
        self.log(f"IMAGE LOADED: {Path(path).name}", "info")
        self.status_var.set(f"IMAGE  //  {Path(path).name}")
        self.mode_var.set("MODE: IMAGE")

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image file!")
            return

        dets        = self._detect(frame)
        frame, fall = self._draw_detections(frame, dets)
        self.stat_counts["frames"] += 1
        self.root.after(100, self._show_frame, frame, fall)
        self._set_running_state(False)

        for d in dets:
            tag = "fall" if d["class_name"] == "FALL" else "ok"
            self.log(f"  DET: {d['class_name']}  conf={d['confidence']:.2f}", tag)
        if not dets:
            self.log("  NO DETECTIONS", "warning")

    # ── Mode 2: Video ─────────────────────────────────────────

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self.stop()
        self._reset_stats()
        self.current_mode = "video"
        self.running      = True
        self.paused       = False

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file!")
            return

        total  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_s  = self.cap.get(cv2.CAP_PROP_FPS)
        self._set_running_state(True)
        self.mode_var.set("MODE: VIDEO")
        self.status_var.set(f"VIDEO  //  {Path(path).name}  [{total} frames  {fps_s:.0f}fps]")
        self.log(f"VIDEO: {Path(path).name}  {total}fr  {fps_s:.0f}fps", "info")
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
            dets        = self._detect(frame)
            frame, fall = self._draw_detections(frame, dets)
            self.stat_counts["frames"] += 1
            self.frame_count += 1
            now = time.time()
            if now - self.last_fps_t >= 1.0:
                self.fps        = self.frame_count
                self.frame_count = 0
                self.last_fps_t  = now
            self.root.after(0, self._show_frame, frame, fall)
            time.sleep(0.01)

    # ── Mode 3: Webcam ────────────────────────────────────────

    def start_webcam(self):
        self.stop()
        self._reset_stats()
        self.current_mode = "webcam"
        self.running      = True
        self.paused       = False

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error",
                "Webcam not found!\nMake sure your camera is connected.")
            self.running = False
            return

        self._set_running_state(True)
        self.mode_var.set("MODE: WEBCAM LIVE")
        self.status_var.set("LIVE  //  WEBCAM STREAM ACTIVE")
        self.log("WEBCAM STREAM STARTED", "ok")
        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def _webcam_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.log("WEBCAM SIGNAL LOST", "warning")
                break
            dets        = self._detect(frame)
            frame, fall = self._draw_detections(frame, dets)
            self.stat_counts["frames"] += 1
            self.frame_count += 1
            now = time.time()
            if now - self.last_fps_t >= 1.0:
                self.fps         = self.frame_count
                self.frame_count = 0
                self.last_fps_t  = now
            self.root.after(0, self._show_frame, frame, fall)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = FallDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app.stop(), root.destroy()))
    root.mainloop()