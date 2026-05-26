"""
Regina Joan Medea Jati Laksono
24/532850/PA/22546
CVL_ASSIGNMENT04 

main.py
Auto human detection + Template vs Optical Flow.
Stable bounding boxes. Normal speed. Popup window. Press Q to quit.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
FILE_ID = "12hx17ISc6rQM4F6CLpvFkd7L12jNWdE9"
VIDEO = "tracking_video.mp4"
OUT = Path("tracking_output")
SKIP = 2              # skip frames for normal playback speed
DETECT_EVERY = 25     # re-detect human every N processed frames

# -------------------- DOWNLOAD --------------------
def download(id, out):
    if os.path.exists(out) and os.path.getsize(out) > 1024:
        return
    print("[INFO] Downloading video...")
    import requests
    s = requests.Session()
    r = s.get("https://drive.google.com/uc?export=download", params={'id': id}, stream=True)
    tok = next((v for k, v in r.cookies.items() if k.startswith('download_warning')), None)
    if tok:
        r = s.get("https://drive.google.com/uc?export=download", params={'id': id, 'confirm': tok}, stream=True)
    with open(out, "wb") as f:
        for c in r.iter_content(32768):
            if c: f.write(c)
    print("[OK] Downloaded.")

# -------------------- DETECTOR (Motion + HOG) --------------------
class Detector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.last_box = None

    def find(self, frame):
        fg = self.fgbg.apply(frame)
        _, fg = cv2.threshold(fg, 128, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_box, best_s = None, 0

        areas = sorted([(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts], reverse=True)[:3]
        for _, (mx, my, mw, mh) in areas:
            if mw < 40 or mh < 80:
                continue
            x1, y1 = max(0, mx-20), max(0, my-20)
            x2, y2 = min(frame.shape[1], mx+mw+20), min(frame.shape[0], my+mh+20)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            boxes, scores = self.hog.detectMultiScale(roi, winStride=(4,4), padding=(8,8), scale=1.05)
            if len(boxes):
                idx = np.argmax(scores)
                bx, by, bw, bh = boxes[idx]
                s = float(scores[idx][0] if hasattr(scores[idx], '__len__') else scores[idx])
                if bw > 0:
                    aspect = bh / bw
                    if aspect < 1.5:
                        bh = int(bw * 2)
                    elif aspect > 3.0:
                        bh = int(bw * 2.5)
                box = (x1+bx, y1+by, bw, bh)
                if s > best_s:
                    best_s, best_box = s, box

        if best_box:
            self.last_box = best_box
            return best_box
        return self.last_box

# -------------------- TEMPLATE TRACKER (center only) --------------------
class TemplateTracker:
    def __init__(self, frame, box):
        x, y, w, h = box
        self.size = (w, h)
        self.center = (x + w//2, y + h//2)
        scale = 100 / max(w, h)
        self.tmpl = cv2.resize(frame[y:y+h, x:x+w], None, fx=scale, fy=scale)
        self.scale = scale
        self.path = []
        self.lost = 0
        self.conf_scores = []  

    def track(self, frame):
        cx, cy = self.center
        w, h = self.size
        H, W = frame.shape[:2]
        m = int(max(w, h) * 1.5)
        x1, y1 = max(0, cx-m), max(0, cy-m)
        x2, y2 = min(W, cx+m), min(H, cy+m)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            self.lost += 1
            self.conf_scores.append(0)
            return None

        r_s = cv2.resize(region, None, fx=self.scale, fy=self.scale)
        res = cv2.matchTemplate(r_s, self.tmpl, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(res)

        nx = int(x1 + loc[0]/self.scale + w/2)
        ny = int(y1 + loc[1]/self.scale + h/2)

        if v > 0.2:
            self.center = (nx, ny)
            nx1, ny1 = nx - w//2, ny - h//2
            if 0 <= ny1 < H-h and 0 <= nx1 < W-w:
                new_t = cv2.resize(frame[ny1:ny1+h, nx1:nx1+w], None, fx=self.scale, fy=self.scale)
                self.tmpl = cv2.addWeighted(self.tmpl, 0.95, new_t, 0.05, 0)
            self.path.append((nx, ny))
            self.conf_scores.append(float(v))
            return (nx - w//2, ny - h//2, w, h)

        self.lost += 1
        self.conf_scores.append(float(v))
        return None

# -------------------- FLOW TRACKER (center only) --------------------
class FlowTracker:
    def __init__(self, frame, box):
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = box
        self.size = (w, h)
        self.center = (x + w//2, y + h//2)
        self.path = []
        self.lost = 0
        self.feature_counts = []  
        self._seed(box)

    def _seed(self, box):
        x, y, w, h = box
        mask = np.zeros_like(self.prev)
        mask[y:y+h, x:x+w] = 255
        self.pts = cv2.goodFeaturesToTrack(self.prev, 100, 0.01, 5, mask=mask)

    def track(self, frame):
        if self.pts is None or len(self.pts) < 5:
            self.lost += 1
            self.feature_counts.append(0)
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev, gray, self.pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if nxt is None:
            self.lost += 1
            self.prev = gray
            self.feature_counts.append(0)
            return None

        old = self.pts[st.flatten() == 1].reshape(-1, 2)
        new = nxt[st.flatten() == 1].reshape(-1, 2)
        if len(new) < 5:
            self.lost += 1
            self.prev = gray
            self.feature_counts.append(len(new))
            return None

        dx = np.median(new[:,0] - old[:,0])
        dy = np.median(new[:,1] - old[:,1])

        cx, cy = self.center
        cx, cy = int(cx + dx), int(cy + dy)
        self.center = (cx, cy)
        w, h = self.size

        self.path.append((cx, cy))
        self.prev = gray
        self.pts = new.reshape(-1, 1, 2)
        self.feature_counts.append(len(self.pts))

        if len(self.pts) < 25:
            self._seed((cx - w//2, cy - h//2, w, h))

        return (cx - w//2, cy - h//2, w, h)

# -------------------- DRAW --------------------
def draw(frame, box, label, color):
    v = frame.copy()
    if box is None:
        cv2.putText(v, f"{label} LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return v
    x, y, w, h = box
    cv2.rectangle(v, (x, y), (x+w, y+h), color, 2)
    cv2.putText(v, label, (x, max(20, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.circle(v, (x+w//2, y+h//2), 3, (0,255,255), -1)
    return v

def draw_traj(frame, path, color):
    v = frame.copy()
    for i in range(1, min(len(path), 60)):
        cv2.line(v, tuple(map(int, path[i-1])), tuple(map(int, path[i])), color, 2)
    return v

# -------------------- GRAPH GENERATOR --------------------
def generate_graphs(T, F, out_dir):
    """Generate 4 comparison graphs from tracking data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Template Matching vs Optical Flow Analysis", fontsize=14, fontweight='bold')

    # 1. Trajectory
    ax = axes[0, 0]
    if T.path:
        t = np.array(T.path)
        ax.plot(t[:, 0], t[:, 1], 'b-', label='Template', linewidth=2)
        ax.scatter(t[0, 0], t[0, 1], c='blue', s=80, marker='o', zorder=5)
    if F.path:
        f = np.array(F.path)
        ax.plot(f[:, 0], f[:, 1], 'r-', label='Optical Flow', linewidth=2)
        ax.scatter(f[0, 0], f[0, 1], c='red', s=80, marker='s', zorder=5)
    ax.set_title('1. Object Trajectory')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # 2. Template Confidence
    ax = axes[0, 1]
    ax.plot(T.conf_scores, 'b-', label='NCC Score', linewidth=1.5)
    ax.axhline(0.2, color='b', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_title('2. Template Matching Confidence')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Correlation Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Feature Count
    ax = axes[1, 0]
    ax.plot(F.feature_counts, 'r-', label='Tracked Points', linewidth=1.5)
    ax.axhline(5, color='r', linestyle='--', alpha=0.5, label='Min Threshold')
    ax.set_title('3. Optical Flow Feature Count')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number of Points')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Divergence
    ax = axes[1, 1]
    n = min(len(T.path), len(F.path))
    if n > 1:
        divs = [np.linalg.norm(np.array(T.path[i]) - np.array(F.path[i])) for i in range(n)]
        ax.plot(divs, 'g-', label='Pixel Distance', linewidth=1.5)
        ax.axhline(40, color='orange', linestyle='--', alpha=0.5, label='High Divergence')
        ax.set_title('4. Tracker Divergence')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Distance (pixels)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = out_dir / "analysis_graphs.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Graphs saved: {save_path}")
    return save_path

# -------------------- MAIN --------------------
def main():
    OUT.mkdir(exist_ok=True)
    download(FILE_ID, VIDEO)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("[ERROR] Cannot open video"); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))

    det = Detector()

    box = None
    frame_idx = 0
    while box is None and frame_idx < 500:
        ret, frame = cap.read()
        if not ret:
            break
        box = det.find(frame)
        frame_idx += 1

    if box is None:
        print("[ERROR] No human found"); return

    print(f"[INFO] Human locked at frame {frame_idx}: {box}")

    T = TemplateTracker(frame, box)
    F = FlowTracker(frame, box)

    writer = cv2.VideoWriter(str(OUT/"result.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps//SKIP, (w*2, h))

    print("[INFO] Press Q to quit")
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % SKIP != 0:
            continue

        # Re-detect periodically to kill drift
        if processed % DETECT_EVERY == 0:
            new_box = det.find(frame)
            if new_box:
                T = TemplateTracker(frame, new_box)
                F = FlowTracker(frame, new_box)
                print(f"[INFO] Re-detected at frame {frame_idx}")

        tb = T.track(frame)
        fb = F.track(frame)
        processed += 1

        L = draw_traj(draw(frame, tb, "Template", (255,0,0)), T.path, (255,0,0))
        R = draw_traj(draw(frame, fb, "Flow", (0,0,255)), F.path, (0,0,255))
        if fb and F.pts is not None:
            for p in F.pts.reshape(-1, 2):
                cv2.circle(R, tuple(p.astype(int)), 2, (0,255,0), -1)

        combo = np.hstack((L, R))
        cv2.putText(combo, f"Frame {frame_idx} | T-lost:{T.lost} F-lost:{F.lost}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Template (Blue) vs Optical Flow (Red) | Press Q", combo)
        writer.write(combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # -------------------- RESULTS + GRAPHS --------------------
    print(f"\n--- RESULTS ---")
    print(f"Template: {len(T.path)} frames tracked, {T.lost} lost")
    print(f"Flow:     {len(F.path)} frames tracked, {F.lost} lost")

    if len(T.path) > 5 and len(F.path) > 5:
        n = min(len(T.path), len(F.path), 20)
        div = np.mean([np.linalg.norm(np.array(T.path[i]) - np.array(F.path[i])) for i in range(n)])
        print(f"Divergence: {div:.1f}px")
        print("-> Good agreement" if div < 40 else "-> Trackers diverged significantly")

    print(f"Saved video: {OUT/'result.mp4'}")

    # GENERATE GRAPHS
    if len(T.conf_scores) > 0 or len(F.feature_counts) > 0:
        generate_graphs(T, F, OUT)
    else:
        print("[WARN] No tracking data to graph")

if __name__ == "__main__":
    main()