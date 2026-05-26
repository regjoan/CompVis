# Comparative Analysis: Template Matching vs. Optical Flow for Human Tracking

## Regina Joan Medea Jati Laksono

## 24/532850/PA/22546

## CSB

---

# 1. Detection Stage — How the Human is Detected

Before tracking begins, the system first initializes a bounding box around the human using a **two-stage detector** (`Detector` class).

## Stage 1 — Background Subtraction (MOG2)

The system identifies all **moving pixels** in the frame.

This helps eliminate:

- Static humans
- Posters
- Mannequins
- Background objects falsely detected as humans

## Stage 2 — HOG Detection on Motion ROIs

The HOG pedestrian detector only runs inside the **top 3 largest moving regions**.

## Why This Matters

Using HOG alone becomes unreliable after frame 200+ because:

- The person becomes smaller
- Partial occlusion occurs
- The background becomes cluttered

By restricting HOG to regions with actual motion:

- False positives are reduced
- Computation becomes faster
- Detection becomes more stable

Additionally, the system enforces a human-like aspect ratio:

```text
height ≈ 2 × width
```

This prevents random rectangular regions from being classified as humans.

---

# 2. Method 1 — Template Matching (Normalized Cross-Correlation)

## How It Works in the Code

## Initialization

- The detected human region is cropped
- It is resized into a normalized template
- Maximum dimension: **100 px**

## Tracking Process

For each new frame:

1. A search window is created around the previous center
   - Size: **1.5× object size**
2. The search patch is resized
3. `cv2.matchTemplate()` with `TM_CCOEFF_NORMED` is applied
4. The highest correlation score is selected

## Template Update

If the confidence score exceeds **0.2**:

- The center position is updated
- The template is slowly adapted:

```text
new_template =
0.95(old_template) + 0.05(new_appearance)
```

This allows minor adaptation to:

- Lighting changes
- Small appearance variations

while still preserving the original target appearance.

---

## Theoretical Characteristics

Template Matching assumes the target is:

- Rigid
- Textured
- Visually consistent

It works like a:

> nearest-neighbor search in appearance space

The tracker has:

- No motion understanding
- No prediction model
- No semantic understanding of humans

It simply searches:

> "Which region looks most similar to the template?"

---

# 3. Method 2 — Optical Flow (Lucas-Kanade + Shi-Tomasi Features)

## How It Works in the Code

## Initialization

The system detects:

- 80–100 Shi-Tomasi corner features
  inside the initial bounding box.

## Tracking Process

The tracker uses:

- Pyramidal Lucas-Kanade Optical Flow
- `calcOpticalFlowPyrLK()`

to estimate where each feature moves between frames.

Features are discarded when:

- Flow estimation fails
- `status = 0`

---

## Motion Estimation

Instead of averaging all feature movements:

```text
mean displacement
```

the system uses:

```text
median displacement
```

This improves robustness because:

- Outlier features are ignored
- Background drift has less influence

---

## Update Process

- The bounding box center shifts using the median displacement
- Bounding box size remains fixed

If the number of tracked features falls below **25**:

- New Shi-Tomasi corners are re-detected

---

## Theoretical Characteristics

Optical Flow assumes:

- Brightness constancy
- Small inter-frame motion

Unlike Template Matching, it does **not** track the whole object appearance.

Instead, it tracks:

- Local corners
- Texture patterns
- Edge movements

This makes it more robust to:

- Human deformation
- Pose changes
- Partial occlusion

because different features can move independently.

---

# 4. Side-by-Side Comparison

| Aspect                | Template Matching                     | Optical Flow                 |
| --------------------- | ------------------------------------- | ---------------------------- |
| **Core assumption**   | Appearance similarity                 | Motion continuity            |
| **Tracks**            | Entire object appearance              | Local corner features        |
| **Box stability**     | Very smooth and stable                | Slight jitter                |
| **Speed**             | Faster (~5–15 ms/frame)               | Slower (~15–40 ms/frame)     |
| **Pose deformation**  | Poor                                  | Good                         |
| **Partial occlusion** | Poor                                  | Moderate                     |
| **Scale changes**     | Poor                                  | Poor                         |
| **Drift behavior**    | Silent drift                          | Visible feature scattering   |
| **Failure mode**      | Locks onto similar background texture | Features drift to background |

---

# 5. Graph Analysis — Interpreting the Metrics

## Graph 1 — Trajectory (X-Y Path)

This graph compares the movement paths produced by both trackers.

### Interpretation

- Strong overlap → both trackers follow the same person
- Divergence → tracking inconsistency

### Typical Behavior

- Optical Flow → more jitter/wiggle
- Template Matching → smoother trajectory but potentially delayed

---

## Graph 2 — Template Confidence (NCC Score)

This acts as the:

> "health bar" of Template Matching

### Interpretation

| NCC Score       | Meaning            |
| --------------- | ------------------ |
| 0.6 – 0.9       | Reliable tracking  |
| Gradual decline | Appearance changes |
| < 0.2           | Tracker failure    |

When confidence drops too low:

- The tracker starts correlating with:
  - walls
  - shadows
  - floor textures

instead of the human.

---

## Graph 3 — Optical Flow Feature Count

This is the:

> "health bar" of Optical Flow

### Interpretation

| Feature Count   | Meaning                           |
| --------------- | --------------------------------- |
| 60 – 100        | Stable tracking                   |
| Gradual decline | Features drifting or disappearing |
| < 5             | Tracker collapse                  |

Once feature count becomes extremely low:

- The tracker loses motion information
- Tracking returns `None`

---

## Graph 4 — Divergence Between Trackers

This measures the pixel distance between tracker centers.

## Why It Is Important

This graph provides:

> cross-validation between trackers

### Interpretation

| Distance | Meaning                   |
| -------- | ------------------------- |
| < 20 px  | Strong agreement          |
| 20–40 px | Moderate disagreement     |
| > 40 px  | One tracker likely failed |

Large divergence indicates:

- Drift
- Occlusion
- Tracker collapse

This becomes the trigger for:

> periodic re-detection

---

# 6. Why Periodic Re-Detection Is Necessary

Both trackers are:

> causal trackers

They only know:

- Current frame
- Previous state

They have:

- No long-term memory
- No semantic understanding of "human"

---

## Drift in Template Matching

The adaptive update gradually contaminates the template:

```text
5% background added each update
```

Over time:

- Background pixels poison the template
- Drift becomes unavoidable

---

## Drift in Optical Flow

Feature points slowly migrate toward:

- High-contrast edges
- Background textures
- Static corners

Even median filtering cannot completely prevent this.

---

## Solution — Hard Re-Initialization

The detector runs every:

> 25 processed frames

This:

- Resets tracker position
- Corrects accumulated drift
- Re-establishes ground truth

This creates a primitive form of:

> tracking-by-detection

---

# 7. Practical Usage — Which Method Works Better?

| Scenario                            | Better Method     | Reason                       |
| ----------------------------------- | ----------------- | ---------------------------- |
| Straight walking, stable appearance | Template Matching | Fast and smooth              |
| Pose changes / bending              | Optical Flow      | Features move independently  |
| Partial occlusion                   | Optical Flow      | Some features remain visible |
| Embedded systems / Raspberry Pi     | Template Matching | Lower CPU usage              |
| Long-term tracking                  | Neither           | Both drift eventually        |
| Fast motion / motion blur           | Neither           | Both become unstable         |

---

# 8. Fundamental Limitations

# Template Matching Limitations

## 1. No Motion Model

The tracker has no understanding of smooth motion.

If a shadow matches better:

- it may jump to the shadow.

---

## 2. No Deformation Model

Different human poses produce drastically different appearances.

Examples:

- Standing
- Sitting
- Turning

NCC correlation becomes weak.

---

## 3. Scale Blindness

If the person moves closer:

- template size remains fixed
- matching accuracy collapses

---

# Optical Flow Limitations

## 1. Aperture Problem

Flat regions with little texture:

- cannot produce reliable flow

Examples:

- White shirts
- Blank walls

---

## 2. Feature Migration

Features drift toward:

- Door frames
- Floor tiles
- High-contrast background corners

because they are easier to track.

---

## 3. No Appearance Verification

The tracker only follows:

> moving dots

It cannot verify:

- whether those dots still belong to a human.

---

# 9. Why Modern Trackers Replaced Classical Methods

This comparison demonstrates why modern systems rely on deep learning.

| Classical Limitation      | Modern Solution  |
| ------------------------- | ---------------- |
| Template drift            | Siamese Networks |
| Feature drift             | DeepSORT         |
| No semantic understanding | YOLO + ByteTrack |
| Scale variation           | SiamRPN          |

Modern systems combine:

- Appearance modeling
- Motion prediction
- Human detection
- Re-identification

into one unified framework.

---

# 10. Final Conclusion

## Template Matching

Template Matching treats the human as a:

> static image patch

### Strengths

- Fast
- Deterministic
- Smooth trajectories

### Weaknesses

Sensitive to:

- pose changes
- occlusion
- scale variation

Its NCC score provides a clear indication of failure.

---

## Optical Flow

Optical Flow treats the human as:

> a collection of moving feature points

### Strengths

More robust to:

- deformation
- partial occlusion

### Weaknesses

- Computationally heavier
- Noisier trajectories
- Feature drift over time

Its feature count provides a strong failure indicator.

---

# Overall Insight

Together, these methods represent two classical tracking philosophies:

| Philosophy           | Method            |
| -------------------- | ----------------- |
| Appearance Constancy | Template Matching |
| Motion Continuity    | Optical Flow      |

Neither approach alone is sufficient for robust long-term human tracking.

This is why modern tracking systems combine:

- Detection
- Motion estimation
- Appearance learning
- Re-identification

inside deep learning-based architectures.
