# CV-TWD

Generate workout-poster pose drawings from your webcam. Strike a pose, press a
key, and get a clean black-outline figure (white body, grey floor shadow) like a
hand-drawn exercise illustration. Capture a start and an end position and save
them together as an editable **SVG** plus a **PNG**.

## Setup

Needs Python 3.9 – 3.12 (MediaPipe does not support newer versions yet).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install mediapipe opencv-python numpy shapely
```

## Use

```bash
python pose_drawing.py                      # live webcam
python pose_drawing.py --image a.jpg b.jpg  # draw from photos instead
```

Live keys: `SPACE` capture a pose (5 s timer) · `ENTER` finish · `R` reset · `Q` quit.
On the review screen: `S` male/female style · `F` flip depth order · `X` mirror ·
`L` layout · `1`/`2` retake a pose · `ENTER` save to `output/`.

Tips: stand far enough back that your whole body is in view; for side-on poses
(plank, lunge) turn sideways to the camera. Run `python pose_drawing.py -h` for
all options. Older experiments live in [docs/](docs/).

### Work in Action

[Demo video](https://github.com/user-attachments/assets/277e7a88-be7b-4671-ab1b-24eee6d22f15)

For any queries or support, contact 72pxperinch@gmail.com.
