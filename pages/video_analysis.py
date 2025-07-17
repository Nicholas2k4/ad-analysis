import streamlit as st
import requests, pathlib, hashlib, urllib.parse as ul
import base64, os, json
from google import genai
from google.genai import types
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
from pathlib import Path
from collections import defaultdict

###############################################################################
# Load Gemini API key (add GEMINI_API_KEY in Streamlit Secrets or .env)       #
###############################################################################

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

###############################################################################
# Helper: cached video downloader                                            #
###############################################################################


@st.cache_data(show_spinner=False)
def download_video_file(url: str) -> pathlib.Path:
    """Download URL → local ./ads/<md5>.mp4  (cached by URL)."""
    dst = pathlib.Path("ads")
    dst.mkdir(exist_ok=True)

    fname = hashlib.md5(url.encode()).hexdigest() + ".mp4"
    path = dst / fname

    # Already exists → skip network
    if path.exists():
        return path

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2**16):
                f.write(chunk)
    return path


def get_video_duration(path: Path) -> float:
    """
    Return video length in **seconds** for the file at *path*.
    Returns 0.0 if the header cannot be read.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0

    # total number of frames
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # frames-per-second
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    return frames / fps if fps else 0.0


@st.cache_data(show_spinner=False)
def extract_frames(path: pathlib.Path, seconds: list[int]) -> dict[int, np.ndarray]:
    """
    Return {second -> RGB NumPy image} for the requested second marks.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = {}
    for s in seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ok, frame = cap.read()
        if ok:
            out[s] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


if "seg_result" not in st.session_state:
    st.session_state.seg_result = None
if "detect_img" not in st.session_state:
    st.session_state.detect_img = None
if "detect_boxes" not in st.session_state:
    st.session_state.detect_boxes = None

###############################################################################
# UI START                                                                    #
###############################################################################

st.set_page_config(page_title="Video Analysis", page_icon="🎞️", layout="wide")

st.title("🎞️ Video Analysis")
st.divider()

# ---------------------------------------------------------------------------
# 1) Ambil video dari query-param ATAU dari input manual
# ---------------------------------------------------------------------------

params = st.query_params
raw_url = params.get("video", [""][0]) if params else ""
video_path: pathlib.Path | None = None  # ← inisialisasi

if raw_url:  # ⬅️  kasus lama: ada ?video=
    video_url = ul.unquote_plus(raw_url)

else:  # ⬅️  kasus baru: minta input manual
    st.info("Upload file video (.mp4) terlebih dulu.")

    # url_inp  = st.text_input("Video URL (opsional)")
    file_up = st.file_uploader("Upload video (.mp4)", type=["mp4"])

    if file_up is None:
        st.stop()  # tunggu sampai salah satu diisi

    if file_up is not None:  # ➜ pakai file yang di-upload
        upload_dir = pathlib.Path("ads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        tmp = pathlib.Path("ads") / f"upload_{datetime.now().timestamp():.0f}.mp4"
        tmp.write_bytes(file_up.read())
        video_path = tmp

# ---------------------------------------------------------------------------
# 2) Jika datang dari query-param, lakukan download seperti biasa
# ---------------------------------------------------------------------------

if video_path is None:  # artinya tadi dari ?video=
    with st.spinner("Downloading video …"):
        try:
            video_path = download_video_file(video_url)
        except Exception as e:
            st.error(f"Failed to download video: {e}")
            st.stop()

col1, col2 = st.columns([3, 7])
with col1:
    st.video(str(video_path))

with col2:
    st.success("Video ready for analysis ✨")

###############################################################################
# 3) Push the video to Gemini 2.5 Flash                                       #
###############################################################################

PROMPT = """
SYSTEM:
You are a precise video-analysis assistant. Reply ONLY with valid JSON.

USER TASK:
1. Analyze the attached promo-video (Meta Ads). Produce per-second segmentation AND mark every visual scene-change for each segmentation. Each segmentation should be used for competitor analysis so that we can compare the effectiveness of our ads. Contextualize the segmentation based on the promotional video in terms of marketing and videography parts. A segmentation may contain one or more visual scene-change. **CONVERT INTO ABSOLUTE SECONDS**
2. Give the transcript of the video.

OUTPUT FORMAT (JSON, nothing else!):
{
  "segmentations": [
    {
      "start_second": <int>,        # inclusive
      "end_second":   <int>,        # inclusive
      "label":        "<short description>",
      "scene_changes": [<int>, …]   # list of absolute seconds where a new scene starts within this segment (empty list if none)
    },
    …
  ],
  "transcript" : "<video transcript>" in string
}

CONSTRAINTS:
1. Ignore audio; rely on visuals only.
2. start_second, end_second, and scene_changes are ABSOLUTE from video start (e.g. 01:10 ⇒ 70 seconds, not 110).
3. Segments must cover the whole runtime without gaps or overlaps.
4. Keep every label concise in Indonesian words and.
5. scene_changes must be sorted ascending and lie between start_second-end_second inclusive.
6. Return EXACTLY the JSON object above – no markdown, no comments, no extra keys.
"""


def call_gemini(video_path: pathlib.Path, prompt: str = PROMPT):
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found.")
        return None
    client = genai.Client()
    video_bytes = open(video_path, "rb").read()

    print(f"VIDEO PATH = {video_path}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
                    ),
                    types.Part(text=PROMPT),
                ]
            ),
            config={"response_mime_type": "application/json"},
        )
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

    # --- tarik hasil (Gemini selalu membalas Part berisi text) ------------
    raw_text = response.text
    try:
        return json.loads(raw_text)  # sudah JSON sesuai instruksi
    except json.JSONDecodeError:
        st.warning("Gemini tidak mengembalikan JSON valid — menampilkan string mentah.")
        return raw_text


# -------------------------- Action button -----------------------------------
with col2:
    if st.button("Segment Video with Gemini"):
        with st.spinner("Gemini is working …"):
            st.session_state.seg_result = call_gemini(video_path)
            print(st.session_state.seg_result)
            st.toast("Done!", icon="✅")

    # ── tampilkan tabel ringkas + (opsional) JSON mentah ────────────────────
    if st.session_state.seg_result and isinstance(st.session_state.seg_result, dict):
        seg_df = pd.DataFrame(st.session_state.seg_result["segmentations"])
        seg_df["duration (s)"] = seg_df["end_second"] - seg_df["start_second"] + 1
        transcript = st.session_state.seg_result["transcript"]

        # tabel lebih mudah dibaca daripada JSON mentah
        # st.dataframe(
        #     seg_df[["start_second", "end_second", "duration (s)", "segmentation"]],
        #     hide_index=True,
        #     use_container_width=True,
        # )

        # ───────────────────────── Visual timeline per-segmen ─────────────────────────
        st.markdown("### 🖼️  Visual Timeline")

        N_COL = 6  # berapa thumbnail per baris
        for _, row in seg_df.iterrows():
            # judul segmen sekali saja
            st.markdown(
                f"**{row['start_second']}s – {row['end_second']}s · {row['label']}**"
            )

            # ambil semua frame di rentang segmen
            print(row['scene_changes'])
            if row["scene_changes"] == []:
                row["scene_changes"] = [row["start_second"]]
            frames_this_seg = extract_frames(video_path, row["scene_changes"])

            # grid thumbnail
            cols = st.columns(N_COL)
            for i, s in enumerate(row["scene_changes"]):
                if s in frames_this_seg:
                    cols[i % N_COL].image(frames_this_seg[s], width=150)

            st.divider()  # pemisah antar-segmen
            
        st.markdown("### 📝  Transcript")
        st.write(transcript)

# ───────────────────── Object-detection single frame ──────────────────────
st.markdown("## 🎯 Object detection at specific second")
st.divider()

if video_path:
    end_second = int(get_video_duration(video_path))
    print(end_second)
    seconds    = list(range(0, end_second + 1))
    frames     = extract_frames(video_path, seconds)

    N_COL = 10                     

    for row_start in range(0, end_second + 1, N_COL):
        row_end = min(row_start + N_COL - 1, end_second)

        st.markdown(f"**{row_start}-{row_end}s**")

        cols = st.columns(N_COL)
        for idx, sec in enumerate(range(row_start, row_end + 1)):
            if sec in frames:
                cols[idx].image(frames[sec], width=95, caption=f"{sec}s")
            else:
                cols[idx].empty()          

detect_sec = st.number_input(
    "Second to analyse",
    min_value=0,
    step=1,
    value=0,
    help="Masukkan detik video yang ingin dideteksi obyeknya",
)

if st.button("Detect objects at that second", key="detect-btn"):
    with st.spinner("Gemini is working..."):
        sec = int(detect_sec)
        frame_dict = extract_frames(video_path, [sec])

        if sec not in frame_dict:
            st.error(f"Tidak ditemukan frame di detik {sec}.")
        else:
            frame_np = frame_dict[sec]
            h, w = frame_np.shape[:2]
            img_pil = Image.fromarray(frame_np.copy())
            draw = ImageDraw.Draw(img_pil)

            detect_prompt = (
                "Give segmentation masks of the image, with no more than 10 items. "
                'Output ONLY A JSON list where each entry contains the 2D bounding box in "box_2d" '
                'and a text label in "label" to describe the object in Indonesian language. The coordinates in "box_2d" should be '
                "[ymin, xmin, ymax, xmax] and normalized to 1000."
            )
            print(detect_prompt)

            try:
                client = genai.Client()
                print("Generating response...")
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[img_pil, detect_prompt],
                    config={"response_mime_type": "application/json"},
                )
                print("Done generating response.")

                raw = resp.text.strip()
                if raw.startswith("```"):  # bersihkan ```json … ```
                    raw = raw.split("```")[1].strip()
                boxes = json.loads(raw)

                print(boxes)
                labels = defaultdict(int)

                # --- gambarkan bounding-box di atas gambar ------------------------------
                for item in boxes:
                    label = item.get("label", "")
                    labels[label] += 1
                    ymin, xmin, ymax, xmax = item["box_2d"]

                    # Skala → ukuran asli frame
                    h, w = frame_np.shape[:2]
                    print(f"Height = {h}, Width = {w}")
                    scale_x = w / 1000.0
                    scale_y = h / 1000.0

                    x1 = int(xmin * scale_x)
                    x2 = int(xmax * scale_x)
                    y1 = int(ymin * scale_y)
                    y2 = int(ymax * scale_y)

                    # Clamp agar tidak keluar gambar
                    x1, x2 = max(0, x1), min(w - 1, x2)
                    y1, y2 = max(0, y1), min(h - 1, y2)

                    print(f"{x1}, {x2}, {y1}, {y2} = {label}")

                    # Draw rectangle & label
                    draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
                    draw.text((x1 + 4, y1 + 4), label, fill="black", stroke_fill="white", stroke_width=1)

                st.session_state.detect_img = img_pil
                st.session_state.detect_boxes = boxes
                st.session_state.labels = labels

            except Exception as e:
                st.error(f"Gemini detection error: {e}")

    if st.session_state.detect_img is not None:
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.image(
                st.session_state.detect_img,
                caption=f"Frame @ {sec}s (dengan bounding-box)",
                use_container_width=False,
            )
        with col2:
            for key, value in st.session_state.labels.items():
                st.write(f"{value}x   **{key}**")
