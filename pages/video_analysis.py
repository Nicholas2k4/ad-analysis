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
import zipfile, io



###############################################################################
# Load Gemini API key (add GEMINI_API_KEY in Streamlit Secrets)               #
###############################################################################
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # raises KeyError early if missing


###############################################################################
# Helpers                                                                     #
###############################################################################
@st.cache_data(show_spinner=False)
def download_video_file(url: str) -> pathlib.Path:
    """Download URL → local ./ads/<md5>.mp4  (cached by URL)."""
    dst = pathlib.Path("ads")
    dst.mkdir(exist_ok=True)

    fname = hashlib.md5(url.encode()).hexdigest() + ".mp4"
    path = dst / fname

    if path.exists():  # Already cached
        return path

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2**16):
                f.write(chunk)
    return path


def get_video_duration(path: Path) -> float:
    """Return video length in seconds. Returns 0.0 on failure."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return frames / fps if fps else 0.0


@st.cache_data(show_spinner=False)
def extract_frames(path: pathlib.Path, seconds: list[int]) -> dict[int, np.ndarray]:
    """Return {second -> RGB NumPy image} for the requested second marks."""
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

def build_timeline_zip(seg_df: pd.DataFrame, video_path: pathlib.Path) -> io.BytesIO:
    """
    Return BytesIO berisi *.zip*:
    Visual Timeline/
      └─ 00m00s‑00m09s Label A/
         ├─ 000.png  (scene‑change #1)
         ├─ 004.png  (scene‑change #2)
         └─ …
      └─ 00m10s‑00m19s Label B/
         └─ …
    """
    zbuffer = io.BytesIO()
    with zipfile.ZipFile(zbuffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        root = "Visual Timeline/"
        for _, row in seg_df.iterrows():
            s, e  = int(row["start_second"]), int(row["end_second"])
            lbl   = row["label"]
            folder = f"{root}{s:02d}s-{e:02d}s {lbl}/"
            scenes = row["scene_changes"] if row["scene_changes"] else [s]

            # ambil gambar sekali saja
            frames = extract_frames(video_path, scenes)
            for sc in scenes:
                if sc not in frames: continue
                buf = io.BytesIO()
                Image.fromarray(frames[sc]).save(buf, format="PNG")
                buf.seek(0)
                zf.writestr(f"{folder}{sc:03d}.png", buf.read())
    zbuffer.seek(0)
    return zbuffer



###############################################################################
# Session State                                                               #
###############################################################################
if "seg_result" not in st.session_state:
    st.session_state.seg_result = None
if "detect_img" not in st.session_state:
    st.session_state.detect_img = None
if "detect_boxes" not in st.session_state:
    st.session_state.detect_boxes = None
if "labels" not in st.session_state:
    st.session_state.labels = defaultdict(int)


###############################################################################
# UI CONFIG (responsive knobs)                                                #
###############################################################################
st.set_page_config(page_title="Video Analysis", page_icon="🎞️", layout="wide")

# --- detect ?mobile= param ---------------------------------------------------
params = st.query_params
_mobile_param_val = ""
if params:
    _p = params.get("mobile")
    if isinstance(_p, list):
        _mobile_param_val = (_p[0] if _p else "")
    elif isinstance(_p, str):
        _mobile_param_val = _p

_force_mobile = str(_mobile_param_val).lower() in ("1", "true", "yes")

# --- sidebar toggle to override ---------------------------------------------
mobile_layout = st.sidebar.toggle(
    "📱 Mobile layout",
    value=_force_mobile,
    help="Aktifkan untuk tampilan layar kecil / mobile.",
)

# --- responsive constants ----------------------------------------------------
if mobile_layout:
    HEADER_SPLIT = None     # stacked
    TIMELINE_COLS = 3       # thumbnails per row in segmentation timeline
    TIMELINE_IMG_W = 96
    ALLSEC_COLS = 5         # grid all-seconds preview
    ALLSEC_IMG_W = 64
    DETECT_OUT_SPLIT = None # stacked
else:
    HEADER_SPLIT = [3, 7]   # video | actions
    TIMELINE_COLS = 6
    TIMELINE_IMG_W = 150
    ALLSEC_COLS = 10
    ALLSEC_IMG_W = 95
    DETECT_OUT_SPLIT = [0.3, 0.7]


###############################################################################
# Page Header                                                                 #
###############################################################################
st.title("🎞️ Video Analysis")
st.divider()


###############################################################################
# 1) Ambil video dari query-param ATAU dari input manual                      #
###############################################################################
raw_url = params.get("video", [""][0]) if params else ""
video_path: pathlib.Path | None = None

if raw_url:  # datang dari halaman utama
    video_url = ul.unquote_plus(raw_url)
else:  # upload manual
    st.info("Upload file video (.mp4) terlebih dulu.")
    file_up = st.file_uploader("Upload video (.mp4)", type=["mp4"])
    if file_up is None:
        st.stop()  # tunggu input
    # Simpan ke cache dir (Streamlit ephemeral OK)
    upload_dir = pathlib.Path("ads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = upload_dir / f"upload_{datetime.now().timestamp():.0f}.mp4"
    tmp.write_bytes(file_up.read())
    video_path = tmp

# Jika dari URL, download
if video_path is None:
    with st.spinner("Downloading video …"):
        try:
            video_path = download_video_file(video_url)
        except Exception as e:
            st.error(f"Failed to download video: {e}")
            st.stop()


###############################################################################
# Header layout: video + action area                                          #
###############################################################################
if mobile_layout:
    st.video(str(video_path))
    st.success("Video ready for analysis ✨")
    action_container = st.container()  # placeholder for button & results
else:
    col1, col2 = st.columns(HEADER_SPLIT)
    with col1:
        st.video(str(video_path))
    with col2:
        st.success("Video ready for analysis ✨")
        action_container = st.container()  # results will render in this col


###############################################################################
# Gemini Prompt                                                               #
###############################################################################
PROMPT = """
SYSTEM:
You are a precise video-analysis assistant. Reply ONLY with valid JSON.

USER TASK:
1. Analyze the attached promo-video (Meta Ads). Produce per-second segmentation AND mark every visual scene-change for each segmentation. Each segmentation should be useful for competitor analysis (marketing / creative intent). A segmentation may contain one or more visual scene-change. **CONVERT INTO ABSOLUTE SECONDS**.
2. Provide a concise transcript of any visible text or on-screen captions (ignore spoken audio unless burned-in text is visible).

OUTPUT FORMAT (JSON ONLY—no markdown, no prose):
{
  "segmentations": [
    {
      "start_second": <int>,        # inclusive
      "end_second":   <int>,        # inclusive
      "label":        "<short description>",   # Indonesian preferred
      "scene_changes": [<int>, ...]            # 0+ ABSOLUTE seconds within [start_second,end_second]
    }
  ],
  "transcript": "<string>"
}

CONSTRAINTS:
- Visuals first; ignore audio unless text is visible.
- Use absolute seconds from start of video (01:10 → 70).
- Cover full runtime with no gaps/overlaps.
- Sort segments & scene_changes ascending.
- Keep labels ringkas (max ±8 kata).
- Return valid JSON. Nothing else.
"""


###############################################################################
# Gemini Call                                                                 #
###############################################################################
def call_gemini(video_path: pathlib.Path, prompt: str = PROMPT):
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found.")
        return None
    client = genai.Client()
    video_bytes = open(video_path, "rb").read()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
                    ),
                    types.Part(text=prompt),
                ]
            ),
            config={"response_mime_type": "application/json"},
        )
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

    raw_text = response.text
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        st.warning("Gemini tidak mengembalikan JSON valid — menampilkan string mentah.")
        return raw_text
    
def adapt_transcript(biz_name: str, industry: str, goal: str, original: str) -> str:
    """
    Kembalikan string transcript baru hasil adaptasi Gemini.
    """
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY not set."

    prompt_tpl = f"""
SYSTEM:
Anda adalah copywriter berpengalaman dalam iklan video berdurasi pendek.

USER TASK:
1. Adaptasikan transcript berikut agar cocok untuk bisnis **{biz_name}** ({industry}).
2. Tujuan utama: **{goal}**.
3. Pertahankan struktur baris & urutan kalimat sedekat mungkin dengan transcript asli
   agar durasi tiap adegan tetap mirip; boleh menambah/mengurangi <15 % kata jika perlu.
4. Gunakan bahasa Indonesia persuasif, selipkan CTA relevan jika cocok.
5. Jangan sebut brand lain ataupun platform Meta.

OUTPUT:
Berikan HANYA teks transcript baru (tanpa markdown, tanpa metadata).
---
TRANSCRIPT ASLI ⬇️
{original}
"""

    client = genai.Client()
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part(text=prompt_tpl)],
            config={"response_mime_type": "text/plain"},
        )
        return resp.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"


###############################################################################
# Segment button + results (render in `action_container`)                     #
###############################################################################
with action_container:
    if st.button("Segment Video with Gemini", use_container_width=True):
        with st.spinner("Gemini is working …"):
            st.session_state.seg_result = call_gemini(video_path)
            st.toast("Done!", icon="✅")

    if st.session_state.seg_result and isinstance(st.session_state.seg_result, dict):
        seg_df = pd.DataFrame(st.session_state.seg_result["segmentations"])
        seg_df["duration (s)"] = seg_df["end_second"] - seg_df["start_second"] + 1
        st.markdown("### 🖼️  Visual Timeline")

        # timeline grid responsive
        for _, row in seg_df.iterrows():
            start_s = int(row["start_second"])
            end_s = int(row["end_second"])
            label = row["label"]
            scene_list = row["scene_changes"] if isinstance(row["scene_changes"], list) else []
            if not scene_list:
                scene_list = [start_s]

            st.markdown(f"**{start_s}s – {end_s}s · {label}**")

            frames_this_seg = extract_frames(video_path, scene_list)

            cols = st.columns(TIMELINE_COLS)
            for i, s in enumerate(scene_list):
                if s in frames_this_seg:
                    cols[i % TIMELINE_COLS].image(frames_this_seg[s], width=TIMELINE_IMG_W, caption=None)
            
         # ### ░░ ZIP TIMELINE ░░ #############################################
        if st.button("⬇️  Download Visual Timeline Images (ZIP)", use_container_width=True):
            with st.spinner("Preparing zip …"):
                zip_io = build_timeline_zip(seg_df, video_path)
            st.download_button(
                "Save ZIP",
                data=zip_io,
                file_name=f"timeline_{video_path.stem}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.divider()

        

###############################################################################
# Transcript & Adaptation                                                     #
###############################################################################
if st.session_state.seg_result and isinstance(st.session_state.seg_result, dict):
    transcript = st.session_state.seg_result.get("transcript", "")
    
    # Transcript
    st.markdown("## 📝 Transcript")
    st.divider()
    st.write(transcript)
    
    # Adaptation
    st.markdown("### 🏗️ Adaptasi ke Bisnis Anda")
    biz_col1, biz_col2, biz_col3 = st.columns(3)
    with biz_col1:
        biz_name = st.text_input("Nama Bisnis", key="biz_name")
    with biz_col2:
        industry = st.text_input("Bidang Usaha", key="industry")
    with biz_col3:
        goal     = st.text_input("Tujuan Iklan", placeholder="Contoh: Meningkatkan booking potong rambut", key="goal")
    
    if st.button("🔄 Buat Transcript Baru", disabled=not transcript):
        if not (biz_name and industry and goal):
            st.warning("Lengkapi semua kolom terlebih dulu.")
        else:
            with st.spinner("Gemini mengadaptasi transcript …"):
                new_script = adapt_transcript(biz_name, industry, goal, transcript)
            st.markdown("### ✨ Transcript Adaptasi")
            st.write(new_script)



###############################################################################
# Object-detection single frame                                               #
###############################################################################
st.markdown("## 🎯 Object detection at specific second")
st.divider()

if video_path:
    end_second = int(get_video_duration(video_path))
    seconds = list(range(0, end_second + 1))
    frames = extract_frames(video_path, seconds)

    # grid all-seconds, responsive
    for row_start in range(0, end_second + 1, ALLSEC_COLS):
        row_end = min(row_start + ALLSEC_COLS - 1, end_second)
        st.markdown(f"**{row_start}-{row_end}s**")
        cols = st.columns(ALLSEC_COLS)
        for idx, sec in enumerate(range(row_start, row_end + 1)):
            if sec in frames:
                cols[idx].image(frames[sec], width=ALLSEC_IMG_W, caption=f"{sec}s")
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
                'and a text label in "label" to describe the object in Indonesian language. '
                'The coordinates in "box_2d" should be [ymin, xmin, ymax, xmax] and normalized to 1000.'
            )

            try:
                client = genai.Client()
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[Image.fromarray(frame_np), detect_prompt],
                    config={"response_mime_type": "application/json"},
                )

                raw = resp.text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1].strip()
                boxes = json.loads(raw)

                labels = defaultdict(int)

                for item in boxes:
                    label = item.get("label", "")
                    labels[label] += 1
                    ymin, xmin, ymax, xmax = item["box_2d"]

                    scale_x = w / 1000.0
                    scale_y = h / 1000.0
                    x1 = int(xmin * scale_x)
                    x2 = int(xmax * scale_x)
                    y1 = int(ymin * scale_y)
                    y2 = int(ymax * scale_y)
                    x1, x2 = max(0, x1), min(w - 1, x2)
                    y1, y2 = max(0, y1), min(h - 1, y2)

                    draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
                    draw.text(
                        (x1 + 4, y1 + 4),
                        label,
                        fill="black",
                        stroke_fill="white",
                        stroke_width=1,
                    )

                st.session_state.detect_img = img_pil
                st.session_state.detect_boxes = boxes
                st.session_state.labels = labels

            except Exception as e:
                st.error(f"Gemini detection error: {e}")


# render detection results persistently
if st.session_state.detect_img is not None:
    if DETECT_OUT_SPLIT is None:  # mobile stacked
        st.image(
            st.session_state.detect_img,
            caption=f"Frame @ {detect_sec}s (dengan bounding-box)",
            use_container_width=True,
        )
        for key, value in st.session_state.labels.items():
            st.write(f"{value}× **{key}**")
    else:  # desktop 2-col
        col1, col2 = st.columns(DETECT_OUT_SPLIT)
        with col1:
            st.image(
                st.session_state.detect_img,
                caption=f"Frame @ {detect_sec}s (dengan bounding-box)",
                use_container_width=False,
            )
        with col2:
            for key, value in st.session_state.labels.items():
                st.write(f"{value}×   **{key}**")
