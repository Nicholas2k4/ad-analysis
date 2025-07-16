import streamlit as st
import os, json, pathlib, hashlib, requests
from collections import defaultdict
from google import genai
from google.genai import types
import PIL.Image as Image
from PIL import ImageDraw

# ──────────────────── Konfigurasi & API key ─────────────────────
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ──────────────────── Helper download (kalau user beri URL) ─────────────────
@st.cache_data(show_spinner=False)
def download_image(url: str) -> pathlib.Path:
    dst  = pathlib.Path("uploads")
    dst.mkdir(exist_ok=True)
    fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    path  = dst / fname
    if path.exists():
        return path
    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2**14):
                f.write(chunk)
    return path

# ──────────────────── UI ─────────────────────────────────────────
st.set_page_config(page_title="Object Detection", page_icon="📸", layout="wide")
st.title("📸 Object Detection")
st.divider()

uploaded_file = st.file_uploader("Upload image (.jpg/.png)", type=["jpg","jpeg","png"])

# ──────────────────── Convert input jadi PIL.Image ──────────────
img_path: pathlib.Path | None = None
if uploaded_file:
    img_path = pathlib.Path("uploads") / f"upload_{uploaded_file.name}"
    img_path.parent.mkdir(exist_ok=True)
    img_path.write_bytes(uploaded_file.read())

if img_path is None:
    st.stop()

img_pil = Image.open(img_path).convert("RGB")
w, h    = img_pil.size
st.image(img_pil, caption="Gambar input", use_container_width=False)

# ──────────────────── Tombol DETECT ─────────────────────────────
if st.button("Detect objects with Gemini"):
    with st.spinner("Gemini sedang menganalisis …"):
        detect_prompt = (
            "Give segmentation masks of the image, with no more than 10 items. "
            'Output ONLY A JSON list where each entry contains the 2D bounding box in "box_2d" '
            'and a text label in "label" to describe the object in Indonesian language. '
            "Coordinates in box_2d must be [ymin, xmin, ymax, xmax] **normalised to 1000**."
        )

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            resp   = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img_pil, detect_prompt],
                config ={"response_mime_type":"application/json"},
            )

            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].strip()
            boxes = json.loads(raw)

        except Exception as e:
            st.error(f"Gemini API error: {e}")
            st.stop()

    # ────────── Gambar bounding-box & hitung label ────────────
    draw   = ImageDraw.Draw(img_pil)
    counts = defaultdict(int)

    for item in boxes:
        label = item.get("label","")
        counts[label] += 1
        ymin,xmin,ymax,xmax = item["box_2d"]        # 0-1000

        # skala ke pixel asli
        sx, sy = w/1000.0, h/1000.0
        x1,x2  = int(xmin*sx), int(xmax*sx)
        y1,y2  = int(ymin*sy), int(ymax*sy)

        draw.rectangle([x1,y1,x2,y2], outline="white", width=3)
        draw.text((x1+4, y1+4), label,
                  fill="black", stroke_fill="white", stroke_width=1)

    # ────────── Tampilkan hasil ────────────
    colA, colB = st.columns([0.3,0.7])
    with colA:
        st.image(img_pil, caption="Deteksi objek", use_container_width=False)
    with colB:
        st.markdown("**Objects**")
        for lbl,cnt in counts.items():
            st.write(f"{cnt}× **{lbl}**")
