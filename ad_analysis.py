import streamlit as st
import requests
import pandas as pd
import urllib.parse as ul
from datetime import timedelta
import plotly.express as px  # ➕ switched to Plotly Express for pie‑chart

API_KEY = st.secrets["SEARCH_API_KEY"]  
ENGINE_ENDPOINT = "https://www.searchapi.io/api/v1/search"

###############################################################################
# Helpers
###############################################################################

def seconds_to_hm(seconds: int | float | None) -> str:
    """Convert seconds → "XXh YYm" or "unknown" if None/NaN."""
    if pd.isna(seconds):
        return "unknown"
    try:
        t = timedelta(seconds=int(seconds))
        h, rem = divmod(t.seconds + t.days * 86400, 3600)
        m = rem // 60
        return f"{h}h {m}m"
    except Exception:
        return "unknown"

@st.cache_data(show_spinner=False)
def fetch_ads(keyword: str) -> pd.DataFrame:
    """Hit SearchAPI meta_ad_library once & cache the DataFrame."""
    resp = requests.get(
        ENGINE_ENDPOINT,
        params={
            "engine": "meta_ad_library",
            "q": keyword,
            "country": "ID",
            "api_key": API_KEY,
        },
        timeout=30,
    ).json()
    print(resp)

    # Normalise → DataFrame
    try:
        ads_df = pd.json_normalize(resp["ads"], errors="ignore")
    except Exception:
        ads_df = pd.DataFrame()
    return ads_df

###############################################################################
# UI
###############################################################################

st.set_page_config(page_title="Meta Ads Library Analysis", page_icon="📚", layout="wide")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.title("📚 Meta Ads Library Analysis")

with st.form("search_form"):
    keyword = st.text_input("Keyword", placeholder="barbershop", value="")
    submitted = st.form_submit_button("Search")

if submitted and keyword:
    with st.spinner("Fetching ads …"):
        ads = fetch_ads(keyword.strip())

    if ads.empty:
        st.warning("No ads returned for that keyword.")
        st.stop()

    st.success(f"Fetched **{len(ads)}** ads for *{keyword}*")

    # ------------------------------------------------------------------
    # SUMMARY ROW  (avg active time & platform distribution pie chart)
    # ------------------------------------------------------------------

    # ---- average active time -----------------------------------------
    avg_secs = ads["total_active_time"].dropna().astype(float).mean()
    avg_display = seconds_to_hm(avg_secs) if not pd.isna(avg_secs) else "unknown"
    st.metric(label="Average Active Time", value=avg_display)

    # ---- platform pie chart ------------------------------------------
    # Flatten & count platforms
    platform_counts = {}
    for p in ads["publisher_platform"].tolist():
        if isinstance(p, str):
            p_list = [p]
        else:
            p_list = p
        for plat in p_list:
            platform_counts[plat] = platform_counts.get(plat, 0) + 1

    if platform_counts:
        df_plat = pd.DataFrame({
            "Platform": list(platform_counts.keys()),
            "Count": list(platform_counts.values()),
        })
        fig = px.pie(df_plat, names="Platform", values="Count", hole=0.35)
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No platform data to plot.")

    st.divider()

    #########################  Render as 2‑cards/row  ########################
    ADS_PER_ROW = 3
   
    for idx in range(0, len(ads), ADS_PER_ROW):
        row_df = ads.iloc[idx : idx + ADS_PER_ROW]
        cols = st.columns(len(row_df))  # 1 or 2 columns depending on tail

        for col, (_, ad) in zip(cols, row_df.iterrows()):
            with col:
                # make bordered card with uniform height via CSS class
                with st.container(border=True, key=f"card-{ad['ad_archive_id']}"):
                    # Header ─ logo + page name --------------------------------
                    hcols = st.columns([2, 10])
                    ppic = ad.get("snapshot.page_profile_picture_url")
                    if ppic and not pd.isna(ppic):
                        hcols[0].image(ppic, width=50)
                    page_name = ad.get("page_name", "Unknown Page")
                    hcols[1].markdown(f"#### {page_name}")

                    # Determine video link early (used in footer column) -----
                    video_link = None
                    vids = ad.get("snapshot.videos")
                    if isinstance(vids, list) and vids:
                        first = vids[0]
                        video_link = first.get("video_sd_url") or first.get("video_hd_url")

                    # --- Upper ------------------------------------------------
                    active_time = seconds_to_hm(ad.get("total_active_time"))
                    platforms = ad.get("publisher_platform", [])
                    if isinstance(platforms, str):
                        platforms = [platforms]
                    platforms_display = ", ".join(platforms) if platforms else "unknown"
                    st.caption(f"**Active:** {active_time} | **Platforms:** {platforms_display}")

                    txt = ad.get("snapshot.body.text", "")
                    if txt and not pd.isna(txt):
                        st.write(txt[:500] + ("…" if len(txt) > 500 else ""))

                    # --- Lower ---------------------------------------------
                    if video_link:
                        safe = ul.quote_plus(video_link)
                        target = f"/video_analysis?video={safe}"
                        st.markdown(
                            f"<a href='{target}' target='_blank'>"
                            "<button style='width:100%;height:100%;padding:0.6em;"
                            "border:none;background:#4CAF50;color:white;border-radius:4px'>"
                            "Analyze Video"
                            "</button></a>",
                            unsafe_allow_html=True,
                        )

                    # close ad-card div
                    st.markdown("</div>", unsafe_allow_html=True)
