import os, asyncio, torch
torch.classes.__path__ = []                                   # fix watcher crash
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------------------------------------------------------------------------
# 📦  standard imports
# ---------------------------------------------------------------------------
import io, pathlib, requests
import streamlit as st
import pandas as pd
from PIL import Image
from st_aggrid import AgGrid
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 🗂️  list all on-disk YOLO weights you want to expose
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "Breed Detection"   : "CattleScanner/breed.pt",
    "Breed Grade"       : "CattleScanner/grade.pt",
    "bcs"               : "CattleScanner/bcs.pt",
    "Skin Coat"         : "CattleScanner/coat.pt",
    "Udder type"        : "CattleScanner/UdderType.pt",
    "Body Segmentation" : "CattleScanner/side_segmentation_model.pt",
    "Teat score"        : "CattleScanner/teat.pt",
    "Side Keypoint"     : "CattleScanner/side_keypoint_model.pt",
}

# ---------------------------------------------------------------------------
# 🔧  page-wide Streamlit settings + CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🐄 Gau Swastha – AI Health Report",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
        html, body, [class*="css"]  { font-size:16px !important; }
        h1, h2, h3, h4              { font-size:1.35em !important; }
        h1                          { font-size:1.6em !important; }
        .ag-theme-streamlit .ag-cell,
        .ag-theme-streamlit .ag-header-cell-label {
            font-size:15px !important; line-height:22px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 🗄️  session-state initialisation
# ---------------------------------------------------------------------------
SS = st.session_state
SS.setdefault("view",         "report")
SS.setdefault("img_bytes",     None)
SS.setdefault("api_json",      None)
SS.setdefault("models",        {})      # {key: YOLO()}
SS.setdefault("renders",       {})      # {key: ndarray}
SS.setdefault("results",       {})      # {key: Results}
SS.setdefault("tables",        {})      # {key: DataFrame}
SS.setdefault("model_choice",  None)

# ---------------------------------------------------------------------------
# 🔀  tiny router helpers
# ---------------------------------------------------------------------------
def switch_view(page: str)          -> None: SS.view = page
def clear_models_and_outputs()      -> None:
    SS.models.clear(); SS.renders.clear()
    SS.results.clear(); SS.tables.clear()
    SS.model_choice = None

# ---------------------------------------------------------------------------
# 🛠  Ultralytics Results → pandas helper  (Ultralytics ≥ 0.5)
# ---------------------------------------------------------------------------
def results_to_df(res) -> pd.DataFrame:
    """Return empty DF if no detections."""
    if res.boxes is None or res.boxes.xyxy is None:
        return pd.DataFrame()
    import numpy as np
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls  = res.boxes.cls.cpu().numpy().astype(int)
    names = [res.names[int(i)] for i in cls]
    return pd.DataFrame(dict(
        xmin=xyxy[:, 0], ymin=xyxy[:, 1],
        xmax=xyxy[:, 2], ymax=xyxy[:, 3],
        confidence=np.round(conf, 3),
        name=names
    )).sort_values("confidence", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# ⚡  model-runner (load-once, run-once, cache)
# ---------------------------------------------------------------------------
def run_model_on_image(model_key: str) -> None:
    if SS.img_bytes is None:
        st.warning("No image found. Please generate a report first.")
        return

    # 1. load model if needed
    if model_key not in SS.models:
        path = pathlib.Path(MODEL_CONFIG[model_key])
        if not path.exists():
            st.error(f"❌ Weight file not found:\n{path}")
            return
        with st.spinner(f"Loading {model_key} …"):
            SS.models[model_key] = YOLO(str(path))

    # 2. run inference if not cached
    if model_key not in SS.renders:
        with st.spinner(f"Running {model_key} …"):
            img = Image.open(io.BytesIO(SS.img_bytes))
            res = SS.models[model_key](img)[0]          # first image
            SS.results[model_key] = res
            SS.renders[model_key] = res.plot()
            SS.tables[model_key]  = results_to_df(res)

    SS.model_choice = model_key

# ---------------------------------------------------------------------------
# 🅰️  REPORT page  (upload → call API → display)
# ---------------------------------------------------------------------------
def show_report():
    st.title("🐄 CattleSense — AI Health Report")

    API_URL = "https://dev-scanner.silofortune.com/api/v2_5/cattle-scanner"
    img_file = st.file_uploader("Upload *side-profile* image", type=["jpg", "jpeg", "png"])
    lang     = st.selectbox("Report language", ["en", "hi", "te", "ta"], index=0)

    if img_file and st.button("Generate report"):
        try:
            img_bytes = img_file.read()
            with st.spinner("Contacting scanner …"):
                r = requests.post(
                    API_URL,
                    files={"side-img": (img_file.name, img_bytes, "image/jpeg")},
                    data={"language": lang},
                    timeout=60,
                )
                r.raise_for_status()
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()

        SS.img_bytes = img_bytes
        SS.api_json  = r.json()
        clear_models_and_outputs()
        st.success("Report generated – scroll down or open **Model Output**")

    # ── nav bar ──────────────────────────────────────────────────
    if SS.api_json:
        col1, col2 = st.columns(2)
        col1.button("🔄 View Report", disabled=True)
        col2.button("🖼️ View Model Output", on_click=switch_view, args=("model",))

        st.image(SS.img_bytes, caption="Uploaded image", use_container_width=True)

        data = SS.api_json

        # 1) Animal details
        st.subheader("Animal Details")
        det = data["animal-details"]
        st.table(pd.DataFrame({
            "Animal Type"   : [det["animal-type"].title()],
            "Breed"         : [det["breed"]["breed"].replace("-", " ")],
            "Breed Grade"   : [det["breed-grade"]["breed-grade"]],
            "Body Weight kg": [det["body-weight"]],
        }))

        # 2) Economic parameters
        st.subheader("Economic Parameters")
        econ = data["animal-economic-status"]
        st.table(pd.DataFrame({
            "Body Condition Score"       : [econ["bcs"]["value"]],
            "Approx. Milk Yield (L/day)" : [econ["milk yield"]],
            "Production Capacity (L/day)": [econ["production-capacity"]],
            "Lactation Yield (L)"        : [econ["lactation-yield"]],
            "Breeding Capacity"          : [econ["breeding-capacity"]],
            "Market Value (₹)"           : [econ["market-value"]],
            "Buying Recommendation"      : [econ["buying-recommendation"]],
        }))

        # 3) General health
        st.subheader("General Health Conditions")
        gh_rows = []
        for k, v in data["general-health-condition"].items():
            gh_rows.append({
                "Parameter"     : k.replace("-", " ").title(),
                "Status"        : v.get("value") if isinstance(v, dict) else v,
                "Interpretation": v.get("interpretation", "") if isinstance(v, dict) else "",
                "Recommendation": v.get("recommendation", "") if isinstance(v, dict) else "",
            })
        AgGrid(pd.DataFrame(gh_rows), fit_columns_on_grid_load=True, height=300, theme="streamlit")

        # 4) Disorders
        st.subheader("Disorder Status by System")
        sd_rows = []
        for system, sys_dict in data["system-of-disorder"].items():
            for issue, meta in sys_dict.items():
                sd_rows.append({
                    "System"        : system.replace("-", " ").title(),
                    "Issue"         : issue.replace("-", " ").title(),
                    "Detected"      : meta["value"],
                    "Interpretation": meta["interpretation"],
                    "Recommendation": meta["recommendation"],
                })
        AgGrid(pd.DataFrame(sd_rows), fit_columns_on_grid_load=True, height=350, theme="alpine")

        # 5) Diet
        st.subheader("Balanced Ration – Feed / Fodder Plan")
        diet_tabs = st.tabs(["Green-Dry Fodder Plan", "Maize-Silage Plan"])
        for (key, plan), tab in zip(data["diet"].items(), diet_tabs):
            with tab:
                st.dataframe(
                    pd.Series(plan).rename_axis("Feed / Fodder").to_frame(key.replace("_", " ").upper()),
                    use_container_width=True,
                )

        with st.expander("🔍 Raw JSON payload"):
            st.json(data, expanded=False)

        st.caption("Disclaimer — AI-generated advisory. Always consult a vet.")

    else:
        st.info("Upload an image and click **Generate report** to begin.")

# ---------------------------------------------------------------------------
# 🅱️  MODEL OUTPUT page
# ---------------------------------------------------------------------------
def show_model_output():
    st.title("🖼️ Model Outputs")
    st.button("🔙 View Report", on_click=switch_view, args=("report",))

    if SS.img_bytes is None:
        st.warning("No image found. Generate a report first.")
        return

    # --- model selector buttons -----------------------------------
    cols = st.columns(len(MODEL_CONFIG))
    for (key, _), col in zip(MODEL_CONFIG.items(), cols):
        col.button(key, on_click=run_model_on_image, args=(key,), key=f"btn_{key}")

    # --- display chosen model output ------------------------------
    if SS.model_choice:
        key = SS.model_choice
        st.subheader(f"Output – {key}")
        st.image(SS.renders[key], caption=key, use_container_width=True)

        df = SS.tables[key]
        if not df.empty:
            st.subheader("Detections")
            st.table(df)

        st.caption("Weight file: " + MODEL_CONFIG[key])
    else:
        st.info("Click a model button above to run inference.")

# ---------------------------------------------------------------------------
# 🚦  ROUTE
# ---------------------------------------------------------------------------
if SS.view == "report":
    show_report()
else:
    show_model_output()