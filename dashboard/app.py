import html
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONSTANTS
# ============================================================
# ============================================================
# CHENNAI REFERENCE LOCATION
# ============================================================

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707

ROOT = Path(__file__).resolve().parents[1]
TIMELINE_FILE = ROOT / "data" / "forecastmitr_live_timeline.csv"
GRID_FILE = ROOT / "data" / "forecastmitr_chennai_grid.csv"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_live_pipeline.py"

BUST_THRESHOLD = 69.0
CRITICAL_THRESHOLD = 85.0
PIPELINE_TIMEOUT_SECONDS = 300  # hard safety cap for the background update

REQUIRED_COLUMNS = [
    "issue_time",
    "target_time",
    "lead_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_min",
    "ensemble_max",
    "ensemble_spread",
    "bust_probability",
    "predicted_bust",
    "risk_level",
    "diagnosis",
    "primary_signal",
]

NUMERIC_COLUMNS = [
    "lead_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_min",
    "ensemble_max",
    "ensemble_spread",
    "bust_probability",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ForecastMitr",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI / THEME
# ============================================================

# ForecastMitr uses a single dark theme. Light mode has been removed.
if "audience" not in st.session_state:
    st.session_state.audience = "Forecast Officer"

# Fixed dark theme
BG = "#0B1220"
SURFACE = "#111827"
SURFACE_2 = "#172033"
TEXT = "#FFFFFF"
MUTED = "#9CA3AF"
BORDER = "#263244"
GRID = "#273449"


st.markdown(
    f"""
    <style>
    :root {{
        --fm-bg: {BG};
        --fm-surface: {SURFACE};
        --fm-surface2: {SURFACE_2};
        --fm-text: {TEXT};
        --fm-muted: {MUTED};
        --fm-border: {BORDER};
        --fm-grid: {GRID};
    }}

    html {{
        scroll-behavior: smooth;
    }}

    .stApp {{
        background: var(--fm-bg);
        color: var(--fm-text);
    }}

    .stApp * {{
        color: var(--fm-text);
    }}

    .fm-anchor {{
        scroll-margin-top: 80px;
        height: 1px;
        width: 1px;
    }}

    /* ----------------------------------------------------
       SIDEBAR
    ---------------------------------------------------- */

    [data-testid="stSidebar"] {{
        background: var(--fm-surface);
        border-right: 1px solid var(--fm-border);
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.4rem;
    }}

    .fm-sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 2px 2px 12px;
    }}

    .fm-sidebar-title {{
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.4px;
        color: var(--fm-text);
        line-height: 1.15;
    }}

    .fm-sidebar-subtitle {{
        font-size: 11px;
        color: var(--fm-muted);
        margin-top: 3px;
        line-height: 1.3;
    }}

    .fm-nav-label {{
        color: var(--fm-muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin: 4px 0 6px;
    }}

    .fm-nav-box {{
        background: var(--fm-surface2);
        border: 1px solid var(--fm-border);
        border-radius: 12px;
        padding: 6px;
    }}

    .fm-nav-link {{
        display: flex;
        align-items: center;
        padding: 8px 10px;
        margin: 2px 0;
        border-radius: 8px;
        color: var(--fm-text) !important;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 13px;
        transition: background 0.12s ease;
    }}

    .fm-nav-link:hover {{
        background: rgba(96, 165, 250, 0.14);
        color: #60A5FA !important;
    }}

    .fm-sidebar-footer {{
        font-size: 11px;
        color: var(--fm-muted);
        line-height: 1.6;
        padding-top: 10px;
        margin-top: 10px;
        border-top: 1px solid var(--fm-border);
    }}

    .fm-sidebar-footer b {{
        color: var(--fm-muted) !important;
    }}

    /* ----------------------------------------------------
       PIPELINE FLOWCHART
    ---------------------------------------------------- */

    .pipeline-flowchart {{
        display: flex;
        align-items: stretch;
        gap: 8px;
        overflow-x: auto;
        padding: 18px 4px 24px 4px;
        margin-top: 12px;
    }}

    .pipeline-flowchart::-webkit-scrollbar {{
        height: 8px;
    }}

    .pipeline-flowchart::-webkit-scrollbar-thumb {{
        background: var(--fm-border);
        border-radius: 8px;
    }}

    .pipeline-flowchart::-webkit-scrollbar-track {{
        background: transparent;
    }}

    .pipeline-node {{
        min-width: 145px;
        max-width: 175px;
        min-height: 96px;
        padding: 14px 12px;
        border: 1px solid #334155;
        border-radius: 12px;
        background: linear-gradient(180deg, #131c2e 0%, #0f1729 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        box-shadow: 0 5px 18px rgba(0,0,0,0.18);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }}

    .pipeline-node:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 26px rgba(0,0,0,0.34);
        border-color: #60A5FA;
    }}

    .pipeline-start,
    .pipeline-end {{
        border-color: #22C55E;
    }}

    .pipeline-start:hover,
    .pipeline-end:hover {{
        border-color: #4ADE80;
    }}

    .pipeline-model {{
        border-color: #F59E0B;
    }}

    .pipeline-model:hover {{
        border-color: #FBBF24;
    }}

    .pipeline-node-title {{
        font-size: 13px;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.25;
    }}

    .pipeline-node-sub {{
        margin-top: 7px;
        font-size: 11px;
        line-height: 1.35;
        color: #CBD5E1 !important;
    }}

    .pipeline-arrow {{
        align-self: center;
        flex: 0 0 auto;
        font-size: 22px;
        font-weight: 800;
        color: #60A5FA !important;
        padding: 0 2px;
    }}

    /* ----------------------------------------------------
       GENERIC TEXT ELEMENTS
    ---------------------------------------------------- */

    .small-label {{
        color: #2563EB;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }}

    .section-title {{
        color: var(--fm-text);
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.4px;
    }}

    .warning-card {{
        border-left: 5px solid #F59E0B;
        background: #241C0D;
        border-top: 1px solid #5B4A29;
        border-right: 1px solid #5B4A29;
        border-bottom: 1px solid #5B4A29;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 16px 0;
    }}

    .warning-title {{
        color: #FCD34D;
        font-size: 17px;
        font-weight: 800;
    }}

    .warning-text {{
        color: var(--fm-text);
        font-size: 14px;
        margin-top: 5px;
        line-height: 1.5;
    }}

    .footer {{
        color: var(--fm-muted);
        text-align: center;
        font-size: 12px;
        padding-top: 24px;
        line-height: 1.7;
    }}

    .stApp .stCaption,
    .stApp [data-testid="stCaptionContainer"] {{
        color: var(--fm-text) !important;
    }}

    .stApp input,
    .stApp textarea,
    .stApp select {{
        color: var(--fm-text) !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1480px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }}

    .fm-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }}

    .fm-mark {{
        width: 42px;
        height: 42px;
        border-radius: 11px;
        display: grid;
        place-items: center;
        font-weight: 800;
        font-size: 18px;
        background: #1D4ED8;
        color: white;
        flex-shrink: 0;
    }}

    .fm-title {{
        font-size: 30px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.7px;
        color: var(--fm-text);
    }}

    .fm-subtitle {{
        color: var(--fm-muted);
        font-size: 14px;
        margin-top: 5px;
    }}

    .fm-eyebrow {{
        color: #2563EB;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .fm-hero {{
        background: var(--fm-surface);
        border: 1px solid var(--fm-border);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }}

    .fm-hero h1 {{
        color: var(--fm-text);
        font-size: 30px;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .fm-hero p {{
        color: var(--fm-muted);
        margin: 7px 0 0 0;
        font-size: 15px;
    }}

    .fm-status {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        border: 1px solid var(--fm-border);
        background: var(--fm-surface2);
        border-radius: 999px;
        padding: 6px 10px;
        color: var(--fm-text);
        font-size: 12px;
        font-weight: 700;
    }}

    .fm-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #16A34A;
    }}

    .fm-card {{
        background: var(--fm-surface);
        border: 1px solid var(--fm-border);
        border-radius: 14px;
        padding: 17px 18px;
        min-height: 112px;
    }}

    .fm-card-label {{
        color: var(--fm-muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .8px;
        font-weight: 800;
    }}

    .fm-card-value {{
        color: var(--fm-text);
        font-size: 24px;
        font-weight: 800;
        margin-top: 6px;
    }}

    .fm-card-note {{
        color: var(--fm-muted);
        font-size: 12px;
        margin-top: 4px;
    }}

    .fm-alert {{
        border-left: 5px solid #DC2626;
        background: #26151A;
        border-top: 1px solid #5B2931;
        border-right: 1px solid #5B2931;
        border-bottom: 1px solid #5B2931;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 16px 0;
    }}

    .fm-alert-title {{
        color: #FCA5A5;
        font-size: 18px;
        font-weight: 800;
    }}

    .fm-alert-body {{
        color: var(--fm-text);
        font-size: 14px;
        margin-top: 5px;
    }}

    .fm-public {{
        background: var(--fm-surface);
        border: 1px solid var(--fm-border);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 10px 0 18px;
    }}

    .fm-risk {{
        font-size: 48px;
        font-weight: 850;
        line-height: 1;
        color: var(--fm-text);
        margin: 7px 0;
    }}

    .fm-explain {{
        color: var(--fm-muted);
        font-size: 14px;
        max-width: 700px;
        margin: 0 auto;
    }}

    .fm-section {{
        color: var(--fm-text);
        font-size: 20px;
        font-weight: 800;
        margin: 26px 0 8px;
    }}

    .fm-section-note {{
        color: var(--fm-muted);
        font-size: 13px;
        margin-bottom: 10px;
    }}

    .fm-footer {{
        color: var(--fm-muted);
        text-align: center;
        font-size: 12px;
        padding-top: 24px;
    }}

    /* Make Streamlit metric/table areas fit the custom theme. */
    [data-testid="stMetric"] {{
        background: var(--fm-surface);
        border: 1px solid var(--fm-border);
        border-radius: 12px;
        padding: 13px 15px;
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--fm-muted) !important;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--fm-text) !important;
    }}

    .stDataFrame {{
        border: 1px solid var(--fm-border);
        border-radius: 12px;
        overflow: hidden;
    }}

    div[data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="fm-sidebar-brand">
        <div class="fm-mark">FM</div>
        <div>
            <div class="fm-sidebar-title">ForecastMitr</div>
            <div class="fm-sidebar-subtitle">Weather Forecast Bust Detection</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

st.sidebar.markdown('<div class="fm-nav-label">Dashboard view</div>', unsafe_allow_html=True)
st.session_state.audience = st.sidebar.radio(
    "Dashboard view",
    ["Forecast Officer", "Public Summary"],
    index=0 if st.session_state.audience == "Forecast Officer" else 1,
    format_func=lambda x: ("👮 " if x == "Forecast Officer" else "📢 ") + x,
    label_visibility="collapsed",
)

if st.session_state.audience == "Forecast Officer":
    jump_links = [
        ("overview", "🧭 Overview"),
        ("chennai-heatmap", "🗺️ Chennai Heatmap"),
        ("forecast-analysis", "📈 Forecast Analysis"),
        ("data-explorer", "🔍 Data Explorer"),
        ("diagnostics", "🩺 Diagnostics"),
        ("methodology", "📘 Methodology"),
    ]
else:
    jump_links = [
        ("simple-forecast", "🌦️ Simple Forecast"),
        ("chennai-heatmap", "🗺️ Chennai Heatmap"),
        ("what-it-means", "💬 What It Means"),
        ("methodology", "📘 Methodology"),
    ]

st.sidebar.markdown('<div class="fm-nav-label">Jump to</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="fm-nav-box">'
    + "".join(
        f'<a class="fm-nav-link" href="#{anchor}">{label}</a>'
        for anchor, label in jump_links
    )
    + "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="fm-sidebar-footer">
        <b>SIH26079</b> • Operational decision support<br>
        ForecastMitr supports forecast assessment; it does not replace official IMD warnings.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def esc(value) -> str:
    """HTML-escape pipeline data before rendering it as HTML."""
    return html.escape(str(value))


def render_card(title: str, value: str, note: str = "") -> None:
    note_html = f'<div class="fm-card-note">{esc(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="fm-card">
            <div class="fm-card-label">{esc(title)}</div>
            <div class="fm-card-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, note: str = "") -> None:
    note_html = f'<div class="fm-section-note">{esc(note)}</div>' if note else ""
    st.markdown(
        f'<div class="fm-section">{esc(title)}</div>{note_html}',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_timeline(path_str: str, mtime: float) -> pd.DataFrame:
    """Load and clean the timeline CSV."""
    frame = pd.read_csv(path_str)

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["lead_hours", "bust_probability"])
    if frame.empty:
        raise ValueError(
            "No valid rows remain after cleaning lead_hours/bust_probability."
        )

    # Engine stores the bust score as 0-1; UI displays 0-100%.
    if frame["bust_probability"].max() <= 1.5:
        frame["bust_probability"] *= 100.0

    return frame.sort_values("lead_hours").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_chennai_grid(path_str: str, mtime: float) -> pd.DataFrame:
    """Load the spatial GEFS ensemble grid generated by the live pipeline."""
    frame = pd.read_csv(path_str)

    required = [
        "issue_time", "target_time", "lead_hours",
        "lat", "lon", "control_forecast",
        "ensemble_mean", "ensemble_median",
        "ensemble_min", "ensemble_max", "ensemble_spread",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing spatial columns: {missing}")

    numeric = [
        "lead_hours", "lat", "lon", "control_forecast",
        "ensemble_mean", "ensemble_median", "ensemble_min",
        "ensemble_max", "ensemble_spread",
    ]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["lead_hours", "lat", "lon", "ensemble_mean"])
    if frame.empty:
        raise ValueError("No valid Chennai spatial grid rows remain after cleaning.")

    return frame.sort_values(["lead_hours", "lat", "lon"]).reset_index(drop=True)


# ============================================================
# HEADER + LIVE UPDATE CONTROL
# ============================================================

st.markdown(
    """
    <div class="fm-hero">
        <div class="fm-brand">
            <div class="fm-mark">FM</div>
            <div>
                <div class="fm-title">ForecastMitr</div>
                <div class="fm-subtitle">Weather Forecast Bust Detection & Uncertainty Intelligence</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

update_col, status_col = st.columns([1.0, 3.2])

with update_col:
    update_clicked = st.button(
        "↻  Update live forecast",
        type="primary",
        use_container_width=True,
        help="Fetch the latest available GEFS cycle and regenerate the ForecastMitr timeline.",
    )

with status_col:
    if TIMELINE_FILE.exists():
        modified_time = pd.Timestamp(TIMELINE_FILE.stat().st_mtime, unit="s")
        st.markdown(
            f"""
            <div class="fm-status">
                <span class="fm-dot"></span>
                Pipeline data available
            </div>
            <div style="color:var(--fm-muted);font-size:12px;margin-top:7px;">
                Last processed locally: {modified_time.strftime('%d %b %Y, %H:%M:%S')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No live forecast timeline has been generated yet.")

def _runtime_paths():
    runtime = ROOT / ".forecastmitr_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime, runtime / "live_pipeline.pid", runtime / "live_pipeline.log"


def _process_is_running(pid: int) -> bool:
    import os

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in result.stdout
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def start_live_update():
    """Launch the pipeline independently so the Streamlit page never waits on it."""
    runtime, pid_file, log_file = _runtime_paths()

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if _process_is_running(old_pid):
                return False
        except Exception:
            pass
        try:
            pid_file.unlink()
        except OSError:
            pass

    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(PIPELINE_SCRIPT)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    return True


def get_live_update_status():
    """Return running state and the latest pipeline log."""
    _, pid_file, log_file = _runtime_paths()

    if not pid_file.exists():
        return False, ""

    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return False, ""

    running = _process_is_running(pid)

    log_text = ""
    if log_file.exists():
        try:
            log_text = log_file.read_text(
                encoding="utf-8", errors="replace"
            )[-6000:]
        except Exception:
            pass

    if not running:
        try:
            pid_file.unlink()
        except OSError:
            pass

    return running, log_text


if update_clicked:
    if not PIPELINE_SCRIPT.exists():
        st.error(f"Live pipeline not found: `{PIPELINE_SCRIPT}`")
    else:
        running, _ = get_live_update_status()
        if running:
            st.warning(
                "A live forecast update is already running. "
                "No duplicate update was started."
            )
        elif start_live_update():
            st.success(
                "Live forecast update started in the background. "
                "The dashboard will remain usable while GEFS data is processed."
            )
        else:
            st.warning("A live forecast update is already running.")

running, live_log = get_live_update_status()

if running:
    st.info(
        "🟡 **Live GEFS update in progress.** "
        "You can continue browsing the existing forecast."
    )
    if st.button("CHECK UPDATE STATUS", use_container_width=False):
        st.rerun()
    if live_log:
        with st.expander("Pipeline progress", expanded=False):
            st.code(live_log, language="text")
elif live_log and (
    "error" in live_log.lower() or "traceback" in live_log.lower()
):
    with st.expander("Last pipeline log", expanded=False):
        st.code(live_log, language="text")


# ============================================================
# LOAD DATA
# ============================================================

if not TIMELINE_FILE.exists():
    st.warning(
        "Live timeline data is not available yet. "
        "Click **Update live forecast** to download GEFS data and generate it."
    )
    st.stop()

try:
    df = load_timeline(str(TIMELINE_FILE), TIMELINE_FILE.stat().st_mtime)
except ValueError as exc:
    st.error(f"Could not load timeline data: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error reading timeline data: {exc}")
    st.stop()

spatial_df = None
if GRID_FILE.exists():
    try:
        spatial_df = load_chennai_grid(
            str(GRID_FILE),
            GRID_FILE.stat().st_mtime,
        )
    except Exception as exc:
        st.warning(f"Could not load the Chennai spatial grid: {exc}")


# ============================================================
# SHARED ANALYTICS
# ============================================================

bust_df = df[df["bust_probability"] >= BUST_THRESHOLD]
high_df = df[df["risk_level"].astype(str).str.upper().isin(["HIGH", "CRITICAL"])]
peak_row = df.loc[df["bust_probability"].idxmax()]
earliest_bust = bust_df.iloc[0] if len(bust_df) else None
peak_probability = float(peak_row["bust_probability"])
issue_time = esc(df.iloc[0]["issue_time"])


def chart_layout(fig, height=430):
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        hovermode="x unified",
        margin=dict(l=40, r=30, t=30, b=50),
    )
    return fig


def page_header(kicker, title, description=""):
    st.markdown(
        f'<div class="small-label">{esc(kicker.upper())}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-title" style="margin-top:4px">{esc(title)}</div>',
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)


def section_anchor(anchor: str) -> None:
    st.markdown(f'<div id="{esc(anchor)}" class="fm-anchor"></div>', unsafe_allow_html=True)


def render_pipeline_flowchart() -> None:
    """Render the ForecastMitr processing pipeline as a visual flowchart."""

    flowchart_html = """
    <style>
        .pipeline-flowchart {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: flex-start;
            gap: 10px;
            padding: 15px 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .pipeline-node {
            background-color: #262730;
            border: 1px solid #464b5d;
            border-radius: 8px;
            padding: 15px 17px;
            width: 480px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }
        .pipeline-start {
            background-color: #1e3a8a;
            border-color: #3b82f6;
        }
        .pipeline-model {
            background-color: #7c2d12;
            border-color: #f97316;
        }
        .pipeline-end {
            background-color: #14532d;
            border-color: #22c55e;
        }
        .pipeline-node-title {
            font-weight: 600;
            font-size: 0.85rem;
            color: #ffffff;
            margin-bottom: 4px;
        }
        .pipeline-node-sub {
            font-size: 0.72rem;
            color: #d1d5db;
            line-height: 1.7;
        }
        .pipeline-arrow {
            font-size: 1.2rem;
            color: #9ca3af;
            font-weight: bold;
        }
    </style>

    <div class="pipeline-flowchart">

        <div class="pipeline-node pipeline-start">
            <div class="pipeline-node-title">START</div>
            <div class="pipeline-node-sub">Live forecast update</div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node">
            <div class="pipeline-node-title">1. GEFS Retrieval</div>
            <div class="pipeline-node-sub">
                Download latest 0.25° Chennai-area data
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node">
            <div class="pipeline-node-title">2. Ensemble Members</div>
            <div class="pipeline-node-sub">
                Control + 4 perturbation members
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node">
            <div class="pipeline-node-title">3. Spatial Grid</div>
            <div class="pipeline-node-sub">
                Preserve rainfall at every GEFS grid point
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node">
            <div class="pipeline-node-title">4. Ensemble Analysis</div>
            <div class="pipeline-node-sub">
                Mean • median • min • max • spread
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node pipeline-model">
            <div class="pipeline-node-title">5. ForecastMitr XGBoost</div>
            <div class="pipeline-node-sub">
                Chennai point risk + bust probability
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node">
            <div class="pipeline-node-title">6. Spatial Uncertainty</div>
            <div class="pipeline-node-sub">
                Relative spread + control divergence
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node">
            <div class="pipeline-node-title">7. Outputs</div>
            <div class="pipeline-node-sub">
                Timeline + spatial heatmap
            </div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-node pipeline-end">
            <div class="pipeline-node-title">DASHBOARD</div>
            <div class="pipeline-node-sub">
                Risk • diagnostics • map
            </div>
        </div>

    </div>
    """

    if hasattr(st, "html"):
        st.html(flowchart_html)
    else:
        st.markdown(flowchart_html, unsafe_allow_html=True)

def render_chennai_heatmap(
    timeline: pd.DataFrame,
    spatial: pd.DataFrame | None,
) -> None:
    """
    Render the spatial GEFS uncertainty heatmap.

    IMPORTANT:
    - Spatial colour is based on actual ensemble disagreement.
    - XGBoost bust_probability is NOT used spatially.
    - Percentile stretching makes small real spatial differences visible.
    - Hover text always reports the original physical values.
    """

    if spatial is None or spatial.empty:
        st.warning(
            "The spatial GEFS grid is not available yet. "
            "Run **Update live forecast** to generate the Chennai grid."
        )
        return

    # ------------------------------------------------------------
    # REQUIRED SPATIAL DATA
    # ------------------------------------------------------------

    required = {
        "lead_hours",
        "lat",
        "lon",
        "ensemble_mean",
        "control_forecast",
        "ensemble_spread",
    }

    missing = required - set(spatial.columns)

    if missing:
        st.error(
            "Spatial dataset is missing: "
            + ", ".join(sorted(missing))
        )
        return

    # ------------------------------------------------------------
    # LEAD SELECTION
    # ------------------------------------------------------------

    leads = sorted(
        pd.to_numeric(
            spatial["lead_hours"],
            errors="coerce"
        ).dropna().unique().tolist()
    )

    if not leads:
        st.warning("No forecast leads are available.")
        return

    st.markdown('<div class="fm-nav-box" style="padding:14px 16px;">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        selected_lead = st.selectbox(
            "Forecast lead",
            leads,
            format_func=lambda x: f"+{int(x)} hours",
            key="chennai_uncertainty_lead",
        )

    with c2:
        selected_metric = st.selectbox(
            "Heatmap variable",
            [
                "Spatial uncertainty",
                "Ensemble spread",
                "Ensemble mean rainfall",
            ],
            key="chennai_uncertainty_metric",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # SELECT DATA
    # ------------------------------------------------------------

    current = spatial[
        spatial["lead_hours"] == selected_lead
    ].copy()

    numeric_columns = [
        "lat",
        "lon",
        "ensemble_mean",
        "control_forecast",
        "ensemble_spread",
    ]

    for col in numeric_columns:
        current[col] = pd.to_numeric(
            current[col],
            errors="coerce"
        )

    current = current.dropna(
        subset=[
            "lat",
            "lon",
            "ensemble_mean",
            "ensemble_spread",
        ]
    )

    if current.empty:
        st.warning(
            f"No spatial data is available for "
            f"+{int(selected_lead)} hours."
        )
        return

    # ------------------------------------------------------------
    # CREATE SPATIAL UNCERTAINTY
    # ------------------------------------------------------------
    #
    # We deliberately calculate this from the ensemble itself.
    #
    # Relative spread:
    #
    #       max(member) - min(member)
    #       ---------------------------
    #             |mean| + 0.10
    #
    # Control divergence:
    #
    #       |control - ensemble mean|
    #       -------------------------
    #              |mean| + 0.10
    #
    # These two signals provide a much more meaningful spatial
    # uncertainty field than applying the Chennai XGBoost score
    # to every grid cell.
    # ------------------------------------------------------------

    denominator = (
        current["ensemble_mean"].abs() + 0.10
    )

    current["relative_spread"] = (
        current["ensemble_spread"] /
        denominator
    )

    current["control_divergence"] = (
        (
            current["control_forecast"] -
            current["ensemble_mean"]
        ).abs()
        /
        denominator
    )

    current["spatial_uncertainty"] = (
        0.70 * current["relative_spread"]
        +
        0.30 * current["control_divergence"]
    )

    # ------------------------------------------------------------
    # CHOOSE ACTUAL VALUE TO DISPLAY
    # ------------------------------------------------------------

    if selected_metric == "Spatial uncertainty":

        metric_values = (
            current["spatial_uncertainty"]
            .astype(float)
        )

        colorbar_title = (
            "Spatial uncertainty<br>percentile"
        )

        hover_label = "Spatial uncertainty index"

        def hover_value(row):
            return (
                f"Uncertainty index: "
                f"{row['spatial_uncertainty']:.3f}<br>"
                f"Ensemble spread: "
                f"{row['ensemble_spread']:.3f} mm<br>"
                f"Control divergence: "
                f"{row['control_divergence']:.3f}"
            )

    elif selected_metric == "Ensemble spread":

        metric_values = (
            current["ensemble_spread"]
            .astype(float)
        )

        colorbar_title = (
            "Ensemble spread<br>percentile"
        )

        hover_label = "Ensemble spread"

        def hover_value(row):
            return (
                f"Ensemble spread: "
                f"{row['ensemble_spread']:.3f} mm"
            )

    else:

        metric_values = (
            current["ensemble_mean"]
            .astype(float)
        )

        colorbar_title = (
            "Rainfall<br>percentile"
        )

        hover_label = "Ensemble mean rainfall"

        def hover_value(row):
            return (
                f"Ensemble mean: "
                f"{row['ensemble_mean']:.3f} mm"
            )

    current["metric_value"] = metric_values

    # ------------------------------------------------------------
    # HIGH-SENSITIVITY PERCENTILE STRETCH
    # ------------------------------------------------------------
    #
    # This is the important part for your "everything looks
    # the same" problem.
    #
    # We don't change the actual weather values.
    #
    # Instead:
    #
    #   actual value → percentile → colour
    #
    # So even a small but genuine difference gets a visibly
    # different colour.
    # ------------------------------------------------------------

    unique_values = current["metric_value"].nunique()

    if unique_values <= 1:

        current["display_percentile"] = 50.0
        uniform = True

    else:

        current["display_percentile"] = (
            current["metric_value"]
            .rank(
                method="average",
                pct=True
            )
            * 100.0
        )

        uniform = False

    # ------------------------------------------------------------
    # ACTUAL RANGE
    # ------------------------------------------------------------

    actual_min = float(
        current["metric_value"].min()
    )

    actual_max = float(
        current["metric_value"].max()
    )

    # ------------------------------------------------------------
    # HIGH-CONTRAST COLOUR SCALE
    # ------------------------------------------------------------

    colorscale = [
        [0.00, "#071A52"],
        [0.12, "#003B8F"],
        [0.25, "#0066CC"],
        [0.38, "#00A6D6"],
        [0.50, "#00C853"],
        [0.62, "#A8E600"],
        [0.74, "#FFD600"],
        [0.84, "#FF8C00"],
        [0.93, "#F12C2C"],
        [1.00, "#7A0019"],
    ]

    def color_for_percentile(percentile):

        f = max(
            0.0,
            min(
                1.0,
                float(percentile) / 100.0
            )
        )

        for i in range(
            len(colorscale) - 1
        ):

            p1, c1 = colorscale[i]
            p2, c2 = colorscale[i + 1]

            if f <= p2:

                ratio = (
                    (f - p1) /
                    (p2 - p1)
                    if p2 != p1
                    else 0.0
                )

                a = tuple(
                    int(
                        c1[j:j + 2],
                        16
                    )
                    for j in (1, 3, 5)
                )

                b = tuple(
                    int(
                        c2[j:j + 2],
                        16
                    )
                    for j in (1, 3, 5)
                )

                rgb = tuple(
                    round(
                        a[j]
                        +
                        (
                            b[j] - a[j]
                        )
                        * ratio
                    )
                    for j in range(3)
                )

                return (
                    "#%02X%02X%02X"
                    % rgb
                )

        return colorscale[-1][1]

    # ------------------------------------------------------------
    # BUILD GRID CELL EDGES
    # ------------------------------------------------------------

    latitudes = sorted(
        current["lat"].unique()
    )

    longitudes = sorted(
        current["lon"].unique()
    )

    def cell_edges(coords):

        coords = sorted(
            float(x)
            for x in coords
        )

        if len(coords) == 1:

            return {
                coords[0]:
                (
                    coords[0] - 0.125,
                    coords[0] + 0.125,
                )
            }

        edges = {}

        for i, value in enumerate(coords):

            if i == 0:

                spacing = (
                    coords[i + 1]
                    - coords[i]
                )

                lower = (
                    value
                    - spacing / 2
                )

            else:

                lower = (
                    value
                    -
                    (
                        coords[i]
                        -
                        coords[i - 1]
                    )
                    / 2
                )

            if i == len(coords) - 1:

                spacing = (
                    coords[i]
                    -
                    coords[i - 1]
                )

                upper = (
                    value
                    + spacing / 2
                )

            else:

                upper = (
                    value
                    +
                    (
                        coords[i + 1]
                        -
                        coords[i]
                    )
                    / 2
                )

            edges[value] = (
                lower,
                upper
            )

        return edges

    lat_edges = cell_edges(
        latitudes
    )

    lon_edges = cell_edges(
        longitudes
    )

    # ------------------------------------------------------------
    # MAP
    # ------------------------------------------------------------

    fig = go.Figure()

    for _, row in current.iterrows():

        lat = float(row["lat"])
        lon = float(row["lon"])

        percentile = float(
            row["display_percentile"]
        )

        lat0, lat1 = lat_edges[lat]
        lon0, lon1 = lon_edges[lon]

        fig.add_trace(
            go.Scattermap(
                lat=[
                    lat0,
                    lat0,
                    lat1,
                    lat1,
                    lat0,
                ],
                lon=[
                    lon0,
                    lon1,
                    lon1,
                    lon0,
                    lon0,
                ],
                mode="lines",
                fill="toself",
                fillcolor=(
                    color_for_percentile(
                        percentile
                    )
                ),
                line=dict(
                    color="rgba(255,255,255,0.35)",
                    width=1,
                ),
                hovertemplate=(
                    "<b>GEFS grid cell</b><br>"
                    f"Latitude: {lat:.2f}°<br>"
                    f"Longitude: {lon:.2f}°<br>"
                    f"{hover_value(row)}<br>"
                    f"Visual percentile: "
                    f"{percentile:.0f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # ------------------------------------------------------------
    # CHENNAI MARKER
    # ------------------------------------------------------------

    fig.add_trace(
        go.Scattermap(
            lat=[CHENNAI_LAT],
            lon=[CHENNAI_LON],
            mode="markers+text",
            marker=dict(
                size=13,
                color="#FFFFFF",
            ),
            text=["Chennai"],
            textposition="top center",
            textfont=dict(
                size=13,
                color=TEXT,
            ),
            hovertemplate=(
                "<b>Chennai</b><br>"
                "13.0827°N, 80.2707°E"
                "<extra></extra>"
            ),
            name="Chennai",
        )
    )

    # ------------------------------------------------------------
    # COLOUR BAR
    # ------------------------------------------------------------

    tick_percentiles = [
        0,
        20,
        40,
        60,
        80,
        100,
    ]

    if actual_max > actual_min:

        tick_actual = [
            actual_min
            +
            (
                actual_max
                -
                actual_min
            )
            * p / 100.0
            for p in tick_percentiles
        ]

    else:

        tick_actual = [
            actual_min
            for _ in tick_percentiles
        ]

    if selected_metric == "Ensemble spread":

        tick_labels = [
            f"{v:.2f} mm"
            for v in tick_actual
        ]

    elif selected_metric == "Ensemble mean rainfall":

        tick_labels = [
            f"{v:.2f} mm"
            for v in tick_actual
        ]

    else:

        tick_labels = [
            f"{v:.3f}"
            for v in tick_actual
        ]

    # Invisible marker trace used only to create the colourbar.

    fig.add_trace(
        go.Scattermap(
            lat=current["lat"],
            lon=current["lon"],
            mode="markers",
            marker=dict(
                size=1,
                opacity=0.01,
                color=current[
                    "display_percentile"
                ],
                cmin=0,
                cmax=100,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(
                        text=colorbar_title
                    ),
                    thickness=16,
                    len=0.65,
                    tickvals=(
                        tick_percentiles
                    ),
                    ticktext=tick_labels,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ------------------------------------------------------------
    # MAP LAYOUT
    # ------------------------------------------------------------

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(
                lat=CHENNAI_LAT,
                lon=CHENNAI_LON,
            ),
            zoom=9.8,
        ),
        height=620,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0,
        ),
        paper_bgcolor=BG,
        font=dict(
            color=TEXT
        ),
        legend=dict(
            bgcolor=BG
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # EXPLANATION
    # ------------------------------------------------------------

    if uniform:

        st.warning(
            f"All GEFS cells have the same "
            f"{hover_label.lower()} "
            f"({actual_min:.4f}). "
            "There is no genuine spatial variation "
            "in this forecast field."
        )

    else:

        st.caption(
            "High-sensitivity percentile stretching is enabled. "
            "Small real differences between GEFS cells are "
            "expanded visually without changing the underlying "
            "weather values."
        )

        st.info(
            f"Actual spatial range: "
            f"{actual_min:.4f} – "
            f"{actual_max:.4f}"
        )

# ============================================================
# TOP STATUS METRICS
# ============================================================

st.markdown('<div class="small-label" style="margin-bottom:8px;">LIVE STATUS</div>', unsafe_allow_html=True)

top1, top2, top3, top4 = st.columns(4)
with top1:
    render_card("Data source", "🟢 LIVE GEFS", f"Run: {issue_time} UTC")
with top2:
    render_card("Monitoring location", "Chennai", "13.00°N, 80.25°E")
with top3:
    render_card(
        "Forecast horizon",
        f"+{int(df['lead_hours'].max())} h",
        f"{len(df)} forecast windows",
    )
with top4:
    render_card(
        "Analytics engine",
        "GEFS + XGBoost",
        "Ensemble uncertainty + bust risk",
    )

st.divider()


# ============================================================
# SINGLE SCROLLABLE DASHBOARD
# ============================================================

if st.session_state.audience == "Forecast Officer":
    section_anchor("overview")
    page_header(
        "Operational overview",
        "Forecast Risk Overview",
        "A rapid situational-awareness page for identifying forecast windows that warrant closer review.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Peak Bust Risk", f"{peak_probability:.1f}%")
    with m2:
        st.metric("Bust-Risk Windows", len(bust_df))
    with m3:
        st.metric("High / Critical", len(high_df))
    with m4:
        st.metric(
            "Earliest Warning",
            f"+{int(earliest_bust['lead_hours'])} h"
            if earliest_bust is not None else "None",
        )

    if earliest_bust is not None:
        st.markdown(
            f"""
            <div class="warning-card">
                <div class="warning-title">Forecast Review Signal</div>
                <div class="warning-text">
                    A potential forecast bust is detected at
                    <b>+{int(earliest_bust['lead_hours'])} hours</b>.
                    This is an assessment signal, not an official warning.
                </div>
                <br>
                <b>Bust Risk Score:</b> {float(earliest_bust['bust_probability']):.1f}%
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Risk:</b> {esc(earliest_bust['risk_level'])}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Diagnosis:</b> {esc(earliest_bust['diagnosis'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("No forecast windows currently exceed the bust detection threshold.")

    page_header(
        "Risk evolution",
        "Bust Risk Timeline",
        "Higher values indicate a greater model-estimated likelihood of forecast instability.",
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["lead_hours"],
            y=df["bust_probability"],
            mode="lines+markers",
            name="Bust Risk",
            line=dict(width=3),
            marker=dict(size=7),
        )
    )
    fig.add_hline(
        y=BUST_THRESHOLD,
        line_dash="dash",
        annotation_text="Bust threshold",
    )
    fig.add_hline(
        y=CRITICAL_THRESHOLD,
        line_dash="dot",
        annotation_text="Critical",
    )
    fig.update_layout(
        xaxis_title="Forecast lead time (hours)",
        yaxis_title="Bust risk score (%)",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(chart_layout(fig, 480), use_container_width=True)


    section_anchor("chennai-heatmap")
    page_header(
        "Chennai monitoring",
        "Chennai Heatmap",
        "Compare forecast risk, ensemble disagreement and deterministic-versus-ensemble divergence across the forecast horizon.",
    )

    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("Chennai", "13.00°N, 80.25°E")
    with h2:
        st.metric("Peak bust risk", f"{peak_probability:.1f}%")
    with h3:
        st.metric("Peak ensemble spread", f"{float(df['ensemble_spread'].max()):.2f} mm")

    render_chennai_heatmap(df, spatial_df)

    page_header(
        "Chennai forecast data",
        "Selected Window Details",
        "The values behind the heatmap, sorted by forecast lead time.",
    )
    heatmap_table = pd.DataFrame({
        "Lead": "+" + df["lead_hours"].astype(int).astype(str) + "h",
        "Control (mm)": df["control_forecast"].round(2),
        "Ensemble Mean (mm)": df["ensemble_mean"].round(2),
        "Ensemble Spread (mm)": df["ensemble_spread"].round(2),
        "Bust Risk (%)": df["bust_probability"].round(1),
        "Risk": df["risk_level"],
    })
    st.dataframe(heatmap_table, use_container_width=True, hide_index=True)


    section_anchor("forecast-analysis")
    page_header(
        "Forecast comparison",
        "Deterministic Forecast vs Ensemble",
        "Divergence between the control forecast and ensemble consensus is a key forecast-instability signal.",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["lead_hours"],
            y=df["control_forecast"],
            mode="lines+markers",
            name="Control Forecast",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["lead_hours"],
            y=df["ensemble_mean"],
            mode="lines+markers",
            name="Ensemble Mean",
            line=dict(width=3),
        )
    )
    fig.update_layout(
        xaxis_title="Forecast lead time (hours)",
        yaxis_title="Rainfall (mm / 3h)",
    )
    st.plotly_chart(chart_layout(fig), use_container_width=True)

    page_header(
        "Uncertainty",
        "Ensemble Spread",
        "Larger spread means the ensemble members disagree more strongly about possible outcomes.",
    )
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=df["lead_hours"],
            y=df["ensemble_spread"],
            name="Ensemble Spread",
        )
    )
    fig2.update_layout(
        xaxis_title="Forecast lead time (hours)",
        yaxis_title="Spread (mm / 3h)",
    )
    st.plotly_chart(chart_layout(fig2, 390), use_container_width=True)


    section_anchor("data-explorer")
    page_header(
        "Raw forecast timeline",
        "Explore the Data",
        "Filter the forecast windows and inspect the model inputs, ensemble statistics and risk outputs.",
    )

    selected_risks = st.multiselect(
        "Risk levels",
        sorted(df["risk_level"].dropna().astype(str).unique()),
        default=sorted(df["risk_level"].dropna().astype(str).unique()),
    )
    min_lead, max_lead = int(df["lead_hours"].min()), int(df["lead_hours"].max())
    lead_range = st.slider(
        "Forecast lead-time range (hours)",
        min_value=min_lead,
        max_value=max_lead,
        value=(min_lead, max_lead),
    )

    filtered = df[
        df["risk_level"].astype(str).isin(selected_risks)
        & df["lead_hours"].between(lead_range[0], lead_range[1])
    ].copy()

    st.caption(f"Showing {len(filtered)} of {len(df)} forecast windows.")

    explorer_df = filtered[
        [
            "issue_time", "target_time", "lead_hours",
            "control_forecast", "ensemble_mean", "ensemble_median",
            "ensemble_min", "ensemble_max", "ensemble_spread",
            "bust_probability", "predicted_bust", "risk_level",
            "primary_signal", "diagnosis",
        ]
    ].copy()
    explorer_df["bust_probability"] = explorer_df["bust_probability"].round(1)
    for col in [
        "control_forecast", "ensemble_mean", "ensemble_median",
        "ensemble_min", "ensemble_max", "ensemble_spread"
    ]:
        explorer_df[col] = explorer_df[col].round(2)

    st.dataframe(explorer_df, use_container_width=True, hide_index=True, height=560)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name="forecastmitr_chennai_filtered.csv",
        mime="text/csv",
    )


    section_anchor("diagnostics")
    page_header(
        "Explainability",
        "Why Did ForecastMitr Flag It?",
        "The peak-risk window is decomposed into interpretable forecast signals.",
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        render_card(
            "Peak risk window",
            f"+{int(peak_row['lead_hours'])} h",
            f"Bust risk: {peak_probability:.1f}%",
        )
    with p2:
        render_card(
            "Control vs ensemble",
            f"{float(peak_row['control_forecast']):.2f} mm",
            f"Ensemble mean: {float(peak_row['ensemble_mean']):.2f} mm",
        )
    with p3:
        render_card(
            "Primary diagnosis",
            str(peak_row["diagnosis"]),
            f"Spread: {float(peak_row['ensemble_spread']):.2f} mm",
        )

    page_header(
        "Forecast windows",
        "Detailed Diagnostics",
        "Review individual lead times, risk scores and the model's reported diagnosis.",
    )

    display_df = pd.DataFrame({
        "Lead": "+" + df["lead_hours"].astype(int).astype(str) + "h",
        "Control (mm)": df["control_forecast"].round(2),
        "Ensemble Mean (mm)": df["ensemble_mean"].round(2),
        "Spread (mm)": df["ensemble_spread"].round(2),
        "Bust Risk": df["bust_probability"].round(1).astype(str) + "%",
        "Risk": df["risk_level"],
        "Primary Signal": df["primary_signal"],
        "Diagnosis": df["diagnosis"],
    })
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=560,
    )


    section_anchor("methodology")
    page_header(
        "System methodology",
        "How ForecastMitr Works",
        "A transparent pipeline from ensemble retrieval to explainable bust-risk assessment.",
    )
    st.markdown(
        """
        **1. Live GEFS retrieval**  
        Retrieve the latest available GEFS cycle and ensemble forecast information.

        **2. Ensemble analysis**  
        Calculate mean, median, minimum, maximum and spread to quantify uncertainty.

        **3. Deterministic vs ensemble comparison**  
        Compare the control forecast against ensemble consensus to identify instability.

        **4. Bust-risk estimation**  
        Use the trained XGBoost model to produce a Bust Risk Score.

        **5. Explainable diagnosis**  
        Surface the strongest forecast signals behind the score.

        **6. Early review signal**  
        Highlight the earliest future window crossing the configured bust-risk threshold.

        > ForecastMitr is a decision-support layer. Official IMD forecasts and warnings remain authoritative.
        """
    )

    st.markdown("### ForecastMitr Processing Pipeline")
    st.caption("End-to-end flow from live GEFS retrieval to the dashboard outputs.")
    render_pipeline_flowchart()


# ============================================================
# PUBLIC
# ============================================================

else:
    section_anchor("simple-forecast")
    page_header(
        "Public summary",
        "How stable is the forecast?",
        "A simplified view of ForecastMitr's uncertainty assessment.",
    )

    if peak_probability >= CRITICAL_THRESHOLD:
        label = "Very high forecast uncertainty"
        icon = "🔴"
        explanation = (
            "The model sees strong disagreement in possible weather outcomes. "
            "The forecast may change as new information arrives."
        )
    elif peak_probability >= BUST_THRESHOLD:
        label = "Forecast may change"
        icon = "🟠"
        explanation = (
            "Different forecast members disagree more than usual. "
            "The forecast may change as new observations arrive."
        )
    else:
        label = "Forecast looks relatively stable"
        icon = "🟢"
        explanation = (
            "The current ensemble shows lower bust-risk signals. "
            "Continue following the latest official forecast."
        )

    st.markdown(
        f"""
        <div class="fm-public">
            <div class="fm-eyebrow">Simple forecast confidence view</div>
            <div class="fm-risk">{peak_probability:.0f}%</div>
            <div style="font-size:18px;font-weight:800;color:var(--fm-text);">
                {icon} {label}
            </div>
            <div class="fm-explain">{esc(explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        render_card(
            "Highest-risk window",
            f"+{int(peak_row['lead_hours'])} h",
            "Time from the forecast run",
        )
    with c2:
        render_card(
            "Forecast horizon",
            f'+{int(df["lead_hours"].max())} h',
            "Current analysis window",
        )

    st.info(
        "For safety, travel or severe-weather decisions, always check the "
        "latest official IMD forecast and warnings."
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["lead_hours"],
            y=df["bust_probability"],
            mode="lines+markers",
            name="Forecast uncertainty",
            line=dict(width=3),
        )
    )
    fig.add_hline(
        y=BUST_THRESHOLD,
        line_dash="dash",
        annotation_text="Higher uncertainty",
    )
    fig.update_layout(
        xaxis_title="Hours ahead",
        yaxis_title="Uncertainty / bust-risk score (%)",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(chart_layout(fig, 430), use_container_width=True)


    section_anchor("chennai-heatmap")
    page_header(
        "Public Chennai view",
        "Chennai Forecast Heatmap",
        "A simple visual of how forecast uncertainty changes across the available Chennai forecast windows.",
    )
    render_chennai_heatmap(df, spatial_df)


    section_anchor("what-it-means")
    page_header(
        "Plain-language explanation",
        "What does ForecastMitr look for?",
        "ForecastMitr assesses forecast confidence; it does not replace the weather forecast itself.",
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown("### 🌦️ Forecast\nWhat the current weather model predicts for a future time window.")
    with b:
        st.markdown("### 👥 Ensemble\nMany slightly different model runs used to understand possible outcomes.")
    with c:
        st.markdown("### 📏 Uncertainty\nHow much those possible outcomes disagree with one another.")

    st.markdown("### Why this matters")
    st.write(
        "A forecast can look precise while the underlying model runs disagree. "
        "ForecastMitr highlights those situations so users can pay more attention "
        "to the latest official forecast."
    )
    st.success(
        "Simple rule: **higher disagreement → higher uncertainty → check the latest official forecast.**"
    )


    section_anchor("methodology")
    page_header(
        "Methodology",
        "How ForecastMitr Works",
        "A simple explanation of the technical pipeline.",
    )
    st.markdown(
        """
        1. **Retrieve** current GEFS ensemble information.
        2. **Compare** multiple ensemble members and the control forecast.
        3. **Measure** ensemble spread and forecast disagreement.
        4. **Estimate** a bust-risk score using the trained XGBoost model.
        5. **Explain** the strongest signals behind the score.

        ForecastMitr is a **decision-support and forecast-assessment tool**, not a replacement for official IMD forecasts or warnings.
        """
    )

    st.markdown("### ForecastMitr Processing Pipeline")
    st.caption("End-to-end flow from live GEFS retrieval to the dashboard outputs.")
    render_pipeline_flowchart()


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div class="footer">
        <b>ForecastMitr</b> • SIH26079 • Weather Bust Detection<br>
        Decision-support for forecast assessment • Official IMD forecasts and warnings remain authoritative
    </div>
    """,
    unsafe_allow_html=True,
)