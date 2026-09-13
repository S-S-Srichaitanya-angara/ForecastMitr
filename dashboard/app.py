import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TIMELINE_FILE = ROOT / "data" / "forecastmitr_live_timeline.csv"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_live_pipeline.py"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ForecastMitr",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 10px 0 15px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 18px;
        opacity: 0.70;
    }

    .status-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 18px;
        background: rgba(128,128,128,0.06);
        height: 100%;
    }

    .status-title {
        font-size: 13px;
        opacity: 0.65;
        margin-bottom: 6px;
        letter-spacing: 0.4px;
    }

    .status-value {
        font-size: 23px;
        font-weight: 700;
    }

    .warning-card {
        border: 2px solid #ff4b4b;
        border-radius: 14px;
        padding: 20px;
        background: rgba(255,75,75,0.07);
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .warning-title {
        font-size: 24px;
        font-weight: 750;
    }

    .warning-text {
        font-size: 16px;
        margin-top: 8px;
    }

    .info-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 20px;
        background: rgba(128,128,128,0.05);
        margin-top: 10px;
        min-height: 145px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .small-label {
        font-size: 13px;
        opacity: 0.65;
    }

    .big-number {
        font-size: 30px;
        font-weight: 750;
    }

    .diagnosis {
        font-size: 18px;
        font-weight: 700;
    }

    .live-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        padding-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LIVE UPDATE CONTROL
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🌦️ ForecastMitr</div>
        <div class="hero-subtitle">
            AI-Powered Weather Forecast Bust Detection
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

update_col, status_col = st.columns([1.0, 2.5])

with update_col:
    update_clicked = st.button(
        "🔄 UPDATE LIVE FORECAST",
        type="primary",
        use_container_width=True,
        help="Fetch the latest available GEFS cycle and regenerate the ForecastMitr timeline.",
    )

with status_col:
    if TIMELINE_FILE.exists():
        modified_time = pd.Timestamp(TIMELINE_FILE.stat().st_mtime, unit="s")
        st.caption(
            f"Last local timeline update: **{modified_time.strftime('%d %b %Y, %H:%M:%S')}**"
        )
    else:
        st.caption("No live timeline has been generated yet.")


if update_clicked:
    if not PIPELINE_SCRIPT.exists():
        st.error(f"Live pipeline not found: `{PIPELINE_SCRIPT}`")
        st.stop()

    with st.status("Updating ForecastMitr from live GEFS...", expanded=True) as status:
        st.write("Searching for the latest available GEFS cycle...")
        result = subprocess.run(
            [sys.executable, str(PIPELINE_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            # Show only the most useful final part during normal operation.
            lines = result.stdout.strip().splitlines()
            tail = "\n".join(lines[-35:])
            st.code(tail, language="text")

        if result.returncode != 0:
            if result.stderr:
                st.code(result.stderr[-5000:], language="text")
            status.update(
                label="Live update failed",
                state="error",
                expanded=True,
            )
            st.stop()

        status.update(
            label="Live forecast updated successfully",
            state="complete",
            expanded=False,
        )

    st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

if not TIMELINE_FILE.exists():
    st.warning(
        "Live timeline data is not available yet. "
        "Click **UPDATE LIVE FORECAST** to download GEFS data and generate it."
    )
    st.stop()


df = pd.read_csv(TIMELINE_FILE)


# ============================================================
# COLUMN VALIDATION
# ============================================================

required_columns = [
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

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

numeric_columns = [
    "lead_hours",
    "control_forecast",
    "ensemble_mean",
    "ensemble_median",
    "ensemble_min",
    "ensemble_max",
    "ensemble_spread",
    "bust_probability",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Engine stores the bust score as 0–1; UI displays 0–100%.
if df["bust_probability"].max() <= 1.5:
    df["bust_probability"] *= 100.0

df = df.dropna(subset=["lead_hours", "bust_probability"]).sort_values(
    "lead_hours"
).reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

bust_threshold = 69.0

bust_df = df[df["bust_probability"] >= bust_threshold]

high_df = df[
    df["risk_level"]
    .astype(str)
    .str.upper()
    .isin(["HIGH", "CRITICAL"])
]

peak_row = df.loc[df["bust_probability"].idxmax()]

earliest_bust = bust_df.iloc[0] if len(bust_df) else None


# ============================================================
# LIVE STATUS
# ============================================================

issue_time = str(df.iloc[0]["issue_time"])

st.markdown(
    f"""
    <div class="status-card">
        <div class="status-title">DATA SOURCE</div>
        <div class="status-value">🟢 LIVE GEFS ENSEMBLE</div>
        <div class="small-label">
            Latest processed forecast run: {issue_time} UTC
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOCATION / SYSTEM INFO
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">MONITORING LOCATION</div>
            <div class="status-value">📍 Chennai</div>
            <div class="small-label">13.00°N, 80.25°E</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">FORECAST HORIZON</div>
            <div class="status-value">+{int(df["lead_hours"].max())} hours</div>
            <div class="small-label">{len(df)} forecast windows analysed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">ANALYTICS ENGINE</div>
            <div class="status-value">GEFS + XGBoost</div>
            <div class="small-label">
                Ensemble disagreement + bust-risk analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# KEY METRICS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Forecast Risk Overview</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Peak Bust Risk", f"{peak_row['bust_probability']:.1f}%")

with m2:
    st.metric("Predicted Bust Windows", len(bust_df))

with m3:
    st.metric("High / Critical", len(high_df))

with m4:
    st.metric(
        "Earliest Warning",
        f"+{int(earliest_bust['lead_hours'])} h"
        if earliest_bust is not None
        else "None",
    )


# ============================================================
# EARLIEST WARNING
# ============================================================

if earliest_bust is not None:
    lead = int(earliest_bust["lead_hours"])
    probability = float(earliest_bust["bust_probability"])
    risk = str(earliest_bust["risk_level"])
    diagnosis = str(earliest_bust["diagnosis"])

    warning_html = f"""
    <div class="warning-card">
        <div class="warning-title">🚨 FORECAST BUST WARNING</div>
        <div class="warning-text">
            ForecastMitr detected a potential forecast bust beginning at
            <b>+{lead} hours</b>.
        </div>
        <br>
        <b>Bust Risk Score:</b> {probability:.1f}%
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>Risk:</b> {risk}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>Diagnosis:</b> {diagnosis}
    </div>
    """

    st.markdown(warning_html, unsafe_allow_html=True)
else:
    st.success(
        "No forecast windows currently exceed the bust detection threshold."
    )


# ============================================================
# RISK TIMELINE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Forecast Bust Risk Timeline</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["lead_hours"],
        y=df["bust_probability"],
        mode="lines+markers",
        name="Bust Risk Score",
        line=dict(width=4),
        marker=dict(size=7),
        hovertemplate=(
            "<b>Lead:</b> +%{x}h"
            "<br><b>Bust Risk:</b> %{y:.1f}%"
            "<extra></extra>"
        ),
    )
)

fig.add_hline(
    y=bust_threshold,
    line_dash="dash",
    annotation_text="Bust detection threshold",
)

fig.add_hline(
    y=85,
    line_dash="dot",
    annotation_text="Critical risk",
)

fig.update_layout(
    height=480,
    xaxis_title="Forecast Lead Time (hours)",
    yaxis_title="Bust Risk Score (%)",
    yaxis=dict(range=[0, 100]),
    hovermode="x unified",
    margin=dict(l=40, r=30, t=30, b=50),
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# FORECAST DISAGREEMENT
# ============================================================

st.markdown(
    '<div class="section-title">🌧️ Deterministic Forecast vs Ensemble</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Large divergence between the control forecast and ensemble consensus "
    "is a key indicator of forecast instability."
)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df["lead_hours"],
        y=df["control_forecast"],
        mode="lines+markers",
        name="Control Forecast",
        line=dict(width=3),
    )
)

fig2.add_trace(
    go.Scatter(
        x=df["lead_hours"],
        y=df["ensemble_mean"],
        mode="lines+markers",
        name="Ensemble Mean",
        line=dict(width=3),
    )
)

fig2.update_layout(
    height=430,
    xaxis_title="Forecast Lead Time (hours)",
    yaxis_title="Rainfall (mm / 3h)",
    hovermode="x unified",
    margin=dict(l=40, r=30, t=30, b=50),
)

st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# ENSEMBLE UNCERTAINTY
# ============================================================

st.markdown(
    '<div class="section-title">📊 Ensemble Uncertainty</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Higher ensemble spread indicates greater disagreement among "
    "possible atmospheric outcomes."
)

fig3 = go.Figure()

fig3.add_trace(
    go.Bar(
        x=df["lead_hours"],
        y=df["ensemble_spread"],
        name="Ensemble Spread",
        hovertemplate=(
            "<b>Lead:</b> +%{x}h"
            "<br><b>Spread:</b> %{y:.2f} mm"
            "<extra></extra>"
        ),
    )
)

fig3.update_layout(
    height=370,
    xaxis_title="Forecast Lead Time (hours)",
    yaxis_title="Spread (mm / 3h)",
    margin=dict(l=40, r=30, t=30, b=50),
)

st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# PEAK RISK EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Why Did ForecastMitr Flag It?</div>',
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-label">PEAK RISK WINDOW</div>
            <div class="big-number">+{int(peak_row['lead_hours'])}h</div>
            <br>
            <b>Bust Risk Score</b><br>
            {peak_row['bust_probability']:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-label">FORECAST DISAGREEMENT</div>
            <div class="big-number">{peak_row['control_forecast']:.2f} mm</div>
            <br>
            Control forecast
            <br><br>
            Ensemble mean:
            <b>{peak_row['ensemble_mean']:.2f} mm</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-label">DIAGNOSIS</div>
            <div class="diagnosis">{peak_row['diagnosis']}</div>
            <br>
            Ensemble spread:
            <b>{peak_row['ensemble_spread']:.2f} mm</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TIMELINE TABLE
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Forecast Window Diagnostics</div>',
    unsafe_allow_html=True,
)

display_df = pd.DataFrame()

display_df["Lead"] = (
    "+" + df["lead_hours"].astype(int).astype(str) + "h"
)
display_df["Control (mm)"] = df["control_forecast"].round(2)
display_df["Ensemble Mean (mm)"] = df["ensemble_mean"].round(2)
display_df["Spread (mm)"] = df["ensemble_spread"].round(2)
display_df["Bust Risk"] = (
    df["bust_probability"].round(1).astype(str) + "%"
)
display_df["Risk"] = df["risk_level"]
display_df["Diagnosis"] = df["diagnosis"]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=520,
)


# ============================================================
# HOW FORECASTMITR WORKS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ How ForecastMitr Works</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    **1. Live GEFS Retrieval**

    ForecastMitr searches for the latest available GEFS cycle and retrieves
    multiple ensemble members for Chennai and future forecast windows.

    **2. Ensemble Analysis**

    It calculates ensemble mean, median, minimum, maximum and spread to
    quantify forecast uncertainty.

    **3. Deterministic vs Ensemble Comparison**

    The control forecast is compared against the ensemble consensus to detect
    forecast instability.

    **4. AI Bust Detection**

    An XGBoost model analyses the forecast characteristics and estimates a
    **Bust Risk Score**.

    **5. Explainable Diagnosis**

    ForecastMitr identifies signals such as possible false alarms, possible
    missed events, high forecast intensity and ensemble uncertainty.

    **6. Early Warning**

    The system identifies the earliest future window where the forecast enters
    the bust-risk region.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>ForecastMitr</b> • SIH26079 • Weather Bust Detection<br>
        Live GEFS Ensemble Analysis + XGBoost + Explainable Diagnostics
    </div>
    """,
    unsafe_allow_html=True,
)
