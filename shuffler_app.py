
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Shuffler Motor Health Monitor",
    page_icon="🔧",
    layout="wide"
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MODE_NAMES = {
    0.0: "Normal",
    1.0: "Endurance", 
    2.0: "Maintenance"
}

MODE_COLORS = {
    0.0: "#2196F3",   # Blue
    1.0: "#FF9800",   # Orange
    2.0: "#4CAF50"    # Green
}

SEQ_NAMES = {
    3000: "Standby",
    3100: "Ready/Idle",
    3110: "Forward Move",
    3115: "Forward Settle",
    3120: "Forward Hold",
    3125: "Return Move",
    3130: "Return Settle"
}

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_all_data():
    try:
        df = pd.read_csv("all_cycles_4days.csv")
        df["start_time"] = pd.to_datetime(
            df["start_time"]
        )
        return df
    except:
        return None

@st.cache_data
def load_baselines():
    try:
        with open("mode_baselines.json") as f:
            raw = json.load(f)
        return {float(k): v for k,v in raw.items()}
    except:
        return {}

def parse_csv(content, day_label):
    try:
        lines = content.split("\n")
        col_names = lines[3].strip().split(",")
        df_temp = pd.read_csv(
            io.StringIO("\n".join(lines[4:])),
            names=col_names,
            on_bad_lines="skip"
        )
        required = ["D3203","D3223","D3224","D3238"]
        if not all(c in df_temp.columns
                  for c in required):
            return None
        df_temp = df_temp.dropna(
            subset=["D3203"]
        ).reset_index(drop=True)
        df_clean = df_temp[[
            "TIME (UTC+09:00)",
            "D3203","D3223","D3224","D3238"
        ]].copy()
        df_clean.columns = [
            "timestamp","sequence",
            "torque_raw","velocity","run_mode"
        ]
        df_clean["timestamp"] = pd.to_datetime(
            df_clean["timestamp"], errors="coerce"
        )
        df_clean["torque_nm"] = (
            df_clean["torque_raw"] * 0.00064
        )
        df_clean["day"] = day_label
        return df_clean
    except:
        return None

def load_uploaded(files, day_label):
    all_data = []
    for f in files:
        content = f.read().decode(
            "utf-8", errors="ignore"
        )
        df = parse_csv(content, day_label)
        if df is not None:
            all_data.append(df)
    if not all_data:
        return None
    return pd.concat(
        all_data, ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

def load_folder(folder_path, day_label):
    all_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".CSV") or f.endswith(".csv")
    ])
    all_data = []
    for filename in all_files:
        fp = os.path.join(folder_path, filename)
        try:
            with open(fp,"r",
                      encoding="utf-8",
                      errors="ignore") as f:
                content = f.read()
            df = parse_csv(content, day_label)
            if df is not None:
                all_data.append(df)
        except:
            continue
    if not all_data:
        return None
    return pd.concat(
        all_data, ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

# ─────────────────────────────────────────
# CYCLE EXTRACTION
# ─────────────────────────────────────────
def extract_cycles(df_all, day_label):
    active = [3110.0,3115.0,3120.0,3125.0,3130.0]
    df_active = df_all[
        df_all["sequence"].isin(active)
    ].copy().reset_index(drop=True)

    cycles = []
    cycle_id = 0
    seq = df_active["sequence"].values
    n = len(seq)
    i = 0

    while i < n - 1:
        if seq[i] == 3110.0:
            fwd_start = i
            j = i + 1
            while j < n and seq[j] == 3110.0:
                j += 1
            if j >= n or seq[j] != 3115.0:
                i = j
                continue
            settle_start = j
        elif seq[i] == 3115.0 and (
            i == 0 or seq[i-1] == 3130.0
        ):
            fwd_start = i
            settle_start = i
        else:
            i += 1
            continue

        k = settle_start + 1
        while k < n and seq[k] == 3115.0:
            k += 1
        if k < n and seq[k] == 3120.0:
            while k < n and seq[k] == 3120.0:
                k += 1
        if k >= n or seq[k] != 3125.0:
            i = k
            continue
        return_start = k
        m = k + 1
        while m < n and seq[m] == 3125.0:
            m += 1
        if m >= n or seq[m] != 3130.0:
            i = m
            continue
        cycle_end = m

        fwd = df_active.iloc[fwd_start:settle_start]
        settle = df_active.iloc[
            settle_start:return_start]
        ret = df_active.iloc[return_start:cycle_end]

        if len(settle) < 3 or len(ret) < 3:
            i = cycle_end + 1
            continue

        cycles.append({
            "cycle_id": cycle_id,
            "day": day_label,
            "start_time": df_active.iloc[
                fwd_start]["timestamp"],
            "run_mode": df_active.iloc[
                fwd_start]["run_mode"],
            "fwd_torque_max": fwd["torque_nm"].max()
                if len(fwd)>0 else np.nan,
            "fwd_torque_mean": fwd["torque_nm"].mean()
                if len(fwd)>0 else np.nan,
            "fwd_torque_std": fwd["torque_nm"].std()
                if len(fwd)>0 else np.nan,
            "fwd_velocity_max": fwd["velocity"].max()
                if len(fwd)>0 else np.nan,
            "fwd_duration": len(fwd),
            "settle_torque_mean": settle[
                "torque_nm"].mean(),
            "settle_torque_std": settle[
                "torque_nm"].std(),
            "settle_duration": len(settle),
            "ret_torque_min": ret["torque_nm"].min(),
            "ret_torque_mean": ret["torque_nm"].mean(),
            "ret_torque_std": ret["torque_nm"].std(),
            "ret_velocity_min": ret["velocity"].min(),
            "ret_duration": len(ret),
            "total_duration": cycle_end - fwd_start,
        })
        cycle_id += 1
        i = cycle_end + 1

    return pd.DataFrame(cycles)

# ─────────────────────────────────────────
# ANALYSIS PER MODE
# ─────────────────────────────────────────
def analyse_one_mode(new_cycles_mode,
                     historical_mode,
                     baseline_info,
                     day_label, mode):
    """
    Analyse one specific mode
    Compare new data to same mode historical
    """
    if len(new_cycles_mode) == 0:
        return None

    b_mean = baseline_info["baseline_mean"]
    b_std  = baseline_info["baseline_std"]
    alert  = baseline_info["alert_threshold"]
    warn   = baseline_info["warning_threshold"]

    ret_mean = new_cycles_mode[
        "ret_torque_min"].mean()
    fwd_mean = new_cycles_mode[
        "fwd_torque_max"].mean()

    # Health score
    health = max(0, min(100,
        (ret_mean - alert) /
        (b_mean - alert) * 100
    ))

    # Trend — same mode only
    if historical_mode is not None and        len(historical_mode) > 0:
        daily = historical_mode.groupby("day")[
            "ret_torque_min"
        ].mean()
        # Add current day
        new_val = pd.Series(
            {day_label: ret_mean}
        )
        daily = pd.concat([daily, new_val])
        daily = daily[~daily.index.duplicated(
            keep="last")]
        
        t_vals = daily.values
        t_nums = list(range(len(t_vals)))
        if len(t_vals) >= 2:
            z = np.polyfit(t_nums, t_vals, 1)
            slope = z[0]
            days_to = int(
                (alert - ret_mean) / slope
            ) if slope < 0 else 999
        else:
            slope = 0
            days_to = 999
    else:
        slope = 0
        days_to = 999

    # Status
    if ret_mean < alert:
        status = "CRITICAL"
    elif ret_mean < warn:
        status = "WARNING"
    elif health < 80:
        status = "CAUTION"
    else:
        status = "NORMAL"

    # Sequence analysis
    # Get baseline cycles for this mode
    if historical_mode is not None:
        base_cycles = historical_mode[
            historical_mode["day"] ==
            baseline_info["baseline_day"]
        ]
    else:
        base_cycles = new_cycles_mode

    seq_analysis = {}
    for seq_key, metrics in [
        ("3110",[
            ("torque_mean","fwd_torque_mean",
             "fwd_torque_mean"),
            ("torque_max","fwd_torque_max",
             "fwd_torque_max"),
            ("duration","fwd_duration",
             "fwd_duration"),
        ]),
        ("3115",[
            ("duration","settle_duration",
             "settle_duration"),
        ]),
        ("3125",[
            ("torque_min","ret_torque_min",
             "ret_torque_min"),
            ("velocity","ret_velocity_min",
             "ret_velocity_min"),
            ("duration","ret_duration",
             "ret_duration"),
        ]),
    ]:
        names = {
            "3110":"Forward Move",
            "3115":"Forward Settle",
            "3125":"Return Move"
        }
        alerts_seq = []
        for label, b_col, c_col in metrics:
            if len(base_cycles) == 0:
                continue
            b = float(base_cycles[b_col].mean())
            c = float(new_cycles_mode[c_col].mean())
            if b == 0:
                continue
            pct = ((c-b)/abs(b))*100
            if abs(pct) > 10:
                alerts_seq.append({
                    "metric": label,
                    "baseline": round(b,4),
                    "current": round(c,4),
                    "change_pct": round(pct,1)
                })
        seq_analysis[seq_key] = {
            "name": names[seq_key],
            "status": "STRESS"
                if alerts_seq else "NORMAL",
            "alerts": alerts_seq
        }

    return {
        "mode": mode,
        "mode_name": MODE_NAMES.get(mode,"Unknown"),
        "cycles": len(new_cycles_mode),
        "health_score": round(health,1),
        "status": status,
        "ret_torque": round(ret_mean,4),
        "fwd_torque": round(fwd_mean,4),
        "alert_threshold": round(alert,4),
        "warning_threshold": round(warn,4),
        "days_to_alert": days_to,
        "trend_per_day": round(slope,6),
        "sequence_analysis": seq_analysis,
        "baseline_day": baseline_info[
            "baseline_day"],
        "baseline_mean": round(b_mean,4),
    }

# ─────────────────────────────────────────
# DISPLAY ONE MODE RESULT
# ─────────────────────────────────────────
def display_mode_result(result, mode_color):
    if result is None:
        st.info("No data for this mode")
        return

    status = result["status"]
    score = result["health_score"]

    icon_map = {
        "CRITICAL":"🔴",
        "WARNING":"🟡",
        "CAUTION":"🟡",
        "NORMAL":"🟢"
    }
    color_map = {
        "CRITICAL":"red",
        "WARNING":"orange",
        "CAUTION":"orange",
        "NORMAL":"green"
    }

    st.markdown(
        f"**{result['cycles']} cycles analysed** "
        f"| Baseline: {result['baseline_day']} "
        f"| Baseline torque: "
        f"{result['baseline_mean']} Nm"
    )

    c1,c2,c3 = st.columns(3)

    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text":"Health Score"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":
                       color_map[status]},
                "steps":[
                    {"range":[0,30],
                     "color":"lightcoral"},
                    {"range":[30,70],
                     "color":"lightyellow"},
                    {"range":[70,100],
                     "color":"lightgreen"},
                ]
            }
        ))
        fig.update_layout(height=220)
        st.plotly_chart(fig,
                        use_container_width=True)

    with c2:
        st.metric("Status",
                  f"{icon_map[status]} {status}")
        st.metric("Return Torque",
                  f"{result['ret_torque']} Nm",
                  delta=f"{result['ret_torque']-result['alert_threshold']:.4f} to alert")
        st.metric("Days to Alert",
                  result["days_to_alert"]
                  if result["days_to_alert"]<999
                  else "Safe ✅")

    with c3:
        st.metric("Alert Threshold",
                  f"{result['alert_threshold']} Nm")
        st.metric("Warning Threshold",
                  f"{result['warning_threshold']} Nm")
        st.metric("Trend/Day",
                  f"{result['trend_per_day']} Nm")

    # Sequence analysis
    st.markdown("**Sequence Level:**")
    s1,s2,s3 = st.columns(3)
    for col, seq_key in zip([s1,s2,s3],
                             ["3110","3115","3125"]):
        with col:
            s = result["sequence_analysis"][seq_key]
            icon = "⚠️" if s["status"]=="STRESS"                    else "✅"
            st.markdown(
                f"{icon} **Seq {seq_key}**\n"
                f"{s['name']}"
            )
            if s["alerts"]:
                for a in s["alerts"]:
                    d = "↑" if a["change_pct"]>0                         else "↓"
                    st.caption(
                        f"{a['metric']}: "
                        f"{a['change_pct']:+.1f}% {d}"
                    )

    # Recommendation
    if status == "CRITICAL":
        st.error(
            "🔴 **STOP MACHINE** — "
            "Torque exceeds critical threshold. "
            "Immediate inspection required."
        )
    elif status == "WARNING":
        st.warning(
            f"🟡 **Schedule maintenance within "
            f"{result['days_to_alert']} days** — "
            "Check belt, lubrication, alignment."
        )
    elif status == "CAUTION":
        st.warning(
            "🟡 **Monitor daily** — "
            "Early signs of increased load."
        )
    else:
        st.success(
            "🟢 **Healthy** — "
            "All parameters within normal range."
        )

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
st.title("🔧 Shuffler Motor Health Monitor")
st.markdown(
    "### Predictive Maintenance — Blackbox Factories"
)
st.markdown("---")

# Load data
historical = load_all_data()
baselines = load_baselines()

if historical is None:
    st.error(
        "❌ Historical data not found. "
        "Upload all_cycles_4days.csv to GitHub."
    )
    st.stop()

# ── SIDEBAR ──
st.sidebar.header("📁 Load New Data")
day_label = st.sidebar.text_input(
    "Date Label", placeholder="e.g. May 01"
)
input_method = st.sidebar.radio(
    "Data Source",
    ["📤 Upload CSV Files",
     "📂 Enter Folder Path"]
)

df_raw = None

if input_method == "📤 Upload CSV Files":
    uploaded = st.sidebar.file_uploader(
        "Select all CSV files",
        type=["csv","CSV"],
        accept_multiple_files=True
    )
    if uploaded and day_label:
        with st.spinner("Loading files..."):
            df_raw = load_uploaded(
                uploaded, day_label
            )
        if df_raw is not None:
            st.sidebar.success(
                f"✅ {len(uploaded)} files loaded"
            )
else:
    folder = st.sidebar.text_input(
        "Folder Path",
        placeholder="/path/to/csv/folder"
    )
    if folder and day_label:
        if os.path.exists(folder):
            with st.spinner("Loading folder..."):
                df_raw = load_folder(
                    folder, day_label
                )
            if df_raw is not None:
                n = len([f for f in
                         os.listdir(folder)
                         if f.endswith(".CSV")
                         or f.endswith(".csv")])
                st.sidebar.success(
                    f"✅ {n} files loaded"
                )
        else:
            st.sidebar.error("❌ Folder not found")

# ── TABS ──
tab1, tab2, tab3 = st.tabs([
    "📊 Historical Overview",
    "🔬 Data Preview",
    "🔍 Health Analysis"
])

# ─────────────────────────────────────────
# TAB 1 — HISTORICAL OVERVIEW
# ─────────────────────────────────────────
with tab1:
    st.markdown("## Historical Overview by Mode")
    st.markdown(
        "Each mode is tracked and compared "
        "independently against its own baseline."
    )

    for mode in [0.0, 2.0, 1.0]:
        mode_name = MODE_NAMES.get(mode,"Unknown")
        mode_data = historical[
            historical["run_mode"] == mode
        ]

        if len(mode_data) == 0:
            continue

        st.markdown(f"### Mode {int(mode)} — "
                    f"{mode_name}")

        if str(mode) not in            {str(k) for k in baselines.keys()}:
            st.info(f"No baseline for Mode {int(mode)}")
            continue

        b_info = baselines.get(
            mode, baselines.get(str(mode))
        )
        if b_info is None:
            continue

        b_mean = b_info["baseline_mean"]
        b_std  = b_info["baseline_std"]
        alert  = b_info["alert_threshold"]
        warn   = b_info["warning_threshold"]

        # Daily summary for this mode
        daily = mode_data.groupby("day").agg(
            cycles=("cycle_id","count"),
            ret_torque=("ret_torque_min","mean"),
            fwd_torque=("fwd_torque_max","mean"),
        ).round(4)

        # Health scores
        daily["health"] = daily[
            "ret_torque"
        ].apply(lambda x: max(0,min(100,
            (x-alert)/(b_mean-alert)*100
        ))).round(1)

        daily["status"] = daily[
            "ret_torque"
        ].apply(lambda x:
            "🔴 CRITICAL" if x < alert else
            "🟡 WARNING" if x < warn else
            "🟡 CAUTION" if (x-alert)/
            (b_mean-alert)*100 < 80 else
            "🟢 NORMAL"
        )

        st.dataframe(
            daily.reset_index(),
            use_container_width=True
        )

        # Trend chart
        if len(daily) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily.index,
                y=daily["ret_torque"],
                mode="lines+markers+text",
                name="Return Torque",
                line=dict(
                    color=MODE_COLORS.get(
                        mode,"blue"),
                    width=2),
                marker=dict(size=10),
                text=[str(v) for v in
                      daily["ret_torque"]],
                textposition="top center"
            ))
            fig.add_hline(
                y=alert, line_dash="dash",
                line_color="red",
                annotation_text=f"Alert {alert}"
            )
            fig.add_hline(
                y=warn, line_dash="dash",
                line_color="orange",
                annotation_text=f"Warning {warn}"
            )
            fig.update_layout(
                title=f"Mode {int(mode)} — "
                      f"{mode_name} Torque Trend",
                yaxis_title="Return Torque (Nm)",
                height=300
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

        st.markdown("---")

# ─────────────────────────────────────────
# TAB 2 — DATA PREVIEW
# ─────────────────────────────────────────
with tab2:
    st.markdown("## Data Preview")
    st.markdown(
        "Check data quality before running analysis"
    )

    if df_raw is None:
        st.info("👈 Load data from sidebar first")
    else:
        total = len(df_raw)
        active = df_raw[
            df_raw["sequence"].isin(
                [3110.0,3115.0,3120.0,
                 3125.0,3130.0]
            )
        ]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Rows", f"{total:,}")
        c2.metric("Active Rows",
                  f"{len(active):,}",
                  f"{len(active)/total*100:.1f}%")
        c3.metric("Time Start",
                  str(df_raw["timestamp"].min()
                      ).split(".")[0])
        c4.metric("Time End",
                  str(df_raw["timestamp"].max()
                      ).split(".")[0])

        # Mode breakdown
        st.markdown("### Mode Breakdown")
        mode_counts = df_raw[
            df_raw["sequence"].isin(
                [3110.0,3115.0,3120.0,
                 3125.0,3130.0]
            )
        ]["run_mode"].value_counts()

        mc1,mc2,mc3 = st.columns(3)
        for col, (mode, mname) in zip(
            [mc1,mc2,mc3],
            [(0.0,"Normal"),(2.0,"Maintenance"),
             (1.0,"Endurance")]
        ):
            count = int(mode_counts.get(mode, 0))
            pct = count/len(active)*100                   if len(active)>0 else 0
            col.metric(
                f"Mode {int(mode)} — {mname}",
                f"{count:,} rows",
                f"{pct:.1f}%"
            )

        # Sequence distribution
        st.markdown("### Sequence Distribution")
        seq_counts = df_raw[
            "sequence"
        ].value_counts().sort_index()
        seq_df = pd.DataFrame({
            "Sequence": [
                f"{int(k)} — "
                f"{SEQ_NAMES.get(int(k),'Unknown')}"
                for k in seq_counts.index
            ],
            "Count": seq_counts.values
        })
        fig = px.bar(
            seq_df, x="Sequence", y="Count",
            title="Rows per Sequence",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig,
                        use_container_width=True)

        # Check key sequences
        st.markdown("### Key Sequences Check")
        k1,k2,k3,k4 = st.columns(4)
        for col, seq, name in zip(
            [k1,k2,k3,k4],
            [3110.0,3115.0,3125.0,3130.0],
            ["Forward","Settle","Return","End"]
        ):
            found = seq in df_raw[
                "sequence"].values
            col.metric(
                f"Seq {int(seq)}\n{name}",
                "✅ Found" if found
                else "❌ Missing"
            )

        can_extract = all(
            s in df_raw["sequence"].values
            for s in [3115.0,3125.0,3130.0]
        )
        if can_extract:
            st.success(
                "✅ Data looks good — proceed to "
                "Health Analysis tab"
            )
        else:
            st.error(
                "❌ Missing key sequences — "
                "cannot extract cycles"
            )

# ─────────────────────────────────────────
# TAB 3 — HEALTH ANALYSIS
# ─────────────────────────────────────────
with tab3:
    st.markdown("## Health Analysis by Mode")
    st.markdown(
        "Each mode is analysed independently "
        "against its own historical baseline."
    )

    if df_raw is None:
        st.info("👈 Load data from sidebar first")
    elif not day_label:
        st.info("👈 Enter date label in sidebar")
    else:
        if st.button(
            "🔍 Run Health Analysis",
            type="primary",
            use_container_width=True
        ):
            # Extract all cycles
            with st.spinner(
                "Extracting cycles..."
            ):
                all_new = extract_cycles(
                    df_raw, day_label
                )

            if len(all_new) == 0:
                st.error(
                    "❌ No cycles found. "
                    "Check Data Preview tab."
                )
                st.stop()

            st.success(
                f"✅ {len(all_new)} total cycles "
                f"extracted"
            )

            # Show mode breakdown
            mode_breakdown = all_new.groupby(
                "run_mode"
            ).size()
            st.markdown("**Cycles per mode:**")
            cols = st.columns(3)
            for i,(mode,mname) in enumerate([
                (0.0,"Normal"),
                (2.0,"Maintenance"),
                (1.0,"Endurance")
            ]):
                count = int(
                    mode_breakdown.get(mode,0)
                )
                cols[i].metric(
                    f"Mode {int(mode)} — {mname}",
                    f"{count} cycles"
                )

            # Update historical
            updated = pd.concat(
                [historical[
                    historical["day"]!=day_label
                 ], all_new],
                ignore_index=True
            )
            updated.to_csv(
                "all_cycles_4days.csv",
                index=False
            )
            st.cache_data.clear()

            st.markdown("---")

            # Analyse each mode separately
            modes_present = all_new[
                "run_mode"
            ].unique()

            for mode in sorted(modes_present):
                mode_name = MODE_NAMES.get(
                    mode, "Unknown"
                )
                new_mode = all_new[
                    all_new["run_mode"]==mode
                ].copy()

                if len(new_mode) == 0:
                    continue

                st.markdown(
                    f"## Mode {int(mode)} — "
                    f"{mode_name}"
                )

                # Get baseline for this mode
                b_info = baselines.get(mode)
                if b_info is None:
                    # First time seeing this mode
                    # Use current data as baseline
                    b_mean = float(
                        new_mode[
                            "ret_torque_min"
                        ].mean()
                    )
                    b_std = float(
                        new_mode[
                            "ret_torque_min"
                        ].std()
                    )
                    b_info = {
                        "baseline_day": day_label,
                        "baseline_mean": round(
                            b_mean,4),
                        "baseline_std": round(
                            b_std,4),
                        "alert_threshold": round(
                            b_mean-3*b_std,4),
                        "warning_threshold": round(
                            b_mean-2*b_std,4),
                    }
                    st.info(
                        f"ℹ️ First time seeing "
                        f"Mode {int(mode)}. "
                        f"Using today as baseline. "
                        f"More data needed for "
                        f"accurate predictions."
                    )
                    # Save new baseline
                    baselines[mode] = b_info
                    with open(
                        "mode_baselines.json","w"
                    ) as f:
                        json.dump(
                            {str(k):v for k,v in
                             baselines.items()},
                            f, indent=2
                        )

                # Historical for this mode only
                hist_mode = historical[
                    historical["run_mode"]==mode
                ].copy()

                result = analyse_one_mode(
                    new_mode, hist_mode,
                    b_info, day_label, mode
                )

                display_mode_result(
                    result,
                    MODE_COLORS.get(mode,"blue")
                )
                st.markdown("---")

            st.success(
                f"✅ {day_label} saved to history"
            )

# Footer
st.markdown("---")
st.caption(
    f"🔧 Shuffler Motor Predictive Maintenance | "
    f"Historical: {len(historical):,} cycles | "
    f"Modes tracked separately"
)
