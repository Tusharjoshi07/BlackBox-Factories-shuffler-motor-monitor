import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import hashlib
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
MODE_NAMES  = {0.0:"Normal", 1.0:"Endurance", 2.0:"Maintenance"}
MODE_COLORS = {0.0:"#2196F3", 1.0:"#FF9800", 2.0:"#4CAF50"}
SEQ_NAMES   = {
    3000:"Standby", 3100:"Ready/Idle",
    3110:"Forward Move", 3115:"Forward Settle",
    3120:"Forward Hold", 3125:"Return Move", 3130:"Return Settle"
}

# Expected CSV format constants
EXPECTED_HEADER_ROW   = 3          # 0-indexed — column names on line 3
EXPECTED_DATA_START   = 4          # data starts line 4
REQUIRED_COLUMNS      = ["D3203","D3223","D3224","D3238"]
TIMESTAMP_COLUMN      = "TIME (UTC+09:00)"
EXPECTED_MIN_COLUMNS  = 10         # minimum columns a valid file should have
EXPECTED_INTERVAL_COL = "INTERVAL[us]"
EXPECTED_MAGIC_HEADER = "[LOGGING]" # first token of line 0

# ─────────────────────────────────────────
# LOAD HISTORICAL DATA
# ─────────────────────────────────────────
@st.cache_data
def load_all_data():
    try:
        df = pd.read_csv("all_cycles_4days.csv")
        df["start_time"] = pd.to_datetime(df["start_time"])
        return df
    except:
        return None

@st.cache_data
def load_baselines():
    try:
        with open("mode_baselines.json") as f:
            raw = json.load(f)
        return {float(k): v for k, v in raw.items()}
    except:
        return {}

# ─────────────────────────────────────────
# FORMAT VALIDATION
# ─────────────────────────────────────────
def validate_file_format(content, filename):
    """
    Validates that a CSV file matches the expected
    Mitsubishi PLC format. Returns (is_valid, issues_list)
    """
    issues = []
    lines = content.split("\n")

    # Check 1 — Minimum line count
    if len(lines) < 6:
        issues.append(
            f"File has only {len(lines)} lines. "
            f"Expected at least 6 (4 header + data)."
        )
        return False, issues

    # Check 2 — Magic header on line 0
    line0 = lines[0].strip()
    if not line0.startswith(EXPECTED_MAGIC_HEADER):
        issues.append(
            f"Line 0 should start with '{EXPECTED_MAGIC_HEADER}' "
            f"but found: '{line0[:30]}...'"
        )

    # Check 3 — Line 1 should be MergedFile
    line1 = lines[1].strip()
    if "MergedFile" not in line1 and line1 != "":
        issues.append(
            f"Line 1 expected 'MergedFile' "
            f"but found: '{line1[:30]}'"
        )

    # Check 4 — Column names on line 3
    try:
        col_names = lines[EXPECTED_HEADER_ROW].strip().split(",")
        if len(col_names) < EXPECTED_MIN_COLUMNS:
            issues.append(
                f"Expected at least {EXPECTED_MIN_COLUMNS} columns "
                f"but found only {len(col_names)}."
            )

        # Check 5 — Timestamp column present
        if TIMESTAMP_COLUMN not in col_names:
            issues.append(
                f"Missing timestamp column '{TIMESTAMP_COLUMN}'. "
                f"Found columns: {col_names[:5]}..."
            )

        # Check 6 — Required sensor columns present
        missing_cols = [c for c in REQUIRED_COLUMNS
                       if c not in col_names]
        if missing_cols:
            issues.append(
                f"Missing required sensor columns: {missing_cols}. "
                f"These are essential for analysis."
            )

        # Check 7 — Interval column present
        if EXPECTED_INTERVAL_COL not in col_names:
            issues.append(
                f"Missing '{EXPECTED_INTERVAL_COL}' column. "
                f"File may be from a different PLC system."
            )

    except Exception as e:
        issues.append(f"Could not parse column names: {str(e)}")
        return False, issues

    # Check 8 — Line 4 should be empty/commas
    line4 = lines[4].strip().replace(",", "")
    if line4 != "":
        issues.append(
            f"Line 4 should be empty separator row "
            f"but found data: '{lines[4][:30]}...'"
        )

    # Check 9 — First data row parseable
    try:
        first_data = lines[EXPECTED_DATA_START].strip()
        if first_data == "":
            issues.append("No data found after headers.")
        else:
            parts = first_data.split(",")
            if len(parts) < EXPECTED_MIN_COLUMNS:
                issues.append(
                    f"First data row has only {len(parts)} values. "
                    f"Expected {len(col_names)}."
                )
    except Exception as e:
        issues.append(f"Cannot read first data row: {str(e)}")

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_all_files(files_contents):
    """
    Validate a list of (filename, content) tuples.
    Returns summary of valid/invalid files.
    """
    valid_files   = []
    invalid_files = []

    for filename, content in files_contents:
        is_valid, issues = validate_file_format(
            content, filename
        )
        if is_valid:
            valid_files.append(filename)
        else:
            invalid_files.append({
                "filename": filename,
                "issues": issues
            })

    return valid_files, invalid_files


# ─────────────────────────────────────────
# CSV PARSING
# ─────────────────────────────────────────
def parse_csv(content, day_label):
    try:
        lines = content.split("\n")
        col_names = lines[EXPECTED_HEADER_ROW].strip().split(",")
        df_temp = pd.read_csv(
            io.StringIO("\n".join(lines[EXPECTED_DATA_START:])),
            names=col_names,
            on_bad_lines="skip"
        )
        if not all(c in df_temp.columns for c in REQUIRED_COLUMNS):
            return None
        df_temp = df_temp.dropna(
            subset=["D3203"]
        ).reset_index(drop=True)
        df_clean = df_temp[[
            TIMESTAMP_COLUMN, "D3203", "D3223", "D3224", "D3238"
        ]].copy()
        df_clean.columns = [
            "timestamp","sequence","torque_raw","velocity","run_mode"
        ]
        df_clean["timestamp"] = pd.to_datetime(
            df_clean["timestamp"], errors="coerce"
        )
        df_clean["torque_nm"] = df_clean["torque_raw"] * 0.00064
        df_clean["day"] = day_label
        return df_clean
    except:
        return None


def load_uploaded(files, day_label):
    file_contents = []
    for f in files:
        content = f.read().decode("utf-8", errors="ignore")
        file_contents.append((f.name, content))

    valid_files, invalid_files = validate_all_files(file_contents)

    all_data = []
    for filename, content in file_contents:
        if filename in valid_files:
            df = parse_csv(content, day_label)
            if df is not None:
                all_data.append(df)

    if not all_data:
        return None, invalid_files, valid_files

    df = pd.concat(
        all_data, ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)
    return df, invalid_files, valid_files


def load_folder(folder_path, day_label):
    all_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".CSV") or f.endswith(".csv")
    ])
    file_contents = []
    for filename in all_files:
        fp = os.path.join(folder_path, filename)
        try:
            with open(fp, "r",
                      encoding="utf-8",
                      errors="ignore") as f:
                content = f.read()
            file_contents.append((filename, content))
        except:
            continue

    valid_files, invalid_files = validate_all_files(file_contents)

    all_data = []
    for filename, content in file_contents:
        if filename in valid_files:
            df = parse_csv(content, day_label)
            if df is not None:
                all_data.append(df)

    if not all_data:
        return None, invalid_files, valid_files

    df = pd.concat(
        all_data, ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)
    return df, invalid_files, valid_files


# ─────────────────────────────────────────
# CYCLE EXTRACTION
# ─────────────────────────────────────────
def extract_cycles(df_all, day_label):
    active = [3110.0, 3115.0, 3120.0, 3125.0, 3130.0]
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

        fwd    = df_active.iloc[fwd_start:settle_start]
        settle = df_active.iloc[settle_start:return_start]
        ret    = df_active.iloc[return_start:cycle_end]

        if len(settle) < 3 or len(ret) < 3:
            i = cycle_end + 1
            continue

        cycles.append({
            "cycle_id":          cycle_id,
            "day":               day_label,
            "start_time":        df_active.iloc[fwd_start]["timestamp"],
            "run_mode":          df_active.iloc[fwd_start]["run_mode"],
            "fwd_torque_max":    fwd["torque_nm"].max()  if len(fwd)>0 else np.nan,
            "fwd_torque_mean":   fwd["torque_nm"].mean() if len(fwd)>0 else np.nan,
            "fwd_torque_std":    fwd["torque_nm"].std()  if len(fwd)>0 else np.nan,
            "fwd_velocity_max":  fwd["velocity"].max()   if len(fwd)>0 else np.nan,
            "fwd_duration":      len(fwd),
            "settle_torque_mean":settle["torque_nm"].mean(),
            "settle_torque_std": settle["torque_nm"].std(),
            "settle_duration":   len(settle),
            "ret_torque_min":    ret["torque_nm"].min(),
            "ret_torque_mean":   ret["torque_nm"].mean(),
            "ret_torque_std":    ret["torque_nm"].std(),
            "ret_velocity_min":  ret["velocity"].min(),
            "ret_duration":      len(ret),
            "total_duration":    cycle_end - fwd_start,
        })
        cycle_id += 1
        i = cycle_end + 1

    return pd.DataFrame(cycles)


# ─────────────────────────────────────────
# DUPLICATE DETECTION — ALL COLUMNS
# ─────────────────────────────────────────
def check_duplicates(new_cycles, historical, day_label):
    """
    Comprehensive duplicate and conflict detection.
    Checks every column for row-level duplication.
    """
    issues   = []
    warnings = []

    if len(new_cycles) == 0:
        return issues, warnings

    # All numeric columns used for comparison
    compare_cols = [
        "run_mode",
        "fwd_torque_max", "fwd_torque_mean", "fwd_torque_std",
        "fwd_velocity_max", "fwd_duration",
        "settle_torque_mean", "settle_torque_std", "settle_duration",
        "ret_torque_min", "ret_torque_mean", "ret_torque_std",
        "ret_velocity_min", "ret_duration",
        "total_duration"
    ]

    # ── Check 1: Same date label ──
    existing_days = historical["day"].unique().tolist()
    if day_label in existing_days:
        existing = historical[historical["day"] == day_label]
        issues.append({
            "type":             "DATE_EXISTS",
            "severity":         "HIGH",
            "message":          f"Date label '{day_label}' already exists in history.",
            "existing_cycles":  len(existing),
            "new_cycles":       len(new_cycles),
            "action":           f"Saving will REPLACE existing {len(existing)} cycles with new {len(new_cycles)} cycles for this date."
        })

    # ── Check 2: Row-level full duplicate detection ──
    # Round values for comparison to avoid float precision issues
    def make_fingerprint(df):
        cols = [c for c in compare_cols if c in df.columns]
        rounded = df[cols].round(6)
        return set(
            rounded.apply(
                lambda row: hashlib.md5(
                    str(tuple(row.values)).encode()
                ).hexdigest(),
                axis=1
            ).tolist()
        )

    new_fp  = make_fingerprint(new_cycles)
    hist_fp = make_fingerprint(historical)
    row_dupes = new_fp & hist_fp
    n_dupes = len(row_dupes)

    if n_dupes > 0:
        dupe_pct = n_dupes / len(new_cycles) * 100
        severity = "HIGH" if dupe_pct > 50 else (
            "MEDIUM" if dupe_pct > 10 else "LOW"
        )
        issues.append({
            "type":      "ROW_DUPLICATES",
            "severity":  severity,
            "message":   f"{n_dupes} cycles ({dupe_pct:.1f}%) in new data are IDENTICAL to existing history rows (checked all {len(compare_cols)} feature columns).",
            "n_dupes":   n_dupes,
            "dupe_pct":  round(dupe_pct, 1),
            "action":    "These are exact duplicate cycles — likely same data uploaded twice."
        })

    # ── Check 3: Timestamp overlap ──
    new_start = pd.to_datetime(new_cycles["start_time"]).min()
    new_end   = pd.to_datetime(new_cycles["start_time"]).max()

    overlap = historical[
        (pd.to_datetime(historical["start_time"]) >= new_start) &
        (pd.to_datetime(historical["start_time"]) <= new_end)
    ]

    if len(overlap) > 0 and day_label not in existing_days:
        overlapping_days = overlap["day"].unique().tolist()
        issues.append({
            "type":             "TIMESTAMP_OVERLAP",
            "severity":         "HIGH",
            "message":          f"New data timestamps overlap with existing history! {len(overlap)} existing cycles fall in the same time window.",
            "overlapping_days": overlapping_days,
            "overlap_count":    len(overlap),
            "new_start":        str(new_start),
            "new_end":          str(new_end),
            "action":           f"This data may already exist under label(s): {overlapping_days}. Check before saving."
        })

    # ── Check 4: New data has fewer cycles than existing ──
    if day_label in existing_days:
        existing_count = len(historical[historical["day"] == day_label])
        if len(new_cycles) < existing_count * 0.5:
            warnings.append({
                "type":    "FEWER_CYCLES",
                "message": f"New data has only {len(new_cycles)} cycles but existing history has {existing_count} for '{day_label}'. You may be uploading partial data (e.g. single file instead of full folder)."
            })

    # ── Check 5: Too few cycles ──
    if len(new_cycles) < 5:
        warnings.append({
            "type":    "TOO_FEW_CYCLES",
            "message": f"Only {len(new_cycles)} cycles found. Minimum 5 recommended for reliable analysis."
        })

    # ── Check 6: Torque values out of expected range ──
    ret_mean = new_cycles["ret_torque_min"].mean()
    if ret_mean > 0:
        warnings.append({
            "type":    "TORQUE_DIRECTION",
            "message": f"Return torque mean is positive ({ret_mean:.4f} Nm). Expected negative. Data may be from a different machine or direction."
        })
    if abs(ret_mean) > 1.0:
        warnings.append({
            "type":    "TORQUE_MAGNITUDE",
            "message": f"Return torque ({ret_mean:.4f} Nm) is unusually large. Check sensor calibration (D3223 × 0.00064 conversion)."
        })

    # ── Check 7: Mode consistency ──
    new_modes  = set(new_cycles["run_mode"].unique())
    hist_modes = set(historical["run_mode"].unique())
    truly_new  = new_modes - hist_modes
    if truly_new:
        warnings.append({
            "type":    "NEW_MODE",
            "message": f"New mode(s) detected not seen before: {[MODE_NAMES.get(m,str(m)) for m in truly_new]}. App will create new baseline automatically."
        })

    return issues, warnings


# ─────────────────────────────────────────
# ANALYSIS PER MODE
# ─────────────────────────────────────────
def analyse_one_mode(new_cycles_mode, historical_mode,
                     baseline_info, day_label, mode):
    if len(new_cycles_mode) == 0:
        return None

    b_mean = baseline_info["baseline_mean"]
    b_std  = baseline_info["baseline_std"]
    alert  = baseline_info["alert_threshold"]
    warn   = baseline_info["warning_threshold"]

    ret_mean = new_cycles_mode["ret_torque_min"].mean()
    fwd_mean = new_cycles_mode["fwd_torque_max"].mean()

    health = max(0, min(100,
        (ret_mean - alert) / (b_mean - alert) * 100
    ))

    if historical_mode is not None and len(historical_mode) > 0:
        daily = historical_mode.groupby("day")["ret_torque_min"].mean()
        new_val = pd.Series({day_label: ret_mean})
        daily = pd.concat([daily, new_val])
        daily = daily[~daily.index.duplicated(keep="last")]
        t_vals = daily.values
        t_nums = list(range(len(t_vals)))
        if len(t_vals) >= 2:
            z = np.polyfit(t_nums, t_vals, 1)
            slope = z[0]
            days_to = int((alert - ret_mean) / slope) if slope < 0 else 999
        else:
            slope = 0
            days_to = 999
    else:
        slope = 0
        days_to = 999

    if   ret_mean < alert: status = "CRITICAL"
    elif ret_mean < warn:  status = "WARNING"
    elif health < 80:      status = "CAUTION"
    else:                  status = "NORMAL"

    base_cycles = (
        historical_mode[
            historical_mode["day"] == baseline_info["baseline_day"]
        ] if historical_mode is not None and len(historical_mode) > 0
        else new_cycles_mode
    )

    seq_analysis = {}
    for seq_key, metrics in [
        ("3110",[
            ("torque_mean","fwd_torque_mean","fwd_torque_mean"),
            ("torque_max","fwd_torque_max","fwd_torque_max"),
            ("duration","fwd_duration","fwd_duration"),
        ]),
        ("3115",[("duration","settle_duration","settle_duration")]),
        ("3125",[
            ("torque_min","ret_torque_min","ret_torque_min"),
            ("velocity","ret_velocity_min","ret_velocity_min"),
            ("duration","ret_duration","ret_duration"),
        ]),
    ]:
        names = {"3110":"Forward Move","3115":"Forward Settle","3125":"Return Move"}
        alerts_seq = []
        for label, b_col, c_col in metrics:
            if len(base_cycles) == 0:
                continue
            b = float(base_cycles[b_col].mean())
            c = float(new_cycles_mode[c_col].mean())
            if b == 0 or np.isnan(b) or np.isnan(c):
                continue
            pct = ((c - b) / abs(b)) * 100
            if abs(pct) > 10:
                alerts_seq.append({
                    "metric": label,
                    "baseline": round(b, 4),
                    "current":  round(c, 4),
                    "change_pct": round(pct, 1)
                })
        seq_analysis[seq_key] = {
            "name":   names[seq_key],
            "status": "STRESS" if alerts_seq else "NORMAL",
            "alerts": alerts_seq
        }

    return {
        "mode":               mode,
        "mode_name":          MODE_NAMES.get(mode, "Unknown"),
        "cycles":             len(new_cycles_mode),
        "health_score":       round(health, 1),
        "status":             status,
        "ret_torque":         round(ret_mean, 4),
        "fwd_torque":         round(fwd_mean, 4),
        "alert_threshold":    round(alert, 4),
        "warning_threshold":  round(warn, 4),
        "days_to_alert":      days_to,
        "trend_per_day":      round(slope, 6),
        "sequence_analysis":  seq_analysis,
        "baseline_day":       baseline_info["baseline_day"],
        "baseline_mean":      round(b_mean, 4),
    }


# ─────────────────────────────────────────
# DISPLAY ONE MODE RESULT
# ─────────────────────────────────────────
def display_mode_result(result):
    if result is None:
        st.info("No data for this mode")
        return

    status    = result["status"]
    score     = result["health_score"]
    icon_map  = {"CRITICAL":"🔴","WARNING":"🟡","CAUTION":"🟡","NORMAL":"🟢"}
    color_map = {"CRITICAL":"red","WARNING":"orange","CAUTION":"orange","NORMAL":"green"}

    st.markdown(
        f"**{result['cycles']} cycles** | "
        f"Baseline: {result['baseline_day']} | "
        f"Baseline torque: {result['baseline_mean']} Nm"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={"text":"Health Score"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":color_map[status]},
                "steps":[
                    {"range":[0,30],  "color":"lightcoral"},
                    {"range":[30,70], "color":"lightyellow"},
                    {"range":[70,100],"color":"lightgreen"},
                ]
            }
        ))
        fig.update_layout(height=220)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.metric("Status", f"{icon_map[status]} {status}")
        st.metric("Return Torque", f"{result['ret_torque']} Nm",
                  delta=f"{result['ret_torque']-result['alert_threshold']:.4f} to alert")
        st.metric("Days to Alert",
                  result["days_to_alert"] if result["days_to_alert"] < 999 else "Safe ✅")

    with c3:
        st.metric("Alert Threshold",   f"{result['alert_threshold']} Nm")
        st.metric("Warning Threshold", f"{result['warning_threshold']} Nm")
        st.metric("Trend/Day",         f"{result['trend_per_day']} Nm")

    st.markdown("**Sequence Level:**")
    s1, s2, s3 = st.columns(3)
    for col, seq_key in zip([s1,s2,s3], ["3110","3115","3125"]):
        with col:
            s    = result["sequence_analysis"][seq_key]
            icon = "⚠️" if s["status"] == "STRESS" else "✅"
            st.markdown(f"{icon} **Seq {seq_key} — {s['name']}**")
            if s["alerts"]:
                for a in s["alerts"]:
                    d = "↑" if a["change_pct"] > 0 else "↓"
                    st.caption(f"{a['metric']}: {a['change_pct']:+.1f}% {d}")
                    st.caption(f"Baseline: {a['baseline']} → Now: {a['current']}")
            else:
                st.caption("All normal ✅")

    if   status == "CRITICAL": st.error(  "🔴 **STOP MACHINE** — Immediate inspection required.")
    elif status == "WARNING":  st.warning(f"🟡 **Maintenance within {result['days_to_alert']} days**")
    elif status == "CAUTION":  st.warning("🟡 **Monitor daily** — early signs of stress")
    else:                      st.success("🟢 **Machine healthy** — all parameters normal")


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
st.title("🔧 Shuffler Motor Health Monitor")
st.markdown("### Predictive Maintenance — Blackbox Factories")
st.markdown("---")

historical = load_all_data()
baselines  = load_baselines()

if historical is None:
    st.error("❌ Historical data not found. Upload all_cycles_4days.csv to GitHub.")
    st.stop()

# ── SIDEBAR ──
st.sidebar.header("📁 Load New Data")
day_label    = st.sidebar.text_input("Date Label", placeholder="e.g. May 01")
input_method = st.sidebar.radio("Data Source", ["📤 Upload CSV Files","📂 Enter Folder Path"])

df_raw        = None
invalid_files = []
valid_files   = []

if input_method == "📤 Upload CSV Files":
    uploaded = st.sidebar.file_uploader(
        "Select all CSV files",
        type=["csv","CSV"],
        accept_multiple_files=True
    )
    if uploaded and day_label:
        with st.spinner("Reading and validating files..."):
            df_raw, invalid_files, valid_files = load_uploaded(uploaded, day_label)
        if df_raw is not None:
            st.sidebar.success(f"✅ {len(valid_files)} valid files loaded")
        if invalid_files:
            st.sidebar.error(f"❌ {len(invalid_files)} files failed validation")
else:
    folder = st.sidebar.text_input("Folder Path", placeholder="/path/to/csv/folder")
    if folder and day_label:
        if os.path.exists(folder):
            with st.spinner("Loading and validating folder..."):
                df_raw, invalid_files, valid_files = load_folder(folder, day_label)
            if df_raw is not None:
                st.sidebar.success(f"✅ {len(valid_files)} valid files loaded")
            if invalid_files:
                st.sidebar.error(f"❌ {len(invalid_files)} files failed validation")
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
    st.markdown("Each mode is tracked and compared independently against its own baseline.")

    for mode in [0.0, 2.0, 1.0]:
        mode_name = MODE_NAMES.get(mode, "Unknown")
        mode_data = historical[historical["run_mode"] == mode]
        if len(mode_data) == 0:
            continue

        b_info = baselines.get(mode)
        if b_info is None:
            continue

        b_mean = b_info["baseline_mean"]
        b_std  = b_info["baseline_std"]
        alert  = b_info["alert_threshold"]
        warn   = b_info["warning_threshold"]

        st.markdown(f"### Mode {int(mode)} — {mode_name}")

        daily = mode_data.groupby("day").agg(
            cycles=("cycle_id","count"),
            ret_torque=("ret_torque_min","mean"),
            fwd_torque=("fwd_torque_max","mean"),
        ).round(4)

        daily["health"] = daily["ret_torque"].apply(
            lambda x: max(0, min(100, (x-alert)/(b_mean-alert)*100))
        ).round(1)

        daily["status"] = daily["ret_torque"].apply(
            lambda x:
            "🔴 CRITICAL" if x < alert else
            "🟡 WARNING"  if x < warn  else
            "🟡 CAUTION"  if (x-alert)/(b_mean-alert)*100 < 80 else
            "🟢 NORMAL"
        )

        st.dataframe(daily.reset_index(), use_container_width=True)

        if len(daily) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily.index, y=daily["ret_torque"],
                mode="lines+markers+text",
                name="Return Torque",
                line=dict(color=MODE_COLORS.get(mode,"blue"), width=2),
                marker=dict(size=10),
                text=[str(v) for v in daily["ret_torque"]],
                textposition="top center"
            ))
            fig.add_hline(y=alert, line_dash="dash", line_color="red",
                          annotation_text=f"Alert {alert}")
            fig.add_hline(y=warn,  line_dash="dash", line_color="orange",
                          annotation_text=f"Warning {warn}")
            fig.update_layout(
                title=f"Mode {int(mode)} — {mode_name} Torque Trend",
                yaxis_title="Return Torque (Nm)", height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

# ─────────────────────────────────────────
# TAB 2 — DATA PREVIEW
# ─────────────────────────────────────────
with tab2:
    st.markdown("## Data Preview")
    st.markdown("Validate your data before running analysis.")

    # Show format validation issues first
    if invalid_files:
        st.markdown("### ❌ Format Validation Issues")
        st.error(
            f"{len(invalid_files)} file(s) failed format validation "
            f"and were excluded from analysis."
        )
        for inv in invalid_files:
            with st.expander(f"❌ {inv['filename']}"):
                for issue in inv["issues"]:
                    st.markdown(f"- {issue}")
        if valid_files:
            st.info(
                f"✅ {len(valid_files)} valid files will be used for analysis."
            )
        else:
            st.error(
                "❌ No valid files found. "
                "Check format requirements below."
            )
            st.markdown("### Expected File Format")
            st.code(
                "Line 0: [LOGGING],RCPU_1,...\n"
                "Line 1: MergedFile\n"
                "Line 2: DATETIME[...],INTERVAL,...\n"
                "Line 3: TIME (UTC+09:00),INTERVAL[us],...,D3203,...,D3223,D3224,...,D3238\n"
                "Line 4: (empty separator row)\n"
                "Line 5+: data rows"
            )

    if df_raw is None:
        if not invalid_files:
            st.info("👈 Load data from sidebar first")
    else:
        total  = len(df_raw)
        active = df_raw[df_raw["sequence"].isin(
            [3110.0,3115.0,3120.0,3125.0,3130.0]
        )]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Rows",  f"{total:,}")
        c2.metric("Active Rows", f"{len(active):,}", f"{len(active)/total*100:.1f}%")
        c3.metric("Start", str(df_raw["timestamp"].min()).split(".")[0])
        c4.metric("End",   str(df_raw["timestamp"].max()).split(".")[0])

        st.markdown("### Mode Breakdown")
        mc1,mc2,mc3 = st.columns(3)
        for col,(mode,mname) in zip([mc1,mc2,mc3],
            [(0.0,"Normal"),(2.0,"Maintenance"),(1.0,"Endurance")]):
            count = int(active[active["run_mode"]==mode].shape[0]) if len(active)>0 else 0
            pct   = count/len(active)*100 if len(active)>0 else 0
            col.metric(f"Mode {int(mode)} — {mname}", f"{count:,} rows", f"{pct:.1f}%")

        st.markdown("### Sequence Distribution")
        seq_counts = df_raw["sequence"].value_counts().sort_index()
        seq_df = pd.DataFrame({
            "Sequence":[f"{int(k)} — {SEQ_NAMES.get(int(k),'Unknown')}"
                        for k in seq_counts.index],
            "Count": seq_counts.values
        })
        fig = px.bar(seq_df, x="Sequence", y="Count",
                     title="Rows per Sequence",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Key Sequences Check")
        k1,k2,k3,k4 = st.columns(4)
        for col,seq,name in zip([k1,k2,k3,k4],
            [3110.0,3115.0,3125.0,3130.0],
            ["Forward","Settle","Return","End"]):
            found = seq in df_raw["sequence"].values
            col.metric(f"Seq {int(seq)}\n{name}",
                       "✅ Found" if found else "❌ Missing")

        can_extract = all(
            s in df_raw["sequence"].values
            for s in [3115.0,3125.0,3130.0]
        )
        if can_extract:
            st.success("✅ Data looks good — proceed to Health Analysis tab")
        else:
            st.error("❌ Missing key sequences — cannot extract cycles")

# ─────────────────────────────────────────
# TAB 3 — HEALTH ANALYSIS
# ─────────────────────────────────────────
with tab3:
    st.markdown("## Health Analysis by Mode")
    st.markdown("Each mode analysed independently against its own baseline.")

    if df_raw is None:
        st.info("👈 Load data from sidebar first")
    elif not day_label:
        st.info("👈 Enter date label in sidebar")
    else:
        if st.button("🔍 Run Health Analysis",
                     type="primary", use_container_width=True):

            with st.spinner("Extracting cycles..."):
                all_new = extract_cycles(df_raw, day_label)

            if len(all_new) == 0:
                st.error("❌ No cycles found. Check Data Preview tab.")
                st.stop()

            st.success(f"✅ {len(all_new)} total cycles extracted")

            # Mode breakdown
            mode_breakdown = all_new.groupby("run_mode").size()
            cols = st.columns(3)
            for i,(mode,mname) in enumerate([(0.0,"Normal"),(2.0,"Maintenance"),(1.0,"Endurance")]):
                cols[i].metric(f"Mode {int(mode)} — {mname}",
                               f"{int(mode_breakdown.get(mode,0))} cycles")

            # ── VALIDATION SECTION ──
            st.markdown("---")
            st.markdown("### 🔎 Data Validation Report")

            issues, warnings = check_duplicates(
                all_new, historical, day_label
            )

            # Warnings
            if warnings:
                for w in warnings:
                    st.warning(f"⚠️ **{w['type']}:** {w['message']}")

            # Issues
            has_issues = len(issues) > 0
            if has_issues:
                st.error("🚨 **Data conflicts detected — review carefully before saving**")
                for issue in issues:
                    sev_color = "🔴" if issue["severity"]=="HIGH" else "🟡"
                    with st.expander(
                        f"{sev_color} {issue['type']} [{issue['severity']}] — Click to expand",
                        expanded=True
                    ):
                        st.markdown(f"**Issue:** {issue['message']}")
                        st.markdown(f"**What will happen if you save:** {issue['action']}")

                        if "existing_cycles" in issue:
                            col1,col2 = st.columns(2)
                            col1.metric("Existing cycles in history", issue["existing_cycles"])
                            col2.metric("New cycles being uploaded",  issue["new_cycles"])

                        if "overlapping_days" in issue:
                            st.markdown(f"**Overlapping with days:** {issue['overlapping_days']}")
                            st.markdown(f"**Time range:** {issue.get('new_start','')} → {issue.get('new_end','')}")

                        if "n_dupes" in issue:
                            col1,col2 = st.columns(2)
                            col1.metric("Duplicate rows found", issue["n_dupes"])
                            col2.metric("Duplication rate",     f"{issue['dupe_pct']}%")
            else:
                st.success(
                    "✅ **No conflicts detected** — "
                    "no duplicate rows, no timestamp overlaps, "
                    "no existing date label conflict. Safe to save."
                )

            # ── ANALYSIS RESULTS ──
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            st.info("ℹ️ Results shown below. Scroll down to confirm save.")

            mode_results = {}
            for mode in sorted(all_new["run_mode"].unique()):
                mode_name  = MODE_NAMES.get(mode,"Unknown")
                new_mode   = all_new[all_new["run_mode"]==mode].copy()
                if len(new_mode) == 0:
                    continue

                st.markdown(f"## Mode {int(mode)} — {mode_name}")

                b_info = baselines.get(mode)
                if b_info is None:
                    b_mean = float(new_mode["ret_torque_min"].mean())
                    b_std  = float(new_mode["ret_torque_min"].std())
                    if np.isnan(b_std) or b_std == 0:
                        b_std = 0.003
                    b_info = {
                        "baseline_day":       day_label,
                        "baseline_mean":      round(b_mean,4),
                        "baseline_std":       round(b_std,4),
                        "alert_threshold":    round(b_mean-3*b_std,4),
                        "warning_threshold":  round(b_mean-2*b_std,4),
                    }
                    st.info(
                        f"ℹ️ First time seeing Mode {int(mode)} ({mode_name}). "
                        f"Using today as baseline. More data needed for accurate predictions."
                    )

                hist_mode = historical[historical["run_mode"]==mode].copy()
                result    = analyse_one_mode(
                    new_mode, hist_mode, b_info, day_label, mode
                )
                if result:
                    mode_results[mode] = {"result":result,"b_info":b_info}
                    display_mode_result(result)
                st.markdown("---")

            # ── SAVE CONFIRMATION ──
            st.markdown("### 💾 Save to History?")
            if has_issues:
                st.warning(
                    "⚠️ Conflicts were detected above. "
                    "Read carefully before saving."
                )

            col_yes, col_no = st.columns(2)
            with col_yes:
                btn_label = (
                    "⚠️ Save Anyway (I understand the conflicts)"
                    if has_issues else "✅ Save to History"
                )
                if st.button(btn_label, type="primary",
                             use_container_width=True):
                    updated = pd.concat(
                        [historical[historical["day"]!=day_label],
                         all_new],
                        ignore_index=True
                    )
                    updated.to_csv("all_cycles_4days.csv", index=False)
                    for mode, data in mode_results.items():
                        if mode not in baselines:
                            baselines[mode] = data["b_info"]
                    with open("mode_baselines.json","w") as f:
                        json.dump(
                            {str(k):v for k,v in baselines.items()},
                            f, indent=2
                        )
                    st.cache_data.clear()
                    st.success(
                        f"✅ {day_label} saved! "
                        f"{len(all_new)} cycles added to history."
                    )
                    st.balloons()

            with col_no:
                if st.button("❌ Discard — Do Not Save",
                             use_container_width=True):
                    st.info(
                        "Analysis complete. "
                        "Data NOT saved to history. "
                        "Historical data unchanged."
                    )

# Footer
st.markdown("---")
st.caption(
    f"🔧 Shuffler Motor Predictive Maintenance | "
    f"Historical: {len(historical):,} cycles | "
    f"Modes tracked independently | "
    f"Full duplicate detection enabled"
)
