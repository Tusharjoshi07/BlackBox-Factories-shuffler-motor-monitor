# 🔧 Shuffler Motor Health Monitor
### Predictive Maintenance System — Blackbox Factories

A machine learning system that monitors servo motor health by mode, detects anomalies, validates data quality, and predicts when maintenance is needed.

---

## 🔗 Live App
👉 [Click here to open the app](https://blackbox-factories-shuffler-motor-monitor-3rhvryvz3id85pbepzee.streamlit.app/)

---

## 📌 What This System Does

```
1. Load daily CSV files from the PLC system
2. Validate file format before doing anything
3. Preview data quality — sequences, modes, rows
4. Extract machine cycles automatically
5. Separate cycles by run mode automatically
6. Run duplicate detection across ALL columns
7. Show analysis results per mode
8. Ask for confirmation before saving to history
9. Save only if user approves
```

---

## 🗂️ Files in This Repository

| File | Purpose |
|---|---|
| `shuffler_app.py` | Main Streamlit web application |
| `all_cycles_4days.csv` | Historical cycle data — all modes |
| `mode_baselines.json` | Baseline stats per mode (auto-updated) |
| `requirements.txt` | Python libraries needed |
| `README.md` | This file |

---

## 🚀 PART 1 — First Time Setup

### Step 1 — Fork This Repository
```
1. Go to the repo link shared with you
2. Click Fork (top right)
3. Click Create Fork
4. Your copy: github.com/YOUR_USERNAME/shuffler-motor-monitor
```

### Step 2 — Deploy on Streamlit
```
1. Go to share.streamlit.io
2. Sign in with GitHub
3. Click New App
4. Select your forked repo
5. Main file: shuffler_app.py
6. Click Deploy — wait 2-3 minutes
```

---

## 📊 PART 2 — Daily Usage

### Complete Flow Every Day

```
Step 1 — Open app in browser

Step 2 — Sidebar:
  → Enter date label (e.g. May 01)
  → Choose: Upload files OR Folder path
  → Load your CSV files

Step 3 — DATA PREVIEW tab:
  → Check if any files failed validation
  → See mode breakdown
  → See sequence distribution
  → Confirm data looks correct

Step 4 — HEALTH ANALYSIS tab:
  → Click Run Health Analysis
  → System extracts cycles
  → Runs duplicate detection
  → Shows validation report
  → Shows results per mode

Step 5 — Review results:
  → Read health scores per mode
  → Check which sequence has stress
  → Read recommendations

Step 6 — Choose:
  → ✅ Save to History
  → ❌ Discard — if something looks wrong
```

---

## 🔍 PART 3 — Format Validation

The app validates every CSV file before analysis.

### What Gets Checked
```
✅ Line 0 starts with [LOGGING]
✅ Line 1 contains MergedFile
✅ Line 3 has all required column names
✅ Required columns present:
   D3203, D3223, D3224, D3238
✅ Timestamp column present:
   TIME (UTC+09:00)
✅ INTERVAL[us] column present
✅ Minimum 10 columns in file
✅ Line 4 is empty separator row
✅ First data row is parseable
✅ File has at least 6 lines
```

### What Happens With Invalid Files
```
Invalid files are:
→ Flagged with specific reason
→ Listed in Data Preview tab
→ Excluded from analysis
→ Valid files still processed normally
→ You see exactly which file failed and why
```

### Expected File Format
```
Line 0: [LOGGING],RCPU_1,3,4,5,6,2
Line 1: MergedFile
Line 2: DATETIME[YYYY/MM/DD hh:mm:ss.sss],...
Line 3: TIME (UTC+09:00),INTERVAL[us],...,D3203,...,D3223,D3224,...,D3238
Line 4: (empty — all commas)
Line 5: 2026/04/24 17:03:45.402,0,6876,...
Line 6: (data continues)
```

---

## 🔎 PART 4 — Duplicate Detection

Every upload is checked for conflicts before saving.

### What Gets Checked

**Check 1 — Same Date Label**
```
If 'May 01' already exists in history:
→ Shows warning with cycle counts
→ Tells you exactly what will be replaced
→ You decide whether to proceed
```

**Check 2 — Row Level Duplication (All Columns)**
```
Compares new cycles against ALL historical cycles
using all 15 feature columns:
→ run_mode
→ fwd_torque_max, fwd_torque_mean, fwd_torque_std
→ fwd_velocity_max, fwd_duration
→ settle_torque_mean, settle_torque_std, settle_duration
→ ret_torque_min, ret_torque_mean, ret_torque_std
→ ret_velocity_min, ret_duration
→ total_duration

Each row gets a unique fingerprint from all columns.
If fingerprints match → exact duplicate detected.
Shows: how many duplicates, what percentage.
```

**Check 3 — Timestamp Overlap**
```
Checks if new data timestamps fall within
an already-recorded time period.
Catches: same data uploaded with different label.
Shows: which existing days overlap.
```

**Check 4 — Fewer Cycles Than Existing**
```
If new upload has <50% of existing cycles for same date:
→ Warning that partial data may be uploaded
→ Example: 10 files uploaded instead of 400
```

**Check 5 — Too Few Cycles**
```
If less than 5 cycles extracted:
→ Warning that results may be unreliable
```

**Check 6 — Torque Direction Check**
```
Return torque should always be negative.
If positive detected → flags wrong machine or direction.
```

**Check 7 — New Mode Detected**
```
If a mode is seen for the first time:
→ Info message shown
→ Baseline created automatically from new data
```

### Severity Levels
```
🔴 HIGH   — Strong evidence of duplication or conflict
🟡 MEDIUM — Partial duplication, worth checking
🟡 LOW    — Minor overlap, likely safe
```

### Save Flow After Analysis
```
No issues → ✅ Save to History button
Has issues → ⚠️ Save Anyway (I understand) button
             + ❌ Discard button

Nothing saves automatically.
User always decides.
```

---

## 📈 PART 5 — Understanding Results

### Health Score
```
100 = Perfect match to healthy baseline
70+ = Normal — continue monitoring
50-69 = Caution — increase monitoring
30-49 = Warning — schedule maintenance
0-29  = Critical — stop machine!
```

### Run Modes
```
Mode 0 — Normal      → Daily production speed
Mode 2 — Maintenance → Reduced inspection speed
Mode 1 — Endurance   → Full speed stress test

Each mode has its own baseline and thresholds.
Never compared across modes.
```

### Sequence Stress
```
Seq 3110 — Forward Move:
  Stress = motor working harder going forward
  Check: belt tension, guide rails, lubrication

Seq 3115 — Forward Settle:
  Stress = settle time changed significantly
  Check: control system, position sensor

Seq 3125 — Return Move (most sensitive):
  Stress = return torque increasing
  Check: bearing wear, belt, mechanical load
```

---

## 🛠️ PART 6 — Code Updates for Developers

### When the CSV Format Changes

If the PLC system is updated and files have a new format:

**Step 1 — Identify what changed**
```
Open a new CSV file manually
Compare against expected format:
→ Which line number has column names now?
→ Which line does data start?
→ Are column names different?
→ Are new columns added or old ones removed?
```

**Step 2 — Update format constants in shuffler_app.py**

Find this section at the top of the file:
```python
# Expected CSV format constants
EXPECTED_HEADER_ROW   = 3      # ← change if column names moved
EXPECTED_DATA_START   = 4      # ← change if data starts on different line
REQUIRED_COLUMNS      = ["D3203","D3223","D3224","D3238"]  # ← add/change columns
TIMESTAMP_COLUMN      = "TIME (UTC+09:00)"  # ← update if timestamp column renamed
EXPECTED_MIN_COLUMNS  = 10     # ← update if file structure changed
EXPECTED_INTERVAL_COL = "INTERVAL[us]"  # ← update if column renamed
EXPECTED_MAGIC_HEADER = "[LOGGING]"     # ← update if header line changed
```

**Step 3 — Update torque conversion if needed**
```python
# Find this line in parse_csv():
df_clean["torque_nm"] = df_clean["torque_raw"] * 0.00064

# Change 0.00064 if PLC scaling factor changed
# This converts raw D3223 value to Newton-metres
```

**Step 4 — Update column mapping if renamed**
```python
# Find this in parse_csv():
df_clean = df_temp[[
    TIMESTAMP_COLUMN, "D3203", "D3223", "D3224", "D3238"
]].copy()
df_clean.columns = [
    "timestamp","sequence","torque_raw","velocity","run_mode"
]
# Update column names on both lines if PLC register names changed
```

**Step 5 — Commit to GitHub**
```
GitHub → shuffler_app.py → Edit (pencil icon)
Make changes → Commit changes
App updates automatically in 1-2 minutes
```

### When Alert Thresholds Need Adjustment

```python
# Find in analyse_one_mode():
b_mean = baseline_info["baseline_mean"]
b_std  = baseline_info["baseline_std"]
alert  = b_mean - (3 * b_std)   # ← change 3 to adjust sensitivity
warn   = b_mean - (2 * b_std)   # ← change 2 to adjust warning level

# 3 std = alert when 3x normal variation (strict)
# 4 std = alert only for extreme deviation (lenient)
# 2 std = alert more frequently (sensitive)
```

### When Stress Detection Threshold Needs Change

```python
# Find in analyse_one_mode() sequence analysis section:
if abs(pct) > 10:   # ← currently 10% deviation = stress

# Change to:
# 5  = very sensitive (flags small changes)
# 15 = less sensitive (only flags big changes)
```

### When Baseline Day Needs Reset (After Major Maintenance)

```python
# In mode_baselines.json on GitHub:
# Delete the entry for the mode you want to reset
# Example — delete Mode 2 entry:
{
  "0.0": { ... keep this ... },
  "1.0": { ... keep this ... }
  // "2.0" entry deleted — will recreate on next upload
}

# On next upload, app creates new baseline
# from the first day of new data
```

### When a New Sequence Number Appears

```python
# Find SEQ_NAMES dictionary at top of file:
SEQ_NAMES = {
    3000:"Standby", 3100:"Ready/Idle",
    3110:"Forward Move", 3115:"Forward Settle",
    3120:"Forward Hold", 3125:"Return Move",
    3130:"Return Settle"
}

# Add new sequence:
SEQ_NAMES = {
    ...existing entries...
    3200:"New Sequence Name"   # ← add here
}
```

### When Duplicate Detection Columns Change

```python
# Find in check_duplicates():
compare_cols = [
    "run_mode",
    "fwd_torque_max", "fwd_torque_mean", ...
    # Add or remove columns here as needed
]
```

---

## 📋 PART 7 — Current Baseline Values

### Mode 0 — Normal (Baseline: Apr 24, 41 cycles)
```
Return torque mean:  -0.1245 Nm
Warning threshold:   -0.1305 Nm
Alert threshold:     -0.1335 Nm
```

### Mode 2 — Maintenance (Baseline: Apr 24, 200 cycles)
```
Return torque mean:  -0.1269 Nm
Warning threshold:   -0.1318 Nm
Alert threshold:     -0.1343 Nm
```

### Mode 1 — Endurance (Baseline: Apr 25)
```
Only 1 cycle as baseline — needs more data
Full endurance run available: Apr 27 (81 cycles)
```

---

## 📉 PART 8 — Current Findings

### Mode 2 — Maintenance Trend
```
Apr 24: -0.1269 Nm  🟢 Healthy baseline
Apr 25: -0.1292 Nm  🟢 Normal
Apr 29: -0.1306 Nm  🟡 Caution

Degradation rate: -0.00074 Nm/day
Estimated alert:  ~May 4, 2026
```

### Stress Location
```
Seq 3110 Forward Move: +14.9% torque, +16.8% duration
Seq 3115 Forward Settle: -26.5% duration
Seq 3125 Return Move: ✅ Normal

→ Problem is in forward path only
→ Inspect: belt, ball screw, guide rails
```

---

## 🔁 PART 9 — Rebuilding From Scratch

If historical data is lost:

```
1. Open Google Colab
2. Mount Google Drive
3. Load all CSV folders using the pipeline code
4. Extract cycles for all days
5. Save all_cycles_4days.csv and mode_baselines.json
6. Upload both files to GitHub repo
7. App uses them automatically
```

Full pipeline code is in the original Colab notebook saved to Google Drive.

---

## 🔧 PART 10 — Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Historical data not found | Files missing from GitHub | Upload all_cycles_4days.csv and mode_baselines.json |
| All files fail validation | Wrong folder or file format changed | Check Data Preview tab for specific error |
| No cycles found | Machine was idle (only seq 3100) | Machine was not running — normal |
| Health score seems wrong | Wrong mode comparison | Each mode has own baseline — check mode label |
| 0 cycles for a mode | Mode not in uploaded data | Normal — only modes present are shown |
| App slow with 400 files | Normal for large uploads | Wait 2-5 minutes for spinner |
| App not updating after code change | Streamlit cache | Go to share.streamlit.io → Reboot app |
| Duplicate detected wrongly | Same data, different label | Use consistent date labels always |

---

## 🗃️ Historical Data Summary

| Day | Mode 0 | Mode 1 | Mode 2 | Total |
|---|---|---|---|---|
| Apr 24 | 41 | 0 | 200 | 241 |
| Apr 25 | 0 | 1 | 400 | 401 |
| Apr 27 | 0 | 81 | 0 | 81 |
| Apr 29 | 1 | 0 | 585 | 586 |
| **Total** | **42** | **82** | **1185** | **1309** |

---

*Built with Python · Streamlit · Scikit-learn · Plotly*
*1,309 cycles · 3 modes tracked independently · Full duplicate detection*
