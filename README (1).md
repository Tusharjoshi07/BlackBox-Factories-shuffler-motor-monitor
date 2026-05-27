# 🔧 Shuffler Motor Health Monitor
### Predictive Maintenance System — Blackbox Factories

A machine learning system that monitors servo motor health by mode, detects anomalies, and predicts when maintenance is needed — using PLC sensor data from Mitsubishi servo systems.

---

## 🔗 Live App
👉 [Click here to open the app](https://your-app-link.streamlit.app)
*(Update this link after deployment)*

---

## 📌 What This System Does

```
1. Load daily CSV files from the PLC system
2. Preview data quality before running analysis
3. Extract machine cycles automatically
4. SEPARATE cycles by run mode automatically
5. Compare each mode against its OWN baseline
6. Detect which sequence number has stress
7. Predict days until maintenance needed per mode
8. Save all data to history for better predictions
```

---

## 🗂️ Files in This Repository

| File | Purpose |
|---|---|
| `shuffler_app.py` | Main Streamlit web application |
| `all_cycles_4days.csv` | Historical cycle data — ALL modes combined |
| `mode_baselines.json` | Baseline stats per mode (auto-updated) |
| `requirements.txt` | Python libraries needed |
| `README.md` | This file |

---

## 🚀 PART 1 — First Time Setup (Do This Once)

### Step 1 — Create GitHub Account
Go to **github.com** → Sign Up → Free account

### Step 2 — Fork This Repository
1. Go to the original repo link shared with you
2. Click **Fork** button (top right)
3. Click **Create Fork**
4. Now you have your own copy at:
   `github.com/YOUR_USERNAME/shuffler-motor-monitor`

### Step 3 — Create Streamlit Account
Go to **share.streamlit.io** → Sign in with GitHub

### Step 4 — Deploy the App
1. Click **New App**
2. Select your forked repository
3. Branch: `main`
4. Main file path: `shuffler_app.py`
5. Click **Deploy**
6. Wait 2-3 minutes
7. App is live at:
   `https://YOUR_USERNAME-shuffler-motor-monitor.streamlit.app`

### Step 5 — Save Your App Link
Share this link with your team.
**No Python or coding knowledge needed to use the app.**

---

## 📊 PART 2 — Daily Usage (Every Day)

### How to Use — Step by Step

```
1. Open your app link in browser

2. In the sidebar:
   → Enter today's date label (e.g. May 01)
   → Choose data source (Upload or Folder Path)

3. Load your CSV files:
   → Upload: Click Browse → select ALL CSV files
             (can select 400+ files at once)
   → Folder: Paste the full folder path

4. Go to DATA PREVIEW tab:
   → Check data quality
   → See which modes are in the data
   → See which sequences are present
   → Confirm data looks correct

5. Go to HEALTH ANALYSIS tab:
   → Click Run Health Analysis
   → System separates cycles by mode automatically
   → Shows results for each mode independently

6. Read the report for each mode
```

---

## 🔍 PART 3 — Understanding Run Modes

This is the most important concept.
The machine runs in different modes — each mode
runs at different speed and torque levels.
**Each mode is compared only against its own history.**

### Mode 0 — Normal Mode
```
Daily production speed
Baseline torque: -0.1245 Nm
Warning:         -0.1305 Nm
Alert:           -0.1335 Nm
```

### Mode 2 — Maintenance Mode
```
Reduced speed for inspection
Baseline torque: -0.1269 Nm
Warning:         -0.1318 Nm
Alert:           -0.1343 Nm
```

### Mode 1 — Endurance Mode
```
Full speed stress testing
Velocity 6x higher than normal
Torque profile completely different
Tracked separately — NOT compared to Normal/Maintenance
```

### Why Modes Must Be Separate
```
Mode 0 velocity:  ~86 units/second
Mode 1 velocity:  ~503 units/second
Mode 2 velocity:  ~88 units/second

Mode 1 forward torque: 0.14 Nm
Mode 0 forward torque: 0.03 Nm

Comparing Mode 1 to Mode 2 baseline
would show false 400% stress alert.
The system handles this automatically.
```

---

## 📈 PART 4 — Understanding the Results

### Health Score
```
100 = Perfect (matches healthy baseline)
70-99 = Normal (continue monitoring)
50-69 = Caution (increase monitoring)
30-49 = Warning (schedule maintenance)
0-29  = Critical (stop machine!)
```

### Status Meanings
```
🟢 NORMAL   → Healthy, continue as usual
🟡 CAUTION  → Early signs, monitor daily
🟡 WARNING  → Schedule maintenance within X days
🔴 CRITICAL → Stop machine, inspect immediately
```

### Sequence Numbers Explained
```
Sequence 3110 — Forward Move
  Motor pushes arm forward fast
  High torque and velocity expected
  Stress here = friction on forward path

Sequence 3115 — Forward Settle
  Arm reaches destination and slows down
  Torque and velocity drop to near zero
  Stress here = control system issue

Sequence 3125 — Return Move (MOST SENSITIVE)
  Motor pulls arm back
  Negative torque = braking force
  Gets more negative as machine wears
  First place failure shows up
```

### Sequence Stress Causes
```
⚠️ Seq 3110 STRESS (Forward Move):
   → Motor working harder going forward
   → Check: belt tension, guide rails,
             ball screw lubrication

⚠️ Seq 3115 STRESS (Forward Settle):
   → Settle time changed significantly
   → Check: control system, position sensor

⚠️ Seq 3125 STRESS (Return Move):
   → Return torque increasing
   → Check: bearing wear, belt condition,
             mechanical load on return path
```

---

## 🔄 PART 5 — How Data Grows Over Time

### What Happens Each Time You Run Analysis
```
New CSV files uploaded
        ↓
System extracts cycles
        ↓
Cycles separated by mode automatically
        ↓
Each mode compared to its own baseline
        ↓
New cycles added to all_cycles_4days.csv
        ↓
mode_baselines.json updated
        ↓
Next run has more historical data
        ↓
Predictions become more accurate
```

### Prediction Accuracy Over Time
```
Week 1  (4-7 days):   Basic trend, rough estimate
Week 2  (8-14 days):  Better trend line
Month 1 (30 days):    Reliable predictions
Month 3 (90 days):    High confidence alerts
Year 1  (365 days):   Seasonal patterns detected
```

### What Improves Automatically
```
✅ Trend line becomes more accurate
✅ Thresholds recalibrated from more data
✅ Anomaly detection improves
✅ Days-to-alert estimate becomes reliable
✅ New modes get their own baseline on first run
```

---

## 🛠️ PART 6 — Technical Updates (For Developers)

### Files to Update and When

| Situation | File to Update | What to Change |
|---|---|---|
| Alert too sensitive | `shuffler_app.py` | Change `3 * b_std` to `4 * b_std` |
| Alert not sensitive enough | `shuffler_app.py` | Change `3 * b_std` to `2 * b_std` |
| New mode appears | Nothing | App creates baseline automatically |
| Machine serviced — reset baseline | `mode_baselines.json` | Delete the mode entry |
| Historical data corrupted | `all_cycles_4days.csv` | Re-extract from Colab (see Part 8) |

### How to Edit Code on GitHub
```
1. Go to your GitHub repo
2. Click shuffler_app.py
3. Click pencil icon (Edit this file)
4. Make your change
5. Click Commit changes
6. Streamlit updates automatically in 1-2 mins
```

### Key Code Locations

#### Change Alert Threshold Sensitivity
```python
# In shuffler_app.py — find this section:
alert = b_mean - (3 * b_std)  # Alert level
warn  = b_mean - (2 * b_std)  # Warning level

# 3 = flags when 3x normal variation
# Lower number = more sensitive (more alerts)
# Higher number = less sensitive (fewer alerts)
```

#### Change Stress Detection Sensitivity
```python
# Find this line:
if abs(pct) > 10:  # Flag if >10% from baseline

# Change 10 to:
# 5  = very sensitive (flags small changes)
# 15 = less sensitive (only big changes)
```

#### Reset Baseline for One Mode
```python
# In mode_baselines.json on GitHub
# Delete the entry for that mode number
# e.g. delete the "2.0" block
# App will create new baseline on next run
```

---

## 📋 PART 7 — Current Baseline Values

These are the healthy reference values from April 24, 2026.
Everything is compared against these numbers.

### Mode 0 — Normal (Baseline: Apr 24, 41 cycles)
```
Return torque mean:  -0.1245 Nm
Return torque std:    0.0030 Nm
Warning threshold:   -0.1305 Nm
Alert threshold:     -0.1335 Nm
```

### Mode 2 — Maintenance (Baseline: Apr 24, 200 cycles)
```
Return torque mean:  -0.1269 Nm
Return torque std:    0.0025 Nm
Warning threshold:   -0.1318 Nm
Alert threshold:     -0.1343 Nm
```

### Mode 1 — Endurance (Baseline: Apr 25, 1 cycle)
```
Note: Only 1 cycle available as baseline
      First full endurance run = Apr 27 (81 cycles)
      Baseline will improve as more data collected
```

---

## 📉 PART 8 — Current Findings (Apr 24 to Apr 29)

### Mode 2 — Maintenance Trend
```
Apr 24: -0.1269 Nm  🟢 Healthy baseline
Apr 25: -0.1292 Nm  🟢 Normal
Apr 29: -0.1306 Nm  🟡 Caution

Rate: -0.00074 Nm per day
Days to alert: ~5 days from Apr 29
Estimated maintenance: ~May 4, 2026
```

### Stress Found At
```
Sequence 3110 (Forward Move):
→ Torque +14.9% above baseline
→ Duration +16.8% longer per cycle
→ Motor working harder on forward stroke

Sequence 3115 (Forward Settle):
→ Duration -26.5% shorter
→ Machine settling faster than normal

Sequence 3125 (Return Move):
→ All values within normal range ✅
```

### Recommendation
```
Inspect forward mechanical path this week:
→ Belt tension on forward stroke
→ Ball screw lubrication
→ Guide rail alignment
→ Motor coupling condition
```

---

## 🔁 PART 9 — Rebuilding From Scratch

Use this if historical data is lost or corrupted.

### Step 1 — Open Google Colab
Create new notebook at colab.research.google.com

### Step 2 — Mount Drive and Load All Folders
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os

# Define your folders
day_folders = {
    'Apr 24': '/content/drive/MyDrive/YOUR_FOLDER_24',
    'Apr 25': '/content/drive/MyDrive/YOUR_FOLDER_25',
    'Apr 27': '/content/drive/MyDrive/YOUR_FOLDER_27',
    'Apr 29': '/content/drive/MyDrive/YOUR_FOLDER_29',
    # Add more days here as you collect data
}
```

### Step 3 — Run the Pipeline
Load → Extract cycles → Separate modes → Save

Full pipeline code is in the original Colab notebook.

### Step 4 — Upload New Files to GitHub
```
Download from Google Drive:
→ all_cycles_4days.csv
→ mode_baselines.json

Upload to GitHub repo
Streamlit uses these files automatically
```

---

## 🔧 PART 10 — Troubleshooting

### "Historical data not found"
Upload `all_cycles_4days.csv` and `mode_baselines.json` to GitHub repo

### "No cycles found"
Data is all idle (sequence 3100 only) — machine was not running.
Check Data Preview tab to confirm.

### "Mode X has no baseline"
First time this mode is seen. App creates baseline from current data automatically. Results next run will be more accurate.

### Health score seems wrong for a mode
Check that you are comparing the right mode.
Mode 1 (Endurance) at full speed will always
look different from Mode 2 (Maintenance).
This is expected — not a bug.

### App slow with 400 files
Normal — 400 CSV files take 2-5 minutes.
Wait for spinner to complete.

### Streamlit not updating after code edit
Go to share.streamlit.io → Your app → Reboot app

---

## 📞 How the System Works — Simple Summary

```
Every day:
1. Machine runs → PLC logs CSV files
2. Upload folder to app
3. App reads files → extracts cycles
4. Separates Mode 0, Mode 1, Mode 2
5. Compares each mode to ITS OWN history
6. Shows health score and stress location
7. Saves today's data for future comparison
8. Predictions improve every day
```

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
*Analysis period: April 24–29, 2026*
*1,309 cycles analyzed · 3 modes tracked independently*
