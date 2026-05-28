import streamlit as st
import pandas as pd
import numpy as np
import os, io, json, hashlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Shuffler Motor Health Monitor",page_icon="🔧",layout="wide")

MODE_NAMES={0.0:"Normal",1.0:"Endurance",2.0:"Maintenance"}
MODE_COLORS={0.0:"#2196F3",1.0:"#FF9800",2.0:"#4CAF50"}
SEQ_NAMES={3000:"Standby",3100:"Ready/Idle",3110:"Forward Move",3115:"Forward Settle",3120:"Forward Hold",3125:"Return Move",3130:"Return Settle"}
SEQ_COLORS={3110:"#E74C3C",3115:"#F39C12",3120:"#F1C40F",3125:"#2980B9",3130:"#27AE60"}
EXPECTED_HEADER_ROW=3;EXPECTED_DATA_START=4
REQUIRED_COLUMNS=["D3203","D3223","D3224","D3238"]
TIMESTAMP_COLUMN="TIME (UTC+09:00)";EXPECTED_MIN_COLUMNS=10
EXPECTED_INTERVAL_COL="INTERVAL[us]";EXPECTED_MAGIC_HEADER="[LOGGING]"

def safe_int(mode):
    try: return int(mode)
    except: return str(mode)

@st.cache_data
def load_all_data():
    try:
        df=pd.read_csv("all_cycles_4days.csv")
        df["start_time"]=pd.to_datetime(df["start_time"])
        return df
    except: return None

@st.cache_data
def load_baselines():
    try:
        with open("mode_baselines.json") as f:
            raw=json.load(f)
        return {float(k):v for k,v in raw.items()}
    except: return {}

def validate_file_format(content,filename):
    issues=[];lines=content.split("\n")
    if len(lines)<6:
        issues.append(f"Only {len(lines)} lines — expected at least 6.")
        return False,issues
    if not lines[0].strip().startswith(EXPECTED_MAGIC_HEADER):
        issues.append(f"Line 0 should start with '{EXPECTED_MAGIC_HEADER}'.")
    try:
        col_names=lines[EXPECTED_HEADER_ROW].strip().split(",")
        if len(col_names)<EXPECTED_MIN_COLUMNS:
            issues.append(f"Only {len(col_names)} columns — expected {EXPECTED_MIN_COLUMNS}+.")
        if TIMESTAMP_COLUMN not in col_names:
            issues.append(f"Missing timestamp column '{TIMESTAMP_COLUMN}'.")
        missing=[c for c in REQUIRED_COLUMNS if c not in col_names]
        if missing: issues.append(f"Missing sensor columns: {missing}.")
        if EXPECTED_INTERVAL_COL not in col_names:
            issues.append(f"Missing '{EXPECTED_INTERVAL_COL}'.")
    except Exception as e:
        issues.append(f"Cannot parse columns: {e}"); return False,issues
    if lines[4].strip().replace(",","")!="":
        issues.append("Line 4 should be empty separator.")
    try:
        first=lines[EXPECTED_DATA_START].strip()
        if first=="": issues.append("No data after headers.")
    except Exception as e: issues.append(f"Cannot read first data row: {e}")
    return len(issues)==0,issues

def validate_all_files(files_contents):
    valid,invalid=[],[]
    for filename,content in files_contents:
        ok,issues=validate_file_format(content,filename)
        if ok: valid.append(filename)
        else: invalid.append({"filename":filename,"issues":issues})
    return valid,invalid

def parse_csv(content,day_label):
    try:
        lines=content.split("\n")
        col_names=lines[EXPECTED_HEADER_ROW].strip().split(",")
        df_temp=pd.read_csv(io.StringIO("\n".join(lines[EXPECTED_DATA_START:])),names=col_names,on_bad_lines="skip")
        if not all(c in df_temp.columns for c in REQUIRED_COLUMNS): return None
        df_temp=df_temp.dropna(subset=["D3203"]).reset_index(drop=True)
        df_clean=df_temp[[TIMESTAMP_COLUMN,"D3203","D3223","D3224","D3238"]].copy()
        df_clean.columns=["timestamp","sequence","torque_raw","velocity","run_mode"]
        df_clean["timestamp"]=pd.to_datetime(df_clean["timestamp"],errors="coerce")
        df_clean["torque_nm"]=df_clean["torque_raw"]*0.00064
        df_clean["day"]=day_label
        return df_clean
    except: return None

def load_uploaded(files,day_label):
    file_contents=[(f.name,f.read().decode("utf-8",errors="ignore")) for f in files]
    valid_files,invalid_files=validate_all_files(file_contents)
    all_data=[parse_csv(c,day_label) for fn,c in file_contents if fn in valid_files and parse_csv(c,day_label) is not None]
    if not all_data: return None,invalid_files,valid_files
    return pd.concat(all_data,ignore_index=True).sort_values("timestamp").reset_index(drop=True),invalid_files,valid_files

def load_folder(folder_path,day_label):
    all_files=sorted([f for f in os.listdir(folder_path) if f.endswith(".CSV") or f.endswith(".csv")])
    file_contents=[]
    for fn in all_files:
        try:
            with open(os.path.join(folder_path,fn),"r",encoding="utf-8",errors="ignore") as f:
                file_contents.append((fn,f.read()))
        except: continue
    valid_files,invalid_files=validate_all_files(file_contents)
    all_data=[parse_csv(c,day_label) for fn,c in file_contents if fn in valid_files and parse_csv(c,day_label) is not None]
    if not all_data: return None,invalid_files,valid_files
    return pd.concat(all_data,ignore_index=True).sort_values("timestamp").reset_index(drop=True),invalid_files,valid_files

def extract_cycles(df_all,day_label):
    active=[3110.0,3115.0,3120.0,3125.0,3130.0]
    df_active=df_all[df_all["sequence"].isin(active)].copy().reset_index(drop=True)
    cycles,cycle_id=[],0
    seq=df_active["sequence"].values;n,i=len(seq),0
    while i<n-1:
        if seq[i]==3110.0:
            fwd_start=i;j=i+1
            while j<n and seq[j]==3110.0: j+=1
            if j>=n or seq[j]!=3115.0: i=j;continue
            settle_start=j
        elif seq[i]==3115.0 and (i==0 or seq[i-1]==3130.0):
            fwd_start=settle_start=i
        else: i+=1;continue
        k=settle_start+1
        while k<n and seq[k]==3115.0: k+=1
        if k<n and seq[k]==3120.0:
            while k<n and seq[k]==3120.0: k+=1
        if k>=n or seq[k]!=3125.0: i=k;continue
        return_start=k;m=k+1
        while m<n and seq[m]==3125.0: m+=1
        if m>=n or seq[m]!=3130.0: i=m;continue
        cycle_end=m
        fwd=df_active.iloc[fwd_start:settle_start]
        settle=df_active.iloc[settle_start:return_start]
        ret=df_active.iloc[return_start:cycle_end]
        if len(settle)<3 or len(ret)<3: i=cycle_end+1;continue
        cycles.append({
            "cycle_id":cycle_id,"day":day_label,
            "start_time":df_active.iloc[fwd_start]["timestamp"],
            "run_mode":df_active.iloc[fwd_start]["run_mode"],
            "fwd_torque_max":fwd["torque_nm"].max() if len(fwd)>0 else np.nan,
            "fwd_torque_mean":fwd["torque_nm"].mean() if len(fwd)>0 else np.nan,
            "fwd_torque_std":fwd["torque_nm"].std() if len(fwd)>0 else np.nan,
            "fwd_velocity_max":fwd["velocity"].max() if len(fwd)>0 else np.nan,
            "fwd_duration":len(fwd),
            "settle_torque_mean":settle["torque_nm"].mean(),
            "settle_torque_std":settle["torque_nm"].std(),
            "settle_duration":len(settle),
            "ret_torque_min":ret["torque_nm"].min(),
            "ret_torque_mean":ret["torque_nm"].mean(),
            "ret_torque_std":ret["torque_nm"].std(),
            "ret_velocity_min":ret["velocity"].min(),
            "ret_duration":len(ret),
            "total_duration":cycle_end-fwd_start,
        })
        cycle_id+=1;i=cycle_end+1
    return pd.DataFrame(cycles)

def check_duplicates(new_cycles,historical,day_label):
    issues,warnings=[],[]
    if len(new_cycles)==0: return issues,warnings
    compare_cols=["run_mode","fwd_torque_max","fwd_torque_mean","fwd_torque_std","fwd_velocity_max","fwd_duration","settle_torque_mean","settle_torque_std","settle_duration","ret_torque_min","ret_torque_mean","ret_torque_std","ret_velocity_min","ret_duration","total_duration"]
    def fp(df):
        cols=[c for c in compare_cols if c in df.columns]
        return set(df[cols].round(6).apply(lambda r:hashlib.md5(str(tuple(r.values)).encode()).hexdigest(),axis=1).tolist())
    existing_days=historical["day"].unique().tolist()
    if day_label in existing_days:
        ex=historical[historical["day"]==day_label]
        issues.append({"type":"DATE_EXISTS","severity":"HIGH","message":f"'{day_label}' already exists with {len(ex)} cycles.","existing_cycles":len(ex),"new_cycles":len(new_cycles),"action":f"Will REPLACE {len(ex)} cycles with {len(new_cycles)} new."})
    n_dupes=len(fp(new_cycles)&fp(historical))
    if n_dupes>0:
        pct=n_dupes/len(new_cycles)*100
        issues.append({"type":"ROW_DUPLICATES","severity":"HIGH" if pct>50 else "MEDIUM" if pct>10 else "LOW","message":f"{n_dupes} cycles ({pct:.1f}%) are IDENTICAL to history (all {len(compare_cols)} columns checked).","n_dupes":n_dupes,"dupe_pct":round(pct,1),"action":"Likely same data uploaded twice."})
    ns=pd.to_datetime(new_cycles["start_time"]).min();ne=pd.to_datetime(new_cycles["start_time"]).max()
    overlap=historical[(pd.to_datetime(historical["start_time"])>=ns)&(pd.to_datetime(historical["start_time"])<=ne)]
    if len(overlap)>0 and day_label not in existing_days:
        issues.append({"type":"TIMESTAMP_OVERLAP","severity":"HIGH","message":f"Timestamps overlap {len(overlap)} existing cycles.","overlapping_days":overlap["day"].unique().tolist(),"overlap_count":len(overlap),"new_start":str(ns),"new_end":str(ne),"action":f"Data may exist under: {overlap['day'].unique().tolist()}."})
    if day_label in existing_days:
        ec=len(historical[historical["day"]==day_label])
        if len(new_cycles)<ec*0.5:
            warnings.append({"type":"FEWER_CYCLES","message":f"New: {len(new_cycles)} cycles vs existing {ec}. Possible partial upload."})
    if len(new_cycles)<5:
        warnings.append({"type":"TOO_FEW_CYCLES","message":f"Only {len(new_cycles)} cycles found."})
    if new_cycles["ret_torque_min"].mean()>0:
        warnings.append({"type":"TORQUE_DIRECTION","message":"Return torque is positive — expected negative."})
    truly_new=set(new_cycles["run_mode"].unique())-set(historical["run_mode"].unique())
    if truly_new:
        warnings.append({"type":"NEW_MODE","message":f"New mode(s): {[MODE_NAMES.get(m,str(m)) for m in truly_new]}. Baseline created automatically."})
    return issues,warnings

def analyse_one_mode(new_cycles_mode,historical_mode,baseline_info,day_label,mode):
    if len(new_cycles_mode)==0: return None
    b_mean=baseline_info["baseline_mean"];b_std=baseline_info["baseline_std"]
    alert=baseline_info["alert_threshold"];warn=baseline_info["warning_threshold"]
    ret_mean=new_cycles_mode["ret_torque_min"].mean();fwd_mean=new_cycles_mode["fwd_torque_max"].mean()
    health=max(0,min(100,(ret_mean-alert)/(b_mean-alert)*100))
    if historical_mode is not None and len(historical_mode)>0:
        daily=historical_mode.groupby("day")["ret_torque_min"].mean()
        daily=pd.concat([daily,pd.Series({day_label:ret_mean})])
        daily=daily[~daily.index.duplicated(keep="last")]
        t_vals=daily.values;t_nums=list(range(len(t_vals)))
        if len(t_vals)>=2:
            z=np.polyfit(t_nums,t_vals,1);slope=z[0]
            days_to=int((alert-ret_mean)/slope) if slope<0 else 999
        else: slope=0;days_to=999
    else: slope=0;days_to=999
    if ret_mean<alert: status="CRITICAL"
    elif ret_mean<warn: status="WARNING"
    elif health<80: status="CAUTION"
    else: status="NORMAL"
    base_cycles=(historical_mode[historical_mode["day"]==baseline_info["baseline_day"]] if historical_mode is not None and len(historical_mode)>0 else new_cycles_mode)
    seq_analysis={}
    for seq_key,metrics in [
        ("3110",[("torque_mean","fwd_torque_mean","fwd_torque_mean"),("torque_max","fwd_torque_max","fwd_torque_max"),("duration","fwd_duration","fwd_duration")]),
        ("3115",[("duration","settle_duration","settle_duration")]),
        ("3125",[("torque_min","ret_torque_min","ret_torque_min"),("velocity","ret_velocity_min","ret_velocity_min"),("duration","ret_duration","ret_duration")]),
    ]:
        names={"3110":"Forward Move","3115":"Forward Settle","3125":"Return Move"}
        alerts_seq=[]
        for label,b_col,c_col in metrics:
            if len(base_cycles)==0: continue
            b=float(base_cycles[b_col].mean());c=float(new_cycles_mode[c_col].mean())
            if b==0 or np.isnan(b) or np.isnan(c): continue
            pct=((c-b)/abs(b))*100
            if abs(pct)>10: alerts_seq.append({"metric":label,"baseline":round(b,4),"current":round(c,4),"change_pct":round(pct,1)})
        seq_analysis[seq_key]={"name":names[seq_key],"status":"STRESS" if alerts_seq else "NORMAL","alerts":alerts_seq}
    return {"mode":mode,"mode_name":MODE_NAMES.get(mode,"Unknown"),"cycles":len(new_cycles_mode),"health_score":round(health,1),"status":status,"ret_torque":round(ret_mean,4),"fwd_torque":round(fwd_mean,4),"alert_threshold":round(alert,4),"warning_threshold":round(warn,4),"days_to_alert":days_to,"trend_per_day":round(slope,6),"sequence_analysis":seq_analysis,"baseline_day":baseline_info["baseline_day"],"baseline_mean":round(b_mean,4),"new_cycles_df":new_cycles_mode,"base_cycles_df":base_cycles}

def auto_analyse_latest(historical,baselines):
    if historical is None or len(historical)==0: return {},""
    latest_day=historical.sort_values("start_time",ascending=False)["day"].iloc[0]
    latest_data=historical[historical["day"]==latest_day]
    results={}
    for mode in sorted([m for m in latest_data["run_mode"].unique() if not pd.isna(m)]):
        new_mode=latest_data[latest_data["run_mode"]==mode].copy()
        b_info=baselines.get(mode)
        if b_info is None: continue
        hist_mode=historical[(historical["run_mode"]==mode)&(historical["day"]!=latest_day)].copy()
        result=analyse_one_mode(new_mode,hist_mode,b_info,latest_day,mode)
        if result: results[mode]=result
    return results,latest_day

def show_root_cause(result):
    if result is None: return
    stressed={k:v for k,v in result["sequence_analysis"].items() if v["status"]=="STRESS"}
    if not stressed:
        st.success(f"✅ No stress detected — Machine operating normally in {result['mode_name']} mode.")
        return
    st.error("🔴 ROOT CAUSE — Exact Stress Location")
    for seq_key,seq_data in stressed.items():
        seq_name=seq_data["name"]
        seq_icon={"3110":"🔴","3115":"🟠","3125":"🔵"}.get(seq_key,"⚠️")
        st.markdown(f"### {seq_icon} Sequence **{seq_key}** — {seq_name} is causing issues")
        if seq_key=="3110":
            st.markdown("> 🔍 Motor struggling during **forward stroke** — increased friction or resistance on forward path.")
            st.markdown("> 🔧 **Inspect:** Ball screw, belt tension, guide rail alignment, forward-side bearing.")
        elif seq_key=="3115":
            st.markdown("> 🔍 Arm spending different time **settling at destination** — control system or position sensor issue.")
            st.markdown("> 🔧 **Inspect:** Position sensor, control parameters, mechanical stop condition.")
        elif seq_key=="3125":
            st.markdown("> 🔍 Motor working harder during **return stroke** — most sensitive indicator of mechanical wear.")
            st.markdown("> 🔧 **Inspect:** Return-side bearing, belt tension, ball screw lubrication, mechanical load.")
        st.markdown("**Exact metrics:**")
        for alert in seq_data["alerts"]:
            direction="increased ↑" if alert["change_pct"]>0 else "decreased ↓"
            severity="🔴" if abs(alert["change_pct"])>20 else "🟡"
            explain={"torque_mean":"Average motor effort","torque_max":"Peak motor effort","torque_min":"Peak return braking force","duration":"Time taken for this stage","velocity":"Peak speed reached"}.get(alert["metric"],alert["metric"])
            st.markdown(f"- {severity} **{explain}** has {direction} by **{abs(alert['change_pct']):.1f}%** — Baseline: `{alert['baseline']} Nm` → Now: `{alert['current']} Nm`")

def run_anomaly_detection(cycles_df):
    features=["ret_torque_min","ret_torque_mean","ret_torque_std","ret_velocity_min","settle_duration","total_duration"]
    X=cycles_df[features].dropna()
    if len(X)<10: return cycles_df.copy(),0
    scaler=StandardScaler();X_s=scaler.fit_transform(X)
    iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=200)
    preds=iso.fit_predict(X_s);scores=iso.score_samples(X_s)
    result=cycles_df.copy()
    result.loc[X.index,"anomaly"]=preds;result.loc[X.index,"anomaly_score"]=scores
    return result,int((preds==-1).sum())

def chart_raw_signal(df_raw,day_label,chart_key="raw"):
    active=df_raw[df_raw["sequence"].isin([3110.0,3115.0,3120.0,3125.0,3130.0])].copy()
    if len(active)==0: return None
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=["Sequence Numbers","Torque (Nm)","Velocity"],vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=active["timestamp"],y=active["sequence"],mode="lines",name="Sequence",line=dict(color="#9B59B6",width=0.8)),row=1,col=1)
    fig.add_trace(go.Scatter(x=active["timestamp"],y=active["torque_nm"],mode="lines",name="Torque",line=dict(color="#E74C3C",width=0.8)),row=2,col=1)
    fig.add_hline(y=0,line_dash="dash",line_color="gray",line_width=0.5,row=2,col=1)
    fig.add_trace(go.Scatter(x=active["timestamp"],y=active["velocity"],mode="lines",name="Velocity",line=dict(color="#27AE60",width=0.8)),row=3,col=1)
    fig.add_hline(y=0,line_dash="dash",line_color="gray",line_width=0.5,row=3,col=1)
    fig.update_layout(title=f"Raw Signal — {day_label}",height=600,showlegend=False)
    return fig

def chart_one_cycle(df_raw,cycle_num=0,chart_key="cycle"):
    active=df_raw[df_raw["sequence"].isin([3110.0,3115.0,3120.0,3125.0,3130.0])].copy().reset_index(drop=True)
    if len(active)==0: return None
    seq_vals=active["sequence"].values
    cycle_starts=[0]+[i for i in range(1,len(seq_vals)) if seq_vals[i]==3110.0 and seq_vals[i-1]!=3110.0]
    if not cycle_starts: cycle_starts=[0]
    cs=cycle_starts[min(cycle_num,len(cycle_starts)-1)]
    next_s=[cp for cp in cycle_starts if cp>cs]
    ce=next_s[0] if next_s else min(cs+2500,len(active))
    cycle_data=active.iloc[cs:ce]
    if len(cycle_data)==0: return None
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=["Sequence State","Torque (Nm) by Sequence","Velocity by Sequence"],vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=cycle_data["timestamp"],y=cycle_data["sequence"],mode="lines",line=dict(color="#9B59B6",width=2),name="Sequence"),row=1,col=1)
    for seq in [3110.0,3115.0,3120.0,3125.0,3130.0]:
        seg=cycle_data[cycle_data["sequence"]==seq]
        if len(seg)==0: continue
        name=SEQ_NAMES.get(int(seq),str(int(seq)));color=SEQ_COLORS.get(int(seq),"#888")
        fig.add_trace(go.Scatter(x=seg["timestamp"],y=seg["torque_nm"],mode="lines",name=name,line=dict(color=color,width=2),legendgroup=name),row=2,col=1)
        fig.add_trace(go.Scatter(x=seg["timestamp"],y=seg["velocity"],mode="lines",name=name,line=dict(color=color,width=2),legendgroup=name,showlegend=False),row=3,col=1)
    fig.add_hline(y=0,line_dash="dash",line_color="gray",row=2,col=1)
    fig.add_hline(y=0,line_dash="dash",line_color="gray",row=3,col=1)
    fig.update_layout(title="One Complete Cycle — Colored by Sequence",height=650)
    return fig

def chart_multiday_trend(historical,baselines,key_prefix="trend"):
    figs={}
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)==0: continue
        b_info=baselines.get(mode)
        if b_info is None: continue
        alert=b_info["alert_threshold"];warn=b_info["warning_threshold"];b_mean=b_info["baseline_mean"]
        daily=mode_data.groupby("day").agg(ret_mean=("ret_torque_min","mean"),ret_std=("ret_torque_min","std"),cycles=("cycle_id","count")).reset_index()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=daily["day"],y=daily["ret_mean"],mode="lines+markers+text",name="Return Torque",line=dict(color=MODE_COLORS.get(mode,"blue"),width=3),marker=dict(size=12),text=[str(v) for v in daily["ret_mean"].round(4)],textposition="top center",error_y=dict(type="data",array=daily["ret_std"].tolist(),visible=True,color="rgba(0,0,0,0.3)")))
        fig.add_hline(y=alert,line_dash="dash",line_color="red",line_width=2,annotation_text=f"Alert {round(alert,4)}",annotation_position="right")
        fig.add_hline(y=warn,line_dash="dash",line_color="orange",line_width=2,annotation_text=f"Warning {round(warn,4)}",annotation_position="right")
        fig.add_hline(y=b_mean,line_dash="dot",line_color="green",line_width=1,annotation_text=f"Baseline {round(b_mean,4)}",annotation_position="right")
        if len(daily)>=2:
            z=np.polyfit(list(range(len(daily))),daily["ret_mean"].values,1);p=np.poly1d(z)
            fig.add_trace(go.Scatter(x=[daily["day"].iloc[0],daily["day"].iloc[-1]],y=[float(p(0)),float(p(len(daily)-1))],mode="lines",name=f"Trend (slope={z[0]:.6f})",line=dict(color="red",dash="dot",width=2)))
        fig.update_layout(title=f"Mode {safe_int(mode)} — {MODE_NAMES.get(mode)} Return Torque Trend",yaxis_title="Return Torque (Nm)",xaxis_title="Date",height=380)
        figs[mode]=fig
    return figs

def chart_line_by_day(historical,baselines,key_prefix="line"):
    figs={}
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)<5: continue
        b_info=baselines.get(mode)
        if b_info is None: continue
        alert=b_info["alert_threshold"];warn=b_info["warning_threshold"];b_mean=b_info["baseline_mean"]
        daily=mode_data.groupby("day").agg(ret_mean=("ret_torque_min","mean"),ret_min=("ret_torque_min","min"),ret_max=("ret_torque_min","max"),fwd_mean=("fwd_torque_max","mean"),cycles=("cycle_id","count")).reset_index()
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=["Return Torque (Nm) — Mean per Day","Forward Torque (Nm) — Mean per Day"],vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=list(daily["day"])+list(daily["day"][::-1]),y=list(daily["ret_max"])+list(daily["ret_min"][::-1]),fill="toself",fillcolor="rgba(100,100,200,0.15)",line=dict(color="rgba(255,255,255,0)"),name="Min-Max Range"),row=1,col=1)
        fig.add_trace(go.Scatter(x=daily["day"],y=daily["ret_mean"],mode="lines+markers+text",name="Return Torque Mean",line=dict(color=MODE_COLORS.get(mode,"blue"),width=3),marker=dict(size=10),text=[str(round(v,4)) for v in daily["ret_mean"]],textposition="top center"),row=1,col=1)
        fig.add_hline(y=alert,line_dash="dash",line_color="red",line_width=2,annotation_text=f"Alert {round(alert,4)}",annotation_position="right",row=1,col=1)
        fig.add_hline(y=warn,line_dash="dash",line_color="orange",line_width=1,annotation_text=f"Warning {round(warn,4)}",annotation_position="right",row=1,col=1)
        fig.add_hline(y=b_mean,line_dash="dot",line_color="green",line_width=1,annotation_text=f"Baseline {round(b_mean,4)}",annotation_position="right",row=1,col=1)
        fig.add_trace(go.Scatter(x=daily["day"],y=daily["fwd_mean"],mode="lines+markers+text",name="Forward Torque Mean",line=dict(color="#E74C3C",width=3),marker=dict(size=10),text=[str(round(v,4)) for v in daily["fwd_mean"]],textposition="top center"),row=2,col=1)
        fig.update_layout(title=f"Mode {safe_int(mode)} — {MODE_NAMES.get(mode)} Torque Trends by Day",height=500)
        fig.update_yaxes(title_text="Return Torque (Nm)",row=1,col=1);fig.update_yaxes(title_text="Forward Torque (Nm)",row=2,col=1)
        figs[mode]=fig
    return figs

def chart_anomaly_scatter(cycles_df,mode,label,chart_key="anom"):
    mode_data=cycles_df[cycles_df["run_mode"]==mode].copy()
    if len(mode_data)<10: return None
    result,n_anom=run_anomaly_detection(mode_data)
    if "anomaly" not in result.columns: return None
    normal=result[result["anomaly"]==1];anomalies=result[result["anomaly"]==-1]
    fig=make_subplots(rows=1,cols=2,subplot_titles=["Return Torque — Anomalies","Cycle Duration — Anomalies"])
    fig.add_trace(go.Scatter(x=normal["start_time"],y=normal["ret_torque_min"],mode="markers",name="Normal",marker=dict(color="#3498DB",size=5,opacity=0.7)),row=1,col=1)
    if len(anomalies)>0:
        fig.add_trace(go.Scatter(x=anomalies["start_time"],y=anomalies["ret_torque_min"],mode="markers",name="Anomaly",marker=dict(color="#E74C3C",size=12,symbol="x",line_width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=normal["start_time"],y=normal["total_duration"],mode="markers",name="Normal",marker=dict(color="#3498DB",size=5,opacity=0.7),showlegend=False),row=1,col=2)
    if len(anomalies)>0:
        fig.add_trace(go.Scatter(x=anomalies["start_time"],y=anomalies["total_duration"],mode="markers",name="Anomaly",marker=dict(color="#E74C3C",size=12,symbol="x",line_width=2),showlegend=False),row=1,col=2)
    fig.update_layout(title=f"Mode {safe_int(mode)} — Anomaly Detection ({n_anom} found) — {label}",height=380)
    fig.update_yaxes(title_text="Return Torque (Nm)",row=1,col=1);fig.update_yaxes(title_text="Duration (rows)",row=1,col=2)
    return fig,n_anom

def chart_sequence_table(result):
    if result is None: return None
    rows=[]
    normal_metrics={"3110":[("torque_mean","fwd_torque_mean"),("torque_max","fwd_torque_max"),("duration","fwd_duration")],"3115":[("duration","settle_duration")],"3125":[("torque_min","ret_torque_min"),("velocity","ret_velocity_min"),("duration","ret_duration")]}
    seq_map={"3110":"Forward Move (3110)","3115":"Forward Settle (3115)","3125":"Return Move (3125)"}
    for seq_key,seq_name in seq_map.items():
        s=result["sequence_analysis"][seq_key]
        alert_metrics={a["metric"] for a in s["alerts"]}
        base_df=result.get("base_cycles_df");new_df=result.get("new_cycles_df")
        for label,col in normal_metrics.get(seq_key,[]):
            if base_df is None or new_df is None or len(base_df)==0: continue
            b=float(base_df[col].mean());c=float(new_df[col].mean())
            if b==0 or np.isnan(b) or np.isnan(c): continue
            pct=((c-b)/abs(b))*100
            rows.append({"Sequence":seq_name,"Metric":label,"Baseline":round(b,4),"Current":round(c,4),"Change %":f"{pct:+.1f}%","Direction":"↑ Higher" if pct>0 else "↓ Lower","Status":"⚠️ STRESS" if label in alert_metrics else "✅ Normal"})
    return pd.DataFrame(rows) if rows else None

def display_mode_result(result,key_prefix="mode"):
    if result is None: st.info("No data for this mode"); return
    status=result["status"];score=result["health_score"]
    icon_map={"CRITICAL":"🔴","WARNING":"🟡","CAUTION":"🟡","NORMAL":"🟢"}
    color_map={"CRITICAL":"red","WARNING":"orange","CAUTION":"orange","NORMAL":"green"}
    st.markdown(f"**{result['cycles']} cycles** | Baseline: {result['baseline_day']} | Baseline torque: {result['baseline_mean']} Nm")
    c1,c2,c3=st.columns(3)
    with c1:
        fig=go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"Health Score"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":color_map[status]},
                   "steps":[{"range":[0,30],"color":"lightcoral"},{"range":[30,70],"color":"lightyellow"},{"range":[70,100],"color":"lightgreen"}]}))
        fig.update_layout(height=220)
        st.plotly_chart(fig,use_container_width=True,key=f"{key_prefix}_gauge")
    with c2:
        st.metric("Status",f"{icon_map[status]} {status}")
        st.metric("Return Torque",f"{result['ret_torque']} Nm",delta=f"{result['ret_torque']-result['alert_threshold']:.4f} to alert")
        st.metric("Days to Alert",result["days_to_alert"] if result["days_to_alert"]<999 else "Safe ✅")
    with c3:
        st.metric("Alert Threshold",f"{result['alert_threshold']} Nm")
        st.metric("Warning Threshold",f"{result['warning_threshold']} Nm")
        st.metric("Trend/Day",f"{result['trend_per_day']} Nm")
    st.markdown("---")
    show_root_cause(result)
    st.markdown("---")
    st.markdown("**Detailed Sequence Table:**")
    tbl=chart_sequence_table(result)
    if tbl is not None:
        st.dataframe(tbl.style.apply(lambda col:["background-color:#ffcccc" if v=="⚠️ STRESS" else "background-color:#ccffcc" if v=="✅ Normal" else "" for v in col],subset=["Status"]),use_container_width=True)
    if status=="CRITICAL": st.error("🔴 STOP MACHINE — Immediate inspection required.")
    elif status=="WARNING": st.warning(f"🟡 Maintenance within {result['days_to_alert']} days")
    elif status=="CAUTION": st.warning("🟡 Monitor daily — early signs of load increase")
    else: st.success("🟢 Machine healthy — all parameters normal")

# ── MAIN ──
st.title("🔧 Shuffler Motor Health Monitor")
st.markdown("### Predictive Maintenance — Blackbox Factories")
st.markdown("---")

historical=load_all_data();baselines=load_baselines()
if historical is None: st.error("❌ Historical data not found."); st.stop()

auto_results,latest_day=auto_analyse_latest(historical,baselines)

st.sidebar.header("📁 Load New Data")
day_label=st.sidebar.text_input("Date Label",placeholder="e.g. May 01")
input_method=st.sidebar.radio("Data Source",["📤 Upload CSV Files","📂 Enter Folder Path"])
df_raw=None;invalid_files=[];valid_files=[]

if input_method=="📤 Upload CSV Files":
    uploaded=st.sidebar.file_uploader("Select all CSV files",type=["csv","CSV"],accept_multiple_files=True)
    if uploaded and day_label:
        with st.spinner("Validating files..."):
            df_raw,invalid_files,valid_files=load_uploaded(uploaded,day_label)
        if df_raw is not None: st.sidebar.success(f"✅ {len(valid_files)} valid files loaded")
        if invalid_files: st.sidebar.error(f"❌ {len(invalid_files)} files failed")
else:
    folder=st.sidebar.text_input("Folder Path",placeholder="/path/to/csv/folder")
    if folder and day_label:
        if os.path.exists(folder):
            with st.spinner("Loading folder..."):
                df_raw,invalid_files,valid_files=load_folder(folder,day_label)
            if df_raw is not None: st.sidebar.success(f"✅ {len(valid_files)} files loaded")
            if invalid_files: st.sidebar.error(f"❌ {len(invalid_files)} failed")
        else: st.sidebar.error("❌ Folder not found")

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 Historical Overview","🔬 Data Preview","🔍 Health Analysis","🔄 Cycle View","🚨 Anomaly Detection","📄 Full Report"])

with tab1:
    st.markdown("## Historical Overview by Mode")
    st.markdown(f"### ⚡ Current Status — Latest Data: **{latest_day}**")
    if auto_results:
        banner_cols=st.columns(len(auto_results))
        for col,(mode,result) in zip(banner_cols,auto_results.items()):
            status=result["status"]
            icon={"CRITICAL":"🔴","WARNING":"🟡","CAUTION":"🟡","NORMAL":"🟢"}.get(status,"⚪")
            with col:
                st.markdown(f"**Mode {safe_int(mode)} — {result['mode_name']}**\n\n{icon} **{status}**\n\nHealth: **{result['health_score']}/100**\n\nRet Torque: `{result['ret_torque']} Nm`\n\nDays to alert: **{result['days_to_alert'] if result['days_to_alert']<999 else 'Safe ✅'}**")
    st.markdown("---")
    summary_rows=[]
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)==0: continue
        b_info=baselines.get(mode)
        if b_info is None: continue
        b_mean=b_info["baseline_mean"];alert=b_info["alert_threshold"];warn=b_info["warning_threshold"]
        for day in mode_data["day"].unique():
            dd=mode_data[mode_data["day"]==day];ret=dd["ret_torque_min"].mean()
            h=max(0,min(100,(ret-alert)/(b_mean-alert)*100))
            summary_rows.append({"Mode":f"Mode {safe_int(mode)} — {MODE_NAMES.get(mode)}","Day":day,"Cycles":len(dd),"Ret Torque (Nm)":round(ret,4),"Health Score":round(h,1),"Status":"🔴 CRITICAL" if ret<alert else "🟡 WARNING" if ret<warn else "🟡 CAUTION" if h<80 else "🟢 NORMAL"})
    if summary_rows:
        st.markdown("### Full History Summary")
        st.dataframe(pd.DataFrame(summary_rows),use_container_width=True)
    st.markdown("---")
    st.markdown("### 📈 Multi-Day Return Torque Trends")
    for mode,fig in chart_multiday_trend(historical,baselines).items():
        st.plotly_chart(fig,use_container_width=True,key=f"t1_trend_{safe_int(mode)}")
    st.markdown("---")
    st.markdown("### 📈 Return & Forward Torque Trends by Day")
    for mode,fig in chart_line_by_day(historical,baselines).items():
        st.plotly_chart(fig,use_container_width=True,key=f"t1_line_{safe_int(mode)}")

with tab2:
    st.markdown("## Data Preview & Validation")
    if invalid_files:
        st.error(f"{len(invalid_files)} file(s) failed validation.")
        for inv in invalid_files:
            with st.expander(f"❌ {inv['filename']}"):
                for issue in inv["issues"]: st.markdown(f"- {issue}")
        st.code("Line 0: [LOGGING],RCPU_1,...\nLine 1: MergedFile\nLine 3: TIME (UTC+09:00),INTERVAL[us],...,D3203,...,D3223,D3224,...,D3238\nLine 4: (empty)\nLine 5+: data")
    if df_raw is None:
        if not invalid_files: st.info("👈 Load data from sidebar first")
    else:
        total=len(df_raw);active=df_raw[df_raw["sequence"].isin([3110.0,3115.0,3120.0,3125.0,3130.0])]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total Rows",f"{total:,}");c2.metric("Active Rows",f"{len(active):,}",f"{len(active)/total*100:.1f}%")
        c3.metric("Start",str(df_raw["timestamp"].min()).split(".")[0]);c4.metric("End",str(df_raw["timestamp"].max()).split(".")[0])
        st.markdown("### Mode Breakdown")
        mc1,mc2,mc3=st.columns(3)
        for col,(mode,mname) in zip([mc1,mc2,mc3],[(0.0,"Normal"),(2.0,"Maintenance"),(1.0,"Endurance")]):
            count=int(active[active["run_mode"]==mode].shape[0]) if len(active)>0 else 0
            col.metric(f"Mode {safe_int(mode)} — {mname}",f"{count:,} rows",f"{count/len(active)*100:.1f}%" if len(active)>0 else "0%")
        st.markdown("### Sequence Distribution")
        seq_counts=df_raw["sequence"].value_counts().sort_index()
        seq_df=pd.DataFrame({"Sequence":[f"{int(k)} — {SEQ_NAMES.get(int(k),'Unknown')}" for k in seq_counts.index],"Count":seq_counts.values})
        fig=px.bar(seq_df,x="Sequence",y="Count",title="Rows per Sequence",color="Count",color_continuous_scale="Blues")
        fig.update_layout(height=300);st.plotly_chart(fig,use_container_width=True,key="t2_seqdist")
        st.markdown("### Key Sequences Check")
        k1,k2,k3,k4=st.columns(4)
        for col,seq,name in zip([k1,k2,k3,k4],[3110.0,3115.0,3125.0,3130.0],["Forward","Settle","Return","End"]):
            found=seq in df_raw["sequence"].values
            col.metric(f"Seq {int(seq)}\n{name}","✅ Found" if found else "❌ Missing")
        can_extract=all(s in df_raw["sequence"].values for s in [3115.0,3125.0,3130.0])
        if can_extract: st.success("✅ Data looks good — proceed to Health Analysis")
        else: st.warning("⚠️ Missing key sequences — these files may be idle-only data. Upload all files from the day.")

with tab3:
    st.markdown("## Health Analysis by Mode")
    st.markdown(f"### ⚡ Current Machine Status — Latest Data: **{latest_day}**")
    st.info("ℹ️ Auto-generated from most recent history. Upload new data and click Run to update.")
    if auto_results:
        for mode,result in auto_results.items():
            st.markdown(f"#### Mode {safe_int(mode)} — {result['mode_name']}")
            display_mode_result(result,key_prefix=f"auto_{safe_int(mode)}")
            st.markdown("---")
    if df_raw is not None and day_label:
        st.markdown("---")
        st.markdown("### 🆕 New Data Analysis")
        if st.button("🔍 Run Health Analysis on New Data",type="primary",use_container_width=True):
            with st.spinner("Extracting cycles..."):
                all_new=extract_cycles(df_raw,day_label)
            if len(all_new)==0: st.error("❌ No cycles found. Check Data Preview tab."); st.stop()
            st.success(f"✅ {len(all_new)} cycles extracted")
            mode_breakdown=all_new.groupby("run_mode").size()
            cols=st.columns(3)
            for i,(mode,mname) in enumerate([(0.0,"Normal"),(2.0,"Maintenance"),(1.0,"Endurance")]):
                cols[i].metric(f"Mode {safe_int(mode)} — {mname}",f"{int(mode_breakdown.get(mode,0))} cycles")
            st.markdown("---");st.markdown("### 🔎 Validation Report")
            issues,warnings=check_duplicates(all_new,historical,day_label)
            for w in warnings: st.warning(f"⚠️ **{w['type']}:** {w['message']}")
            has_issues=len(issues)>0
            if has_issues:
                st.error("🚨 Conflicts detected")
                for issue in issues:
                    sev="🔴" if issue["severity"]=="HIGH" else "🟡"
                    with st.expander(f"{sev} {issue['type']}",expanded=True):
                        st.markdown(f"**Issue:** {issue['message']}")
                        st.markdown(f"**Action:** {issue['action']}")
                        if "existing_cycles" in issue:
                            c1,c2=st.columns(2);c1.metric("Existing",issue["existing_cycles"]);c2.metric("New",issue["new_cycles"])
                        if "n_dupes" in issue:
                            c1,c2=st.columns(2);c1.metric("Duplicates",issue["n_dupes"]);c2.metric("Rate",f"{issue['dupe_pct']}%")
            else: st.success("✅ No conflicts — safe to save.")
            st.markdown("---");st.markdown("### 📊 New Data Results")
            mode_results={}
            for mode in sorted([m for m in all_new["run_mode"].unique() if not pd.isna(m)]):
                new_mode=all_new[all_new["run_mode"]==mode].copy()
                if len(new_mode)==0: continue
                st.markdown(f"#### Mode {safe_int(mode)} — {MODE_NAMES.get(mode,'Unknown')}")
                b_info=baselines.get(mode)
                if b_info is None:
                    b_mean=float(new_mode["ret_torque_min"].mean());b_std=float(new_mode["ret_torque_min"].std())
                    if np.isnan(b_std) or b_std==0: b_std=0.003
                    b_info={"baseline_day":day_label,"baseline_mean":round(b_mean,4),"baseline_std":round(b_std,4),"alert_threshold":round(b_mean-3*b_std,4),"warning_threshold":round(b_mean-2*b_std,4)}
                    st.info(f"ℹ️ First time Mode {safe_int(mode)} — using today as baseline.")
                hist_mode=historical[historical["run_mode"]==mode].copy()
                result=analyse_one_mode(new_mode,hist_mode,b_info,day_label,mode)
                if result:
                    mode_results[mode]={"result":result,"b_info":b_info}
                    display_mode_result(result,key_prefix=f"new_{safe_int(mode)}")
                st.markdown("---")
            st.markdown("### 💾 Save to History?")
            if has_issues: st.warning("⚠️ Conflicts above — review carefully.")
            col_yes,col_no=st.columns(2)
            with col_yes:
                btn="⚠️ Save Anyway" if has_issues else "✅ Save to History"
                if st.button(btn,type="primary",use_container_width=True):
                    updated=pd.concat([historical[historical["day"]!=day_label],all_new],ignore_index=True)
                    updated.to_csv("all_cycles_4days.csv",index=False)
                    for mode,data in mode_results.items():
                        if mode not in baselines: baselines[mode]=data["b_info"]
                    with open("mode_baselines.json","w") as f:
                        json.dump({str(k):v for k,v in baselines.items()},f,indent=2)
                    st.cache_data.clear();st.success(f"✅ {day_label} saved!");st.balloons()
            with col_no:
                if st.button("❌ Discard",use_container_width=True):
                    st.info("Data NOT saved. Historical data unchanged.")

with tab4:
    st.markdown("## Cycle View")
    if df_raw is None:
        st.info("👈 Load data from sidebar first")
    else:
        st.markdown("### 📡 Raw Signal Overview")
        with st.spinner("Building signal chart..."):
            raw_fig=chart_raw_signal(df_raw,day_label or "Loaded Data")
        if raw_fig: st.plotly_chart(raw_fig,use_container_width=True,key="t4_rawsig")
        else: st.warning("No active sequence data found. These files may be idle-only.")
        st.markdown("---")
        st.markdown("### 🔬 Single Cycle Breakdown")
        active_raw=df_raw[df_raw["sequence"].isin([3110.0,3115.0,3120.0,3125.0,3130.0])]
        if len(active_raw)==0:
            st.warning("No active machine cycles in uploaded files. These files contain only idle data (sequence 3100). Upload files from when the machine was running.")
        else:
            seq_vals=active_raw["sequence"].values
            cycle_starts=[0]+[i for i in range(1,len(seq_vals)) if seq_vals[i]==3110.0 and seq_vals[i-1]!=3110.0]
            n_cycles=max(1,len(cycle_starts))
            cycle_num=st.slider("Select Cycle",min_value=0,max_value=n_cycles-1,value=0,help=f"{n_cycles} cycle starts detected")
            with st.spinner("Building cycle chart..."):
                cycle_fig=chart_one_cycle(df_raw,cycle_num)
            if cycle_fig:
                st.plotly_chart(cycle_fig,use_container_width=True,key="t4_cycle")
                st.markdown("**Color Legend:**")
                l1,l2,l3,l4,l5=st.columns(5)
                l1.markdown("🔴 **3110** Forward Move");l2.markdown("🟠 **3115** Forward Settle")
                l3.markdown("🟡 **3120** Forward Hold");l4.markdown("🔵 **3125** Return Move")
                l5.markdown("🟢 **3130** Return Settle")

with tab5:
    st.markdown("## Anomaly Detection")
    st.markdown(f"### ⚡ Anomalies in Latest Data — **{latest_day}**")
    latest_data=historical[historical["day"]==latest_day]
    if len(latest_data)>=10:
        for mode in sorted([m for m in latest_data["run_mode"].unique() if not pd.isna(m)]):
            mode_name=MODE_NAMES.get(mode,"Unknown")
            st.markdown(f"#### Mode {safe_int(mode)} — {mode_name}")
            out=chart_anomaly_scatter(latest_data,mode,latest_day,chart_key=f"t5_latest_{safe_int(mode)}")
            if out:
                fig,n_anom=out
                st.plotly_chart(fig,use_container_width=True,key=f"t5_latest_fig_{safe_int(mode)}")
                if n_anom>0: st.warning(f"⚠️ {n_anom} anomalous cycles")
                else: st.success("✅ No anomalies detected")
    else: st.info("Not enough cycles in latest day for anomaly detection.")
    st.markdown("---")
    if df_raw is not None and day_label:
        st.markdown("### 🆕 Anomalies in New Uploaded Data")
        if st.button("🚨 Detect Anomalies in New Data",use_container_width=True):
            with st.spinner("Extracting and detecting..."):
                new_cycles=extract_cycles(df_raw,day_label)
            if len(new_cycles)==0: st.error("❌ No cycles found.")
            else:
                for mode in sorted([m for m in new_cycles["run_mode"].unique() if not pd.isna(m)]):
                    st.markdown(f"#### Mode {safe_int(mode)} — {MODE_NAMES.get(mode,'Unknown')}")
                    out=chart_anomaly_scatter(new_cycles,mode,day_label,chart_key=f"t5_new_{safe_int(mode)}")
                    if out:
                        fig,n_anom=out
                        st.plotly_chart(fig,use_container_width=True,key=f"t5_new_fig_{safe_int(mode)}")
                        if n_anom>0: st.warning(f"⚠️ {n_anom} anomalies detected")
                        else: st.success("✅ No anomalies detected")
    st.markdown("---")
    st.markdown("### 📚 Full Historical Anomaly Overview")
    if st.button("Run on All Historical Data",use_container_width=True):
        for mode in [0.0,2.0,1.0]:
            mode_name=MODE_NAMES.get(mode,"Unknown")
            mode_data=historical[historical["run_mode"]==mode]
            if len(mode_data)<10: continue
            st.markdown(f"### Mode {safe_int(mode)} — {mode_name}")
            anom_counts={}
            for day in sorted(mode_data["day"].unique()):
                day_data=mode_data[mode_data["day"]==day]
                if len(day_data)<5: continue
                _,n=run_anomaly_detection(day_data);anom_counts[day]=n
            if anom_counts:
                anom_df=pd.DataFrame({"Day":list(anom_counts.keys()),"Anomalies":list(anom_counts.values())})
                fig=px.bar(anom_df,x="Day",y="Anomalies",title=f"Mode {safe_int(mode)} — Anomaly Count per Day",color="Anomalies",color_continuous_scale="RdYlGn_r",text="Anomalies")
                fig.update_layout(height=300);st.plotly_chart(fig,use_container_width=True,key=f"t5_hist_bar_{safe_int(mode)}")
            out=chart_anomaly_scatter(mode_data,mode,"All Days",chart_key=f"t5_hist_{safe_int(mode)}")
            if out:
                fig,n_anom=out
                fig.update_layout(title=f"Mode {safe_int(mode)} — All Historical ({n_anom} anomalies)")
                st.plotly_chart(fig,use_container_width=True,key=f"t5_hist_scatter_{safe_int(mode)}")
            st.markdown("---")

with tab6:
    st.markdown("## Full Detailed Analysis Report")
    st.markdown("Complete report with all observations, metrics, charts and download options.")
    st.markdown("---")
    st.markdown("### 1. Historical Data Summary")
    summary_rows=[]
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)==0: continue
        b_info=baselines.get(mode)
        if b_info is None: continue
        b_mean=b_info["baseline_mean"];alert=b_info["alert_threshold"];warn=b_info["warning_threshold"]
        for day in sorted(mode_data["day"].unique()):
            dd=mode_data[mode_data["day"]==day];ret=dd["ret_torque_min"].mean();fwd=dd["fwd_torque_max"].mean()
            h=max(0,min(100,(ret-alert)/(b_mean-alert)*100))
            summary_rows.append({"Mode":f"Mode {safe_int(mode)} — {MODE_NAMES.get(mode)}","Day":day,"Cycles":len(dd),"Ret Torque Mean":round(ret,4),"Fwd Torque Mean":round(fwd,4),"Health Score":round(h,1),"Alert Threshold":round(alert,4),"Warning Threshold":round(warn,4),"Baseline":round(b_mean,4),"Gap to Alert":round(ret-alert,4),"Status":"CRITICAL" if ret<alert else "WARNING" if ret<warn else "CAUTION" if h<80 else "NORMAL"})
    if summary_rows:
        df_summary=pd.DataFrame(summary_rows)
        st.dataframe(df_summary,use_container_width=True)
        st.download_button("⬇️ Download Summary CSV",df_summary.to_csv(index=False).encode(),"shuffler_summary.csv","text/csv")
    st.markdown("---")
    st.markdown("### 2. Trend Analysis")
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)<5: continue
        b_info=baselines.get(mode)
        if b_info is None: continue
        st.markdown(f"#### Mode {safe_int(mode)} — {MODE_NAMES.get(mode)}")
        daily=mode_data.groupby("day").agg(cycles=("cycle_id","count"),ret_mean=("ret_torque_min","mean"),ret_std=("ret_torque_min","std"),ret_min=("ret_torque_min","min"),ret_max=("ret_torque_min","max"),fwd_mean=("fwd_torque_max","mean"),settle_mean=("settle_duration","mean"),total_mean=("total_duration","mean")).round(4).reset_index()
        st.dataframe(daily,use_container_width=True)
        if len(daily)>=2:
            z=np.polyfit(range(len(daily)),daily["ret_mean"].values,1);slope=z[0]
            alert=b_info["alert_threshold"]
            days_to=int((alert-daily["ret_mean"].iloc[-1])/slope) if slope<0 else 999
            c1,c2,c3=st.columns(3)
            c1.metric("Trend per Day",f"{slope:.6f} Nm")
            c2.metric("Current vs Baseline",f"{daily['ret_mean'].iloc[-1]:.4f} Nm",delta=f"{daily['ret_mean'].iloc[-1]-b_info['baseline_mean']:.4f}")
            c3.metric("Days to Alert",days_to if days_to<999 else "Safe")
    st.markdown("---")
    st.markdown("### 3. Sequence Level Analysis — Latest Day")
    st.markdown(f"Based on: **{latest_day}**")
    latest=historical[historical["day"]==latest_day]
    for mode in sorted([m for m in latest["run_mode"].unique() if not pd.isna(m)]):
        mode_name=MODE_NAMES.get(mode,"Unknown")
        new_mode=latest[latest["run_mode"]==mode].copy()
        if len(new_mode)==0: continue
        b_info=baselines.get(mode)
        if b_info is None: continue
        hist_mode=historical[(historical["run_mode"]==mode)&(historical["day"]!=latest_day)].copy()
        result=analyse_one_mode(new_mode,hist_mode,b_info,latest_day,mode)
        if result is None: continue
        st.markdown(f"#### Mode {safe_int(mode)} — {mode_name}")
        tbl=chart_sequence_table(result)
        if tbl is not None:
            st.dataframe(tbl.style.apply(lambda col:["background-color:#ffcccc" if v=="⚠️ STRESS" else "background-color:#ccffcc" if v=="✅ Normal" else "" for v in col],subset=["Status"]),use_container_width=True)
    st.markdown("---")
    st.markdown("### 4. Anomaly Report")
    anom_report=[]
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)<10: continue
        for day in sorted(mode_data["day"].unique()):
            day_data=mode_data[mode_data["day"]==day]
            if len(day_data)<5: continue
            _,n=run_anomaly_detection(day_data);pct=n/len(day_data)*100
            anom_report.append({"Mode":f"Mode {safe_int(mode)} — {MODE_NAMES.get(mode)}","Day":day,"Total Cycles":len(day_data),"Anomalies":n,"Anomaly Rate (%)":round(pct,1),"Risk":"HIGH" if pct>10 else "MEDIUM" if pct>5 else "LOW"})
    if anom_report:
        df_anom=pd.DataFrame(anom_report);st.dataframe(df_anom,use_container_width=True)
        st.download_button("⬇️ Download Anomaly CSV",df_anom.to_csv(index=False).encode(),"shuffler_anomalies.csv","text/csv")
        fig=px.bar(df_anom,x="Day",y="Anomaly Rate (%)",color="Mode",barmode="group",title="Anomaly Rate per Day per Mode")
        fig.update_layout(height=350);st.plotly_chart(fig,use_container_width=True,key="t6_anom_bar")
    st.markdown("---")
    st.markdown("### 5. Full Cycles Data")
    rc1,rc2=st.columns(2)
    with rc1: filter_day=st.selectbox("Filter by Day",["All"]+sorted(historical["day"].unique().tolist()))
    with rc2: filter_mode=st.selectbox("Filter by Mode",["All","Normal (0)","Maintenance (2)","Endurance (1)"])
    df_filtered=historical.copy()
    if filter_day!="All": df_filtered=df_filtered[df_filtered["day"]==filter_day]
    if filter_mode=="Normal (0)": df_filtered=df_filtered[df_filtered["run_mode"]==0.0]
    elif filter_mode=="Maintenance (2)": df_filtered=df_filtered[df_filtered["run_mode"]==2.0]
    elif filter_mode=="Endurance (1)": df_filtered=df_filtered[df_filtered["run_mode"]==1.0]
    show_cols=["cycle_id","day","start_time","run_mode","fwd_torque_max","fwd_torque_mean","fwd_duration","settle_duration","ret_torque_min","ret_torque_mean","ret_velocity_min","ret_duration","total_duration"]
    st.markdown(f"Showing **{len(df_filtered):,}** cycles")
    st.dataframe(df_filtered[show_cols].round(4),use_container_width=True)
    st.download_button("⬇️ Download Full Cycles CSV",df_filtered[show_cols].to_csv(index=False).encode(),f"shuffler_cycles.csv","text/csv")
    st.markdown("---")
    st.markdown("### 6. All Charts")
    st.markdown("#### Return Torque Trends")
    for mode,fig in chart_multiday_trend(historical,baselines).items():
        st.plotly_chart(fig,use_container_width=True,key=f"t6_trend_{safe_int(mode)}")
    st.markdown("#### Torque Line Charts by Day")
    for mode,fig in chart_line_by_day(historical,baselines).items():
        st.plotly_chart(fig,use_container_width=True,key=f"t6_line_{safe_int(mode)}")
    st.markdown("#### Anomaly Scatter Charts")
    for mode in [0.0,2.0,1.0]:
        mode_data=historical[historical["run_mode"]==mode]
        if len(mode_data)<10: continue
        out=chart_anomaly_scatter(mode_data,mode,"All Historical Data",chart_key=f"t6_anom_{safe_int(mode)}")
        if out:
            fig,n_anom=out
            st.plotly_chart(fig,use_container_width=True,key=f"t6_anom_scatter_{safe_int(mode)}")

st.markdown("---")
st.caption(f"🔧 Shuffler Motor Predictive Maintenance | Latest: {latest_day} | Historical: {len(historical):,} cycles | 6 tabs")
