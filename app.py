
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Data Quality Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Based Database Data-Quality Analyzer")
st.caption("Oracle SQL + PL/SQL + Machine Learning")

@st.cache_data
def load_data():
    students = pd.read_csv("students.csv")
    marks = pd.read_csv("marks.csv")
    attendance = pd.read_csv("attendance.csv")
    quality_report = pd.read_csv("quality_report.csv")

    for df in [students, marks, attendance, quality_report]:
        df.columns = df.columns.str.strip().str.upper()

    students["DUPLICATE_COUNT"] = students.groupby(
        ["NAME", "EMAIL", "PHONE"], dropna=False
    )["STUDENT_ID"].transform("count")

    students["IS_DUPLICATE"] = (
        students["DUPLICATE_COUNT"] > 1
    ).astype(int)

    students["MISSING_EMAIL"] = (
        students["EMAIL"].isna()
    ).astype(int)

    marks["INVALID_MARK"] = (
        (marks["MARKS"] < 0) | (marks["MARKS"] > 100)
    ).astype(int)

    marks_summary = marks.groupby("STUDENT_ID").agg(
        AVG_MARKS=("MARKS", "mean"),
        INVALID_MARK_COUNT=("INVALID_MARK", "sum")
    ).reset_index()

    attendance["INVALID_ATTENDANCE"] = (
        (attendance["ATTENDED_CLASSES"] < 0) |
        (attendance["ATTENDED_CLASSES"] > attendance["TOTAL_CLASSES"])
    ).astype(int)

    attendance_summary = attendance.groupby("STUDENT_ID").agg(
        TOTAL_CLASSES=("TOTAL_CLASSES", "sum"),
        ATTENDED_CLASSES=("ATTENDED_CLASSES", "sum"),
        INVALID_ATTENDANCE_COUNT=("INVALID_ATTENDANCE", "sum")
    ).reset_index()

    data = students.merge(
        marks_summary, on="STUDENT_ID", how="left"
    )

    data = data.merge(
        attendance_summary, on="STUDENT_ID", how="left"
    )

    data["ATTENDANCE_RATIO"] = (
        data["ATTENDED_CLASSES"] /
        data["TOTAL_CLASSES"].replace(0, np.nan)
    )

    features = [
        "MISSING_EMAIL",
        "IS_DUPLICATE",
        "INVALID_MARK_COUNT",
        "INVALID_ATTENDANCE_COUNT",
        "AVG_MARKS",
        "ATTENDANCE_RATIO"
    ]

    X = data[features].replace(
        [np.inf, -np.inf], np.nan
    )

    X = X.fillna(X.median()).fillna(0)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,
        random_state=42
    )

    model.fit(X)

    data["ANOMALY_PREDICTION"] = model.predict(X)
    data["ANOMALY_SCORE"] = model.decision_function(X)

    data["ML_RESULT"] = data["ANOMALY_PREDICTION"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    def classify_record(row):
        issues = []

        if row["MISSING_EMAIL"] == 1:
            issues.append("Missing Email")

        if row["IS_DUPLICATE"] == 1:
            issues.append("Duplicate Record")

        if row["INVALID_MARK_COUNT"] > 0:
            issues.append("Invalid Marks")

        if row["INVALID_ATTENDANCE_COUNT"] > 0:
            issues.append("Invalid Attendance")

        if row["ML_RESULT"] == "Anomaly":
            issues.append("ML Anomaly")

        if (
            row["INVALID_MARK_COUNT"] > 0
            or row["INVALID_ATTENDANCE_COUNT"] > 0
        ):
            status = "Critical"

        elif (
            row["MISSING_EMAIL"] == 1
            or row["IS_DUPLICATE"] == 1
        ):
            status = "Warning"

        elif row["ML_RESULT"] == "Anomaly":
            status = "ML Anomaly"

        else:
            status = "Normal"

        return pd.Series({
            "SUSPICIOUS_STATUS": status,
            "DETECTED_ISSUES": (
                ", ".join(issues)
                if issues else "No issues detected"
            )
        })

    classification = data.apply(classify_record, axis=1)
    data = pd.concat([data, classification], axis=1)

    return data, quality_report


data, quality_report = load_data()

st.subheader("📊 Dashboard Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", len(data))
c2.metric("Quality Issues", len(quality_report))
c3.metric(
    "ML Anomalies",
    (data["ML_RESULT"] == "Anomaly").sum()
)
c4.metric(
    "Critical Records",
    (data["SUSPICIOUS_STATUS"] == "Critical").sum()
)

st.divider()

st.subheader("🚨 Suspicious Record Classification")

counts = data["SUSPICIOUS_STATUS"].value_counts()

st.bar_chart(counts)

search = st.text_input("Search by student name or ID")

statuses = ["All"] + sorted(
    data["SUSPICIOUS_STATUS"].unique().tolist()
)

selected_status = st.selectbox(
    "Filter by status",
    statuses
)

filtered = data.copy()

if selected_status != "All":
    filtered = filtered[
        filtered["SUSPICIOUS_STATUS"] == selected_status
    ]

if search:
    filtered = filtered[
        filtered["NAME"].astype(str).str.contains(
            search, case=False, na=False
        ) |
        filtered["STUDENT_ID"].astype(str).str.contains(
            search, na=False
        )
    ]

st.dataframe(
    filtered[
        [
            "STUDENT_ID",
            "NAME",
            "SUSPICIOUS_STATUS",
            "DETECTED_ISSUES",
            "ANOMALY_SCORE"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ Download Classification Report",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="suspicious_records_report.csv",
    mime="text/csv"
)

st.divider()

st.subheader("📋 Original Oracle Quality Report")

st.dataframe(
    quality_report,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ Download Oracle Quality Report",
    data=quality_report.to_csv(index=False).encode("utf-8"),
    file_name="oracle_quality_report.csv",
    mime="text/csv"
)

st.caption(
    "ML anomaly flags indicate unusual records for review; "
    "they do not automatically prove a data error. "
    "Data is loaded from CSV exports of Oracle."
)
