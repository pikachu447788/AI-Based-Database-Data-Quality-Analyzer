
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Database Data Quality Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Based Database Data-Quality Analyzer")
st.caption(
    "Oracle Database + Python Machine Learning + Isolation Forest"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Section",
    [
        "Dashboard",
        "Student Analysis",
        "ML Anomaly Detection",
        "Oracle Quality Report",
        "Download Reports",
        "About Project"
    ]
)


# --------------------------------------------------
# LOAD PROJECT DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    students = pd.read_csv("students.csv")
    marks = pd.read_csv("marks.csv")
    attendance = pd.read_csv("attendance.csv")
    oracle_report = pd.read_csv("quality_report.csv")

    analysis = pd.read_csv("final_analysis.csv")
    summary = pd.read_csv("analysis_summary.csv")
    ml_report = pd.read_csv("ml_anomaly_report.csv")

    return (
        students,
        marks,
        attendance,
        oracle_report,
        analysis,
        summary,
        ml_report
    )


try:
    (
        students,
        marks,
        attendance,
        oracle_report,
        analysis,
        summary,
        ml_report
    ) = load_data()

except Exception as e:
    st.error(f"Unable to load project CSV files: {e}")
    st.info(
        "Check that all seven CSV files are present "
        "in the same GitHub folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

analysis.columns = analysis.columns.str.upper()
summary.columns = summary.columns.str.upper()
ml_report.columns = ml_report.columns.str.upper()
oracle_report.columns = oracle_report.columns.str.upper()

analysis["STUDENT_ID"] = pd.to_numeric(
    analysis["STUDENT_ID"], errors="coerce"
)

analysis["ANOMALY_SCORE"] = pd.to_numeric(
    analysis["ANOMALY_SCORE"], errors="coerce"
)

analysis["SUSPICIOUS_STATUS"] = (
    analysis["SUSPICIOUS_STATUS"].fillna("Unknown")
)

analysis["DETECTED_ISSUES"] = (
    analysis["DETECTED_ISSUES"].fillna("No issues detected")
)

ml_report["ML_RESULT"] = ml_report["ML_RESULT"].fillna("Unknown")

total_students = len(analysis)

missing_emails = int(
    pd.to_numeric(analysis["MISSING_EMAIL"], errors="coerce")
    .fillna(0).sum()
)

duplicate_records = int(
    pd.to_numeric(analysis["IS_DUPLICATE"], errors="coerce")
    .fillna(0).sum()
)

invalid_marks = int(
    pd.to_numeric(analysis["INVALID_MARK_COUNT"], errors="coerce")
    .fillna(0).sum()
)

invalid_attendance = int(
    pd.to_numeric(
        analysis["INVALID_ATTENDANCE_COUNT"], errors="coerce"
    ).fillna(0).sum()
)

ml_anomalies = int(
    (ml_report["ML_RESULT"] == "Anomaly").sum()
)

critical_records = int(
    (analysis["SUSPICIOUS_STATUS"] == "Critical").sum()
)

warning_records = int(
    (analysis["SUSPICIOUS_STATUS"] == "Warning").sum()
)

normal_records = int(
    (analysis["SUSPICIOUS_STATUS"] == "Normal").sum()
)

oracle_issues = len(oracle_report)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
def create_pdf():

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        "AI-Based Database Data-Quality Analyzer",
        styles["Title"]
    ))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "Complete Project Analysis Report",
        styles["Heading2"]
    ))

    elements.append(Paragraph(
        "Generated on: " + datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        ),
        styles["Normal"]
    ))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph(
        "1. Analysis Summary",
        styles["Heading2"]
    ))

    summary_data = [
        ["Metric", "Value"],
        ["Total Students", total_students],
        ["Oracle Quality Issues", oracle_issues],
        ["Missing Emails", missing_emails],
        ["Duplicate Records", duplicate_records],
        ["Invalid Marks", invalid_marks],
        ["Invalid Attendance", invalid_attendance],
        ["ML Anomalies", ml_anomalies],
        ["Critical Records", critical_records],
        ["Warning Records", warning_records],
        ["Normal Records", normal_records]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[250, 120]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EAF0F8")])
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph(
        "2. Student-Wise Analysis",
        styles["Heading2"]
    ))

    student_data = [[
        "ID", "Name", "Status", "Detected Issues", "ML Score"
    ]]

    for _, row in analysis.iterrows():

        student_data.append([
            str(row["STUDENT_ID"]),
            str(row["NAME"]),
            str(row["SUSPICIOUS_STATUS"]),
            str(row["DETECTED_ISSUES"]),
            str(round(row["ANOMALY_SCORE"], 5))
        ])

    student_table = Table(
        student_data,
        colWidths=[45, 85, 85, 330, 75],
        repeatRows=1
    )

    student_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EAF0F8")])
    ]))

    elements.append(student_table)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph(
        "3. Oracle Quality Report",
        styles["Heading2"]
    ))

    oracle_data = [[
        "Issue ID", "Student ID", "Issue Type", "Description"
    ]]

    for _, row in oracle_report.iterrows():

        oracle_data.append([
            str(row.get("ISSUE_ID", "")),
            str(row.get("STUDENT_ID", "")),
            str(row.get("ISSUE_TYPE", "")),
            str(row.get("DESCRIPTION", ""))
        ])

    oracle_table = Table(
        oracle_data,
        colWidths=[65, 75, 140, 350],
        repeatRows=1
    )

    oracle_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    elements.append(oracle_table)

    elements.append(Spacer(1, 18))

    elements.append(Paragraph(
        "4. Machine Learning Methodology",
        styles["Heading2"]
    ))

    elements.append(Paragraph(
        "Isolation Forest was used for unsupervised anomaly "
        "detection. SQL and PL/SQL rules detect predefined "
        "data-quality issues, while the ML model identifies "
        "unusual combinations of student features. ML anomalies "
        "should be investigated and are not automatically proof "
        "of incorrect records.",
        styles["Normal"]
    ))

    doc.build(elements)

    buffer.seek(0)

    return buffer.getvalue()
if page == "Dashboard":

    st.header("Project Dashboard")

    st.write(
        "Overview of database quality checks and "
        "machine-learning anomaly detection."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Oracle Issues", oracle_issues)
    col3.metric("ML Anomalies", ml_anomalies)
    col4.metric("Critical Records", critical_records)

    st.divider()

    st.subheader("Data Quality Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Missing Emails", missing_emails)
    col2.metric("Duplicate Records", duplicate_records)
    col3.metric("Invalid Marks", invalid_marks)
    col4.metric("Invalid Attendance", invalid_attendance)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Student Classification")

        status_counts = analysis[
            "SUSPICIOUS_STATUS"
        ].value_counts()

        st.bar_chart(status_counts)

    with right:
        st.subheader("Data Quality Issues")

        issue_data = pd.Series({
            "Missing Emails": missing_emails,
            "Duplicate Records": duplicate_records,
            "Invalid Marks": invalid_marks,
            "Invalid Attendance": invalid_attendance
        })

        st.bar_chart(issue_data)

    st.divider()

    st.subheader("Student Records Preview")

    st.dataframe(
        analysis[
            [
                "STUDENT_ID",
                "NAME",
                "SUSPICIOUS_STATUS",
                "DETECTED_ISSUES"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# STUDENT ANALYSIS
# --------------------------------------------------

elif page == "Student Analysis":

    st.header("Student-Wise Data Quality Analysis")

    search = st.text_input(
        "Search by Student ID, Name or Issue"
    )

    statuses = sorted(
        analysis["SUSPICIOUS_STATUS"].unique().tolist()
    )

    selected_status = st.multiselect(
        "Filter by Status",
        statuses,
        default=statuses
    )

    filtered = analysis[
        analysis["SUSPICIOUS_STATUS"].isin(selected_status)
    ].copy()

    if search:
        search_text = search.strip().lower()

        mask = (
            filtered["STUDENT_ID"].astype(str).str.lower()
            .str.contains(search_text, na=False)
            |
            filtered["NAME"].astype(str).str.lower()
            .str.contains(search_text, na=False)
            |
            filtered["DETECTED_ISSUES"].astype(str).str.lower()
            .str.contains(search_text, na=False)
        )

        filtered = filtered[mask]

    st.write(f"Records found: **{len(filtered)}**")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Individual Student Details")

    student_ids = analysis["STUDENT_ID"].dropna().tolist()

    selected_id = st.selectbox(
        "Select Student ID",
        student_ids
    )

    student = analysis[
        analysis["STUDENT_ID"] == selected_id
    ].iloc[0]

    st.write(f"### {student['NAME']}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Student ID", int(student["STUDENT_ID"]))
    col2.metric("Status", student["SUSPICIOUS_STATUS"])
    col3.metric("Anomaly Score", round(
        student["ANOMALY_SCORE"], 5
    ))

    st.write("**Detected Issues:**")
    st.write(student["DETECTED_ISSUES"])

    st.write("**Email:**", student.get("EMAIL", "Not available"))
    st.write("**Phone:**", student.get("PHONE", "Not available"))


# --------------------------------------------------
# ML ANOMALY DETECTION
# --------------------------------------------------

elif page == "ML Anomaly Detection":

    st.header("Isolation Forest Anomaly Detection")

    st.write(
        "The Isolation Forest model was trained in Google Colab. "
        "This dashboard displays its saved predictions."
    )

    anomalies = ml_report[
        ml_report["ML_RESULT"] == "Anomaly"
    ].copy()

    normal = ml_report[
        ml_report["ML_RESULT"] == "Normal"
    ].copy()

    col1, col2 = st.columns(2)

    col1.metric("Anomalies Detected", len(anomalies))
    col2.metric("Normal Records", len(normal))

    st.subheader("Anomaly Detection Results")

    st.dataframe(
        ml_report,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Flagged Anomalies")

    st.dataframe(
        anomalies,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Anomaly Score Chart")

    chart_data = ml_report.set_index("NAME")[
        "ANOMALY_SCORE"
    ]

    st.bar_chart(chart_data)

    st.caption(
        "Lower anomaly scores indicate records that the model "
        "considers more unusual. An anomaly is not automatically "
        "proof of incorrect data."
    )


# --------------------------------------------------
# ORACLE QUALITY REPORT
# --------------------------------------------------

elif page == "Oracle Quality Report":

    st.header("Oracle SQL / PL-SQL Quality Report")

    st.write(
        "These issues were detected by the "
        "ANALYZE_DATA_QUALITY stored procedure."
    )

    st.metric("Total Oracle Quality Issues", oracle_issues)

    st.dataframe(
        oracle_report,
        use_container_width=True,
        hide_index=True
    )

    if "ISSUE_TYPE" in oracle_report.columns:

        st.subheader("Issue Type Breakdown")

        oracle_counts = oracle_report[
            "ISSUE_TYPE"
        ].value_counts()

        st.bar_chart(oracle_counts)


# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------




# --------------------------------------------------
# DOWNLOAD REPORTS
# --------------------------------------------------

elif page == "Download Reports":

    st.header("Download Project Reports")

    st.write(
        "Download the analysis outputs generated "
        "by the Google Colab AIML model."
    )

    st.download_button(
        "Download Complete Analysis CSV",
        data=analysis.to_csv(index=False).encode("utf-8"),
        file_name="final_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "Download Analysis Summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="analysis_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "Download ML Anomaly Report CSV",
        data=ml_report.to_csv(index=False).encode("utf-8"),
        file_name="ml_anomaly_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "Download Oracle Quality Report CSV",
        data=oracle_report.to_csv(index=False).encode("utf-8"),
        file_name="quality_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    st.subheader("Complete PDF Analysis Report")

    try:
        pdf_data = create_pdf()

        st.download_button(
            "📄 Download Complete PDF Report",
            data=pdf_data,
            file_name="Database_Data_Quality_Analysis_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"PDF generation error: {e}")


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.header("About the Project")

    st.subheader("AI-Based Database Data-Quality Analyzer")

    st.write(
        "This project automatically identifies suspicious "
        "database records using SQL, PL/SQL and machine learning."
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - **Oracle Database:** Stores student, marks and attendance data.
    - **SQL:** Detects missing values, duplicates and invalid records.
    - **PL/SQL:** Automates data-quality analysis using a stored procedure.
    - **Python:** Performs data preprocessing and analysis.
    - **Pandas and NumPy:** Data manipulation.
    - **Scikit-learn:** Isolation Forest anomaly detection.
    - **Streamlit:** Interactive dashboard.
    """)

    st.subheader("Machine Learning Algorithm")

    st.write(
        "Isolation Forest is an unsupervised machine-learning "
        "algorithm used to identify unusual data patterns."
    )

    st.subheader("Project Dataset")

    st.write(f"Total student records: {total_students}")

    st.write(
        "The dashboard displays the saved Colab analysis. "
        "To refresh results after changing Oracle data, export "
        "the CSV files again, rerun the Colab notebook, and "
        "replace the corresponding GitHub files."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Database Data-Quality Analyzer | "
    "DBMS Lab Project | Oracle + Python + Machine Learning"
)
