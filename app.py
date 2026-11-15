import html
from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Database Data-Quality Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------
# VISUAL THEME
# ---------------------------------------------------------
st.markdown("""
<style>
    :root {
        --ink: #f1f5f9;
        --muted: #a7b4c8;
        --line: #334155;
        --surface: #172338;
        --accent: #60a5fa;
    }
    .stApp {
        background: linear-gradient(180deg, #080f1d 0%, #111b2e 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(8, 15, 29, 0.96); }
    [data-testid="stSidebar"] { background: #080f1d; }
    [data-testid="stSidebar"] * { color: #edf3ff !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.42rem 0.25rem;
        border-radius: 8px;
    }
    .hero {
        padding: 1.45rem 1.65rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #172554 0%, #263b83 58%, #4c1d95 100%);
        color: white;
        margin: 0.25rem 0 1.35rem 0;
        box-shadow: 0 12px 30px rgba(0,0,0,0.28);
        border: 1px solid rgba(147,197,253,0.18);
    }
    .hero h1 {
        color: white !important;
        font-size: clamp(1.55rem, 3vw, 2.25rem);
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.03em;
    }
    .hero p { color: #dbeafe !important; margin: 0; font-size: 1rem; }
    h1, h2, h3 { color: #e2e8f0 !important; letter-spacing: -0.02em; }
    p, label { color: #e2e8f0; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1b2a42, #142136);
        border: 1px solid #334766;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        box-shadow: 0 5px 16px rgba(0,0,0,0.18);
    }
    [data-testid="stMetricLabel"] { color: #a7b4c8 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 750; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #172338;
        border: 1px solid #334155;
        border-radius: 12px;
        overflow: hidden;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        background: linear-gradient(100deg, #2563eb, #6d28d9);
        color: white;
        border: 1px solid #6488ef;
    }
    div.stDownloadButton > button:hover, div.stButton > button:hover {
        background: #1d4ed8; color: white; border-color: #93c5fd;
    }
    .section-note {
        color: #a7b4c8; margin-top: -0.45rem;
        margin-bottom: 1rem; font-size: 0.93rem;
    }
    .info-card {
        background: #172338; border: 1px solid #334766;
        border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
    }
    .info-card h4 { color: #bfdbfe; margin: 0 0 0.45rem 0; }
    .info-card p { color: #cbd5e1; margin: 0; }
    .plain-table-wrap {
        overflow-x: auto; border: 1px solid #334155;
        border-radius: 12px; margin: 0.4rem 0 1rem 0;
    }
    table.plain-table { width: 100%; border-collapse: collapse; color: #e2e8f0; }
    .plain-table th {
        background: #243653; color: #dbeafe; text-align: left;
        padding: 0.7rem; white-space: nowrap;
    }
    .plain-table td {
        padding: 0.65rem 0.7rem; border-top: 1px solid #334155;
        white-space: nowrap;
    }
    .plain-table tr:nth-child(even) { background: #142136; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# COMMON HELPERS
# ---------------------------------------------------------
def style_chart(ax, fig):
    fig.patch.set_facecolor("#172338")
    ax.set_facecolor("#172338")
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#f1f5f9")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(color="#64748b", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)


def render_plain_table(dataframe, columns):
    """Render an escaped HTML table so sample emails are never auto-linked."""
    view = dataframe[columns].copy()
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for value in row.tolist():
            if pd.isna(value):
                value = ""
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows) if rows else (
        f'<tr><td colspan="{len(columns)}">No records found.</td></tr>'
    )
    st.markdown(
        '<div class="plain-table-wrap"><table class="plain-table">'
        f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody>"
        "</table></div>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PROJECT HEADER AND NAVIGATION
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>AI-Based Database Data-Quality Analyzer</h1>
    <p>Intelligent detection of database errors, inconsistencies and unusual records.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Section",
    [
        "About Project",
        "Dashboard",
        "Student Analysis",
        "Database Quality Report",
        "ML Anomaly Detection",
        "Download Reports"
    ]
)

# ---------------------------------------------------------
# LOAD PROJECT DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    students = pd.read_csv("students.csv")
    marks = pd.read_csv("marks.csv")
    attendance = pd.read_csv("attendance.csv")
    oracle_report = pd.read_csv("quality_report.csv")
    analysis = pd.read_csv("final_analysis.csv")
    summary = pd.read_csv("analysis_summary.csv")
    ml_report = pd.read_csv("ml_anomaly_report.csv")
    return students, marks, attendance, oracle_report, analysis, summary, ml_report


try:
    (
        students, marks, attendance, oracle_report,
        analysis, summary, ml_report
    ) = load_data()
except Exception as e:
    st.error(f"Unable to load project CSV files: {e}")
    st.info(
        "Check that all seven CSV files are present "
        "in the same GitHub folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------
# PREPARE DATA AND CALCULATE METRICS
# ---------------------------------------------------------
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
analysis["SUSPICIOUS_STATUS"] = analysis["SUSPICIOUS_STATUS"].fillna("Unknown")
analysis["DETECTED_ISSUES"] = analysis["DETECTED_ISSUES"].fillna("No issues detected")
ml_report["ML_RESULT"] = ml_report["ML_RESULT"].fillna("Unknown")

total_students = len(analysis)
missing_emails = int(pd.to_numeric(
    analysis["MISSING_EMAIL"], errors="coerce"
).fillna(0).sum())
duplicate_records = int(pd.to_numeric(
    analysis["IS_DUPLICATE"], errors="coerce"
).fillna(0).sum())
invalid_marks = int(pd.to_numeric(
    analysis["INVALID_MARK_COUNT"], errors="coerce"
).fillna(0).sum())
invalid_attendance = int(pd.to_numeric(
    analysis["INVALID_ATTENDANCE_COUNT"], errors="coerce"
).fillna(0).sum())
ml_anomalies = int((ml_report["ML_RESULT"] == "Anomaly").sum())
critical_records = int((analysis["SUSPICIOUS_STATUS"] == "Critical").sum())
warning_records = int((analysis["SUSPICIOUS_STATUS"] == "Warning").sum())
normal_records = int((analysis["SUSPICIOUS_STATUS"] == "Normal").sum())
database_issues = len(oracle_report)

# ---------------------------------------------------------
# PDF REPORT FUNCTION (kept outside page navigation)
# ---------------------------------------------------------
def create_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        "AI-Based Database Data-Quality Analyzer", styles["Title"]
    ))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Complete Project Analysis Report", styles["Heading2"]))
    elements.append(Paragraph(
        "Generated on: " + datetime.now().strftime("%d-%m-%Y %H:%M"),
        styles["Normal"]
    ))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("1. Analysis Summary", styles["Heading2"]))

    summary_data = [
        ["Metric", "Value"],
        ["Total Students", total_students],
        ["Database Issues", database_issues],
        ["Missing Emails", missing_emails],
        ["Duplicate Records", duplicate_records],
        ["Invalid Marks", invalid_marks],
        ["Invalid Attendance", invalid_attendance],
        ["ML Anomalies", ml_anomalies],
        ["Critical Records", critical_records],
        ["Warning Records", warning_records],
        ["Normal Records", normal_records]
    ]
    summary_table = Table(summary_data, colWidths=[250, 120])
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
    elements.append(Paragraph("2. Student-Wise Analysis", styles["Heading2"]))

    student_data = [["ID", "Name", "Status", "Detected Issues", "ML Score"]]
    for _, row in analysis.iterrows():
        student_data.append([
            str(row.get("STUDENT_ID", "")),
            str(row.get("NAME", "")),
            str(row.get("SUSPICIOUS_STATUS", "")),
            str(row.get("DETECTED_ISSUES", "")),
            str(round(row.get("ANOMALY_SCORE", 0), 5))
        ])
    student_table = Table(
        student_data, colWidths=[45, 85, 85, 330, 75], repeatRows=1
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
    elements.append(Paragraph("3. Database Quality Report", styles["Heading2"]))

    oracle_data = [["Issue ID", "Student ID", "Issue Type", "Description"]]
    for _, row in oracle_report.iterrows():
        oracle_data.append([
            str(row.get("ISSUE_ID", "")),
            str(row.get("STUDENT_ID", "")),
            str(row.get("ISSUE_TYPE", "")),
            str(row.get("DESCRIPTION", ""))
        ])
    oracle_table = Table(
        oracle_data, colWidths=[65, 75, 140, 350], repeatRows=1
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
    elements.append(Paragraph("4. Machine Learning Methodology", styles["Heading2"]))
    elements.append(Paragraph(
        "Isolation Forest was used for unsupervised anomaly detection. "
        "SQL and PL/SQL rules detect predefined data-quality issues, while "
        "the ML model identifies unusual combinations of student features. "
        "ML anomalies should be investigated and are not automatically "
        "proof of incorrect records.",
        styles["Normal"]
    ))
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------
# PAGE 1: ABOUT PROJECT
# ---------------------------------------------------------
if page == "About Project":
    st.subheader("About the Project")
    st.markdown(
        '<p class="section-note">Understand the purpose, technology and workflow behind the analyzer.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This project analyzes student database records to identify missing values,
    duplicate records, invalid marks, invalid attendance and unusual data patterns.
    It combines rule-based checks using Oracle SQL and PL/SQL with machine-learning
    anomaly detection using Isolation Forest.
    """)

    st.markdown("### Project Objectives")
    obj1, obj2, obj3 = st.columns(3)
    with obj1:
        st.markdown('<div class="info-card"><h4>Data Validation</h4><p>Identify missing, duplicate and invalid student records using database rules.</p></div>', unsafe_allow_html=True)
    with obj2:
        st.markdown('<div class="info-card"><h4>AI-Based Detection</h4><p>Find unusual combinations of student data using the Isolation Forest algorithm.</p></div>', unsafe_allow_html=True)
    with obj3:
        st.markdown('<div class="info-card"><h4>Clear Reporting</h4><p>Present database quality results in an interactive dashboard with downloadable reports.</p></div>', unsafe_allow_html=True)

    st.markdown("### Technology Stack")
    tech1, tech2 = st.columns(2)
    with tech1:
        st.markdown("""
        - **Oracle Database:** Stores student, marks and attendance data.
        - **SQL:** Retrieves and validates database records.
        - **PL/SQL:** Detects predefined data-quality issues.
        - **Python:** Processes exported CSV files.
        """)
    with tech2:
        st.markdown("""
        - **Pandas and NumPy:** Data processing and analysis.
        - **Scikit-learn:** Machine-learning implementation.
        - **Isolation Forest:** Unsupervised anomaly detection.
        - **Streamlit:** Interactive web dashboard.
        - **Matplotlib:** Data visualizations.
        """)

    st.markdown("### Project Workflow")
    workflow = [
        ("01", "Oracle Database", "Student, marks and attendance data are stored in Oracle."),
        ("02", "SQL / PL-SQL Checks", "Predefined rules identify missing, duplicate and invalid values."),
        ("03", "CSV Export", "Oracle tables and the quality report are exported as CSV files."),
        ("04", "Python + Machine Learning", "Python processes the files and Isolation Forest flags unusual records."),
        ("05", "Streamlit Dashboard", "Analysis results are displayed and can be downloaded.")
    ]
    for number, heading, description in workflow:
        with st.container(border=True):
            st.markdown(f"**{number} · {heading}**")
            st.caption(description)

    st.markdown("### Data Quality Checks")
    st.write(
        "- Missing student information\n"
        "- Duplicate student records\n"
        "- Invalid marks (outside the 0–100 range)\n"
        "- Invalid attendance values"
    )

    st.markdown("### Understanding the ML Results")
    st.info(
        "Isolation Forest is an unsupervised algorithm that identifies records "
        "that differ from common data patterns. An ML anomaly is a record worth "
        "investigating; it does not automatically mean the record is incorrect."
    )

    st.markdown("### Data Refresh and Project Limitation")
    st.warning(
        "This dashboard reads CSV files exported from Oracle; it does not connect "
        "directly to Oracle in real time. To refresh the dashboard, export the latest "
        "Oracle CSV files, rerun the Python analysis, and replace the corresponding "
        "CSV files in the GitHub repository."
    )

    st.markdown("### Current Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students Analyzed", total_students)
    c2.metric("Database Issues", database_issues)
    c3.metric("ML Anomalies", ml_anomalies)


# ---------------------------------------------------------
# PAGE 2: DASHBOARD
# ---------------------------------------------------------
elif page == "Dashboard":
    st.subheader("Project Dashboard")
    st.markdown(
        '<p class="section-note">A snapshot of database health and machine-learning findings.</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Database Issues", database_issues)
    col3.metric("ML Anomalies", ml_anomalies)
    col4.metric("Critical Records", critical_records)

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Missing Emails", missing_emails)
    col6.metric("Duplicate Records", duplicate_records)
    col7.metric("Invalid Marks", invalid_marks)
    col8.metric("Invalid Attendance", invalid_attendance)

    st.markdown(
        f"**Record classification:** 🔴 Critical: {critical_records}  |  "
        f"🟠 Warning: {warning_records}  |  🟢 Normal: {normal_records}"
    )
    st.divider()
    st.subheader("Data Quality Overview")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Student Record Classification**")
        labels = ["Critical", "Warning", "Normal"]
        values = [critical_records, warning_records, normal_records]
        colors_list = ["#fb7185", "#fbbf24", "#34d399"]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        fig.patch.set_facecolor("#172338")
        ax.set_facecolor("#172338")
        wedges, _ = ax.pie(
            values, colors=colors_list, startangle=90,
            wedgeprops={"width": 0.42, "edgecolor": "#172338", "linewidth": 3}
        )
        total = sum(values)
        ax.text(0, 0.08, f"{total}", ha="center", va="center",
                color="white", fontsize=24, fontweight="bold")
        ax.text(0, -0.17, "Students", ha="center", va="center",
                color="#a7b4c8", fontsize=10)
        ax.legend(
            wedges, [f"{name}: {value}" for name, value in zip(labels, values)],
            loc="lower center", bbox_to_anchor=(0.5, -0.12),
            ncol=3, frameon=False, labelcolor="#e2e8f0", fontsize=8
        )
        ax.set_title("Classification of Student Records", color="#f1f5f9", pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_col2:
        st.markdown("**Detected Data Quality Issues**")
        issue_data = pd.DataFrame({
            "Issue": ["Missing Emails", "Duplicates", "Invalid Marks", "Invalid Attendance"],
            "Count": [missing_emails, duplicate_records, invalid_marks, invalid_attendance]
        }).sort_values("Count", ascending=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        style_chart(ax, fig)
        bars = ax.barh(
            issue_data["Issue"], issue_data["Count"],
            color=["#a78bfa", "#60a5fa", "#fbbf24", "#fb7185"],
            height=0.62
        )
        ax.set_xlabel("Number of Issues")
        ax.set_title("Database Quality Checks", pad=12)
        ax.xaxis.grid(True)
        for bar in bars:
            ax.text(
                bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width())}", va="center",
                color="#f1f5f9", fontweight="bold"
            )
        ax.set_xlim(0, max(issue_data["Count"].max() + 1, 2))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.subheader("Student Analysis Preview")
    preview_columns = [
        col for col in [
            "STUDENT_ID", "NAME", "SUSPICIOUS_STATUS",
            "DETECTED_ISSUES", "ANOMALY_SCORE"
        ] if col in analysis.columns
    ]
    st.dataframe(
        analysis[preview_columns],
        use_container_width=True, height=340, hide_index=True
    )


# ---------------------------------------------------------
# PAGE 3: STUDENT ANALYSIS
# ---------------------------------------------------------
elif page == "Student Analysis":
    st.subheader("Student-Wise Data Quality Analysis")
    st.markdown(
        '<p class="section-note">Search student records, filter by classification and inspect individual findings.</p>',
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        "Search by Student ID or Name",
        placeholder="Enter student ID or student name..."
    )
    status_options = ["All"] + sorted(
        analysis["SUSPICIOUS_STATUS"].dropna().unique().tolist()
    )
    selected_status = st.selectbox("Filter by Classification", status_options)

    filtered_data = analysis.copy()
    if search_text:
        search_mask = filtered_data["STUDENT_ID"].astype(str).str.contains(
            search_text, case=False, na=False
        )
        if "NAME" in filtered_data.columns:
            search_mask = search_mask | filtered_data["NAME"].astype(str).str.contains(
                search_text, case=False, na=False
            )
        filtered_data = filtered_data[search_mask]
    if selected_status != "All":
        filtered_data = filtered_data[
            filtered_data["SUSPICIOUS_STATUS"] == selected_status
        ]

    st.write(f"**Records found:** {len(filtered_data)}")
    display_columns = [
        col for col in [
            "STUDENT_ID", "NAME", "EMAIL", "PHONE",
            "SUSPICIOUS_STATUS", "DETECTED_ISSUES", "ANOMALY_SCORE"
        ] if col in filtered_data.columns
    ]
    if display_columns:
        render_plain_table(filtered_data, display_columns)

    st.divider()
    st.subheader("View Individual Student Details")
    student_ids = filtered_data["STUDENT_ID"].dropna().tolist()
    if student_ids:
        selected_student = st.selectbox("Select Student ID", student_ids)
        student_record = filtered_data[
            filtered_data["STUDENT_ID"] == selected_student
        ]
        if not student_record.empty:
            record = student_record.iloc[0]
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown("**Student Information**")
                st.write("Student ID:", record.get("STUDENT_ID", "Not available"))
                st.write("Name:", record.get("NAME", "Not available"))
                st.text(f"Email: {record.get('EMAIL', 'Not available')}")
                st.text(f"Phone: {record.get('PHONE', 'Not available')}")
            with detail_col2:
                st.markdown("**Analysis Results**")
                st.write("Classification:", record.get("SUSPICIOUS_STATUS", "Not available"))
                st.write("Detected Issues:", record.get("DETECTED_ISSUES", "No issues detected"))
                st.write("Anomaly Score:", record.get("ANOMALY_SCORE", "Not available"))
    else:
        st.info("No student records match your search or filter.")


# ---------------------------------------------------------
# PAGE 4: DATABASE QUALITY REPORT
# ---------------------------------------------------------
elif page == "Database Quality Report":
    st.subheader("Database Quality Report")
    st.markdown(
        '<p class="section-note">Issues detected using Oracle SQL and PL/SQL validation rules.</p>',
        unsafe_allow_html=True
    )
    st.metric("Total Database Issues", database_issues)
    st.divider()
    st.markdown("### Issue Type Breakdown")

    if "ISSUE_TYPE" in oracle_report.columns:
        issue_counts = (
            oracle_report["ISSUE_TYPE"].fillna("Unknown")
            .value_counts().reset_index()
        )
        issue_counts.columns = ["Issue Type", "Count"]
        col1, col2 = st.columns([0.9, 1.4])
        with col1:
            st.dataframe(issue_counts, use_container_width=True, hide_index=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            style_chart(ax, fig)
            chart_data = issue_counts.sort_values("Count", ascending=True)
            bars = ax.barh(
                chart_data["Issue Type"], chart_data["Count"],
                color=["#60a5fa", "#a78bfa", "#fbbf24", "#fb7185"][:len(chart_data)],
                height=0.62
            )
            ax.set_xlabel("Number of Issues")
            ax.set_title("Database Issue Distribution", pad=12)
            for bar in bars:
                ax.text(
                    bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())), va="center",
                    color="#f1f5f9", fontweight="bold"
                )
            ax.set_xlim(0, max(chart_data["Count"].max() + 1, 2))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()
    st.markdown("### Complete Database Quality Report")
    st.dataframe(oracle_report, use_container_width=True, hide_index=True)
# ---------------------------------------------------------
# PAGE 5: ML ANOMALY DETECTION
# ---------------------------------------------------------
elif page == "ML Anomaly Detection":
    st.subheader("Machine Learning Anomaly Detection")
    st.markdown(
        '<p class="section-note">Isolation Forest identifies unusual student records based on their data patterns.</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total ML Anomalies", ml_anomalies)
    col2.metric("Normal ML Records", total_students - ml_anomalies)
    col3.metric(
        "Anomaly Percentage",
        f"{(ml_anomalies / total_students * 100):.1f}%"
        if total_students > 0 else "0%"
    )

    st.divider()
    st.markdown("### ML Anomaly Results")
    if "ML_RESULT" in ml_report.columns:
        anomaly_data = ml_report[ml_report["ML_RESULT"] == "Anomaly"]
        st.write("**Records identified as anomalies:**")
        st.dataframe(anomaly_data, use_container_width=True, hide_index=True)
        normal_data = ml_report[ml_report["ML_RESULT"] != "Anomaly"]
        with st.expander("View normal ML records"):
            st.dataframe(normal_data, use_container_width=True, hide_index=True)
    else:
        st.warning("The ML report does not contain the ML_RESULT column.")

    st.divider()
    st.markdown("### Student Anomaly Scores")
    if "STUDENT_ID" in analysis.columns and "ANOMALY_SCORE" in analysis.columns:
        score_data = analysis.sort_values("ANOMALY_SCORE").copy()
        anomaly_ids = set()
        if "STUDENT_ID" in ml_report.columns:
            anomaly_ids = set(
                pd.to_numeric(
                    ml_report.loc[ml_report["ML_RESULT"] == "Anomaly", "STUDENT_ID"],
                    errors="coerce"
                ).dropna().tolist()
            )
        bar_colors = [
            "#fb7185" if sid in anomaly_ids else "#60a5fa"
            for sid in score_data["STUDENT_ID"].tolist()
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        style_chart(ax, fig)
        bars = ax.bar(
            score_data["STUDENT_ID"].astype("Int64").astype(str),
            score_data["ANOMALY_SCORE"],
            color=bar_colors, width=0.72
        )
        ax.axhline(0, color="#f8fafc", linewidth=1.1, linestyle="--", alpha=0.8)
        ax.set_xlabel("Student ID")
        ax.set_ylabel("Anomaly Score")
        ax.set_title("Isolation Forest Anomaly Scores", pad=12)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")
        for bar, value in zip(bars, score_data["ANOMALY_SCORE"].tolist()):
            offset = 0.006 if value >= 0 else -0.012
            ax.text(
                bar.get_x() + bar.get_width()/2, value + offset,
                f"{value:.3f}", ha="center",
                va="bottom" if value >= 0 else "top",
                color="#e2e8f0", fontsize=7, rotation=90
            )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        "A lower anomaly score indicates a more unusual record. An ML anomaly "
        "is a record worth investigating; it does not automatically mean the "
        "data is incorrect."
    )


# ---------------------------------------------------------
# PAGE 6: DOWNLOAD REPORTS
# ---------------------------------------------------------
elif page == "Download Reports":
    st.subheader("Download Project Reports")
    st.markdown(
        '<p class="section-note">Export analysis results and the complete project report.</p>',
        unsafe_allow_html=True
    )

    st.markdown("### CSV Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Final Analysis CSV",
            data=analysis.to_csv(index=False).encode("utf-8"),
            file_name="final_analysis.csv", mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            label="Download Database Quality Report",
            data=oracle_report.to_csv(index=False).encode("utf-8"),
            file_name="quality_report.csv", mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            label="Download Students CSV",
            data=students.to_csv(index=False).encode("utf-8"),
            file_name="students.csv", mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            label="Download Marks CSV",
            data=marks.to_csv(index=False).encode("utf-8"),
            file_name="marks.csv", mime="text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="Download ML Anomaly Report",
            data=ml_report.to_csv(index=False).encode("utf-8"),
            file_name="ml_anomaly_report.csv", mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            label="Download Analysis Summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="analysis_summary.csv", mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            label="Download Attendance CSV",
            data=attendance.to_csv(index=False).encode("utf-8"),
            file_name="attendance.csv", mime="text/csv",
            use_container_width=True
        )

    st.divider()
    st.markdown("### Complete PDF Report")
    if st.button("Generate PDF Report"):
        try:
            pdf_file = create_pdf()
            st.download_button(
                label="Download Complete PDF Report",
                data=pdf_file,
                file_name="Database_Data_Quality_Report.pdf",
                mime="application/pdf"
            )
            st.success("PDF report generated successfully.")
        except Exception as e:
            st.error(f"Unable to generate PDF: {e}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:10px;">
        <p><b>AI-Based Database Data-Quality Analyzer</b></p>
        <p>DBMS Lab Project | Oracle · Python · Machine Learning · Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
