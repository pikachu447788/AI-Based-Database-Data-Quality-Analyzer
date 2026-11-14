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


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Database Data Quality Analyzer",
    page_icon="🔍",
    layout="wide"
)


# VISUAL THEME
st.markdown("""
<style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #e5eaf2;
        --surface: #ffffff;
        --soft: #f5f7fb;
        --accent: #3157a4;
    }

    .stApp {
        background: linear-gradient(180deg, #f7f9fd 0%, #ffffff 42%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 253, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #101a2d;
    }

    [data-testid="stSidebar"] * {
        color: #edf3ff !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.35rem 0.2rem;
    }

    .hero {
        padding: 1.35rem 1.55rem;
        border-radius: 18px;
        background: linear-gradient(
            120deg, #142747 0%, #294f91 62%, #3b6bb5 100%
        );
        color: white;
        margin: 0.25rem 0 1.35rem 0;
        box-shadow: 0 12px 28px rgba(25, 48, 88, 0.14);
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(1.55rem, 3vw, 2.25rem);
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
    }

    .hero p {
        color: #dce8ff !important;
        margin: 0;
        font-size: 0.98rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    h2 {
        color: #172b4d;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 14px rgba(24, 43, 77, 0.045);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #172b4d;
        font-weight: 750;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #cbd7eb;
    }

    div.stDownloadButton > button {
        background: #f1f5fc;
        color: #1f3f78;
    }

    hr {
        border-color: var(--line);
    }

    .section-note {
        color: #64748b;
        margin-top: -0.45rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>AI-Based Database Data-Quality Analyzer</h1>
    <p>Oracle Database · SQL / PL-SQL · Python · Isolation Forest</p>
</div>
""", unsafe_allow_html=True)


# NAVIGATION
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


# LOAD PROJECT DATA
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


# PREPARE DATA
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


# CALCULATE METRICS
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


# PDF REPORT FUNCTION
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
        "Generated on: " + datetime.now().strftime("%d-%m-%Y %H:%M"),
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
    
# ==========================================
# PAGE 1: DASHBOARD
# ==========================================

if page == "Dashboard":

    st.subheader("Project Dashboard")

    st.markdown(
        '<p class="section-note">A quick overview of database '
        'quality checks and machine-learning results.</p>',
        unsafe_allow_html=True
    )

    # MAIN METRICS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Oracle Issues", oracle_issues)
    col3.metric("ML Anomalies", ml_anomalies)
    col4.metric("Critical Records", critical_records)

    st.write("")

    # QUALITY METRICS
    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Missing Emails", missing_emails)
    col6.metric("Duplicate Records", duplicate_records)
    col7.metric("Invalid Marks", invalid_marks)
    col8.metric("Invalid Attendance", invalid_attendance)

    st.write("")

    st.markdown(
        f"**Record classification:** "
        f"🔴 Critical: {critical_records}  |  "
        f"🟠 Warning: {warning_records}  |  "
        f"🟢 Normal: {normal_records}"
    )

    st.divider()

    # CHARTS
    st.subheader("Data Quality Overview")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("**Student Record Classification**")

        status_data = pd.DataFrame({
            "Status": ["Critical", "Warning", "Normal"],
            "Students": [
                critical_records,
                warning_records,
                normal_records
            ]
        })

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            status_data["Status"],
            status_data["Students"]
        )

        ax.set_ylabel("Number of Students")
        ax.set_title("Classification of Student Records")
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:

        st.markdown("**Detected Data Quality Issues**")

        issue_data = pd.DataFrame({
            "Issue": [
                "Missing Emails",
                "Duplicates",
                "Invalid Marks",
                "Invalid Attendance"
            ],
            "Count": [
                missing_emails,
                duplicate_records,
                invalid_marks,
                invalid_attendance
            ]
        })

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            issue_data["Issue"],
            issue_data["Count"]
        )

        ax.set_xlabel("Number of Issues")
        ax.set_title("Oracle Data Quality Checks")
        ax.grid(axis="x", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # STUDENT DATA PREVIEW
    st.subheader("Student Analysis Preview")

    preview_columns = [
        col for col in [
            "STUDENT_ID",
            "NAME",
            "SUSPICIOUS_STATUS",
            "DETECTED_ISSUES",
            "ANOMALY_SCORE"
        ]
        if col in analysis.columns
    ]

    st.dataframe(
        analysis[preview_columns],
        use_container_width=True,
        height=360,
        hide_index=True
    )


# ==========================================
# PAGE 2: STUDENT ANALYSIS
# ==========================================

elif page == "Student Analysis":

    st.subheader("Student-Wise Data Quality Analysis")

    st.markdown(
        '<p class="section-note">Search student records, filter '
        'by classification, and inspect detected issues.</p>',
        unsafe_allow_html=True
    )

    # SEARCH
    search_text = st.text_input(
        "Search by Student ID or Name",
        placeholder="Enter student ID or student name..."
    )

    # FILTER
    status_options = ["All"] + sorted(
        analysis["SUSPICIOUS_STATUS"].dropna().unique().tolist()
    )

    selected_status = st.selectbox(
        "Filter by Classification",
        status_options
    )

    filtered_data = analysis.copy()

    if search_text:

        search_mask = (
            filtered_data["STUDENT_ID"].astype(str).str.contains(
                search_text,
                case=False,
                na=False
            )
        )

        if "NAME" in filtered_data.columns:
            search_mask = search_mask | (
                filtered_data["NAME"].astype(str).str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            )

        filtered_data = filtered_data[search_mask]

    if selected_status != "All":

        filtered_data = filtered_data[
            filtered_data["SUSPICIOUS_STATUS"] == selected_status
        ]

    st.write(f"**Records found:** {len(filtered_data)}")

    display_columns = [
        col for col in [
            "STUDENT_ID",
            "NAME",
            "EMAIL",
            "PHONE",
            "SUSPICIOUS_STATUS",
            "DETECTED_ISSUES",
            "ANOMALY_SCORE"
        ]
        if col in filtered_data.columns
    ]

    st.dataframe(
        filtered_data[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # SELECT A STUDENT
    st.subheader("View Individual Student Details")

    student_ids = filtered_data["STUDENT_ID"].dropna().tolist()

    if student_ids:

        selected_student = st.selectbox(
            "Select Student ID",
            student_ids
        )

        student_record = filtered_data[
            filtered_data["STUDENT_ID"] == selected_student
        ]

        if not student_record.empty:

            record = student_record.iloc[0]

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.markdown("**Student Information**")

                st.write(
                    "Student ID:",
                    record.get("STUDENT_ID", "Not available")
                )

                st.write(
                    "Name:",
                    record.get("NAME", "Not available")
                )

                st.write(
                    "Email:",
                    record.get("EMAIL", "Not available")
                )

                st.write(
                    "Phone:",
                    record.get("PHONE", "Not available")
                )

            with detail_col2:

                st.markdown("**Analysis Results**")

                st.write(
                    "Classification:",
                    record.get("SUSPICIOUS_STATUS", "Not available")
                )

                st.write(
                    "Detected Issues:",
                    record.get("DETECTED_ISSUES", "No issues detected")
                )

                st.write(
                    "Anomaly Score:",
                    record.get("ANOMALY_SCORE", "Not available")
                )

    else:

        st.info("No student records match your search or filter.")
        
# ==========================================
# PAGE 3: ML ANOMALY DETECTION
# ==========================================

elif page == "ML Anomaly Detection":

    st.subheader("Machine Learning Anomaly Detection")

    st.markdown(
        '<p class="section-note">Isolation Forest identifies '
        'unusual student records based on their data patterns.</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total ML Anomalies", ml_anomalies)
    col2.metric("Normal Records", total_students - ml_anomalies)
    col3.metric(
        "Anomaly Percentage",
        f"{(ml_anomalies / total_students * 100):.1f}%"
        if total_students > 0 else "0%"
    )

    st.divider()

    st.markdown("### ML Anomaly Results")

    if "ML_RESULT" in ml_report.columns:

        anomaly_data = ml_report[
            ml_report["ML_RESULT"] == "Anomaly"
        ]

        st.write("**Records identified as anomalies:**")

        st.dataframe(
            anomaly_data,
            use_container_width=True,
            hide_index=True
        )

        normal_data = ml_report[
            ml_report["ML_RESULT"] != "Anomaly"
        ]

        with st.expander("View normal ML records"):

            st.dataframe(
                normal_data,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.warning(
            "The ML report does not contain the ML_RESULT column."
        )

    st.divider()

    # ANOMALY SCORE CHART
    st.markdown("### Student Anomaly Scores")

    if (
        "STUDENT_ID" in analysis.columns
        and "ANOMALY_SCORE" in analysis.columns
    ):

        score_data = analysis.sort_values(
            by="ANOMALY_SCORE"
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.bar(
            score_data["STUDENT_ID"].astype(str),
            score_data["ANOMALY_SCORE"]
        )

        ax.axhline(
            0,
            linewidth=1,
            linestyle="--"
        )

        ax.set_xlabel("Student ID")
        ax.set_ylabel("Anomaly Score")
        ax.set_title("Isolation Forest Anomaly Scores")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "A lower anomaly score indicates a more unusual record. "
        "An ML anomaly is a record worth investigating; it does "
        "not automatically mean the data is incorrect."
    )


# ==========================================
# PAGE 4: ORACLE QUALITY REPORT
# ==========================================

elif page == "Oracle Quality Report":

    st.subheader("Oracle Database Quality Report")

    st.markdown(
        '<p class="section-note">Issues detected using SQL and '
        'PL/SQL validation rules.</p>',
        unsafe_allow_html=True
    )

    st.metric("Total Oracle Issues", oracle_issues)

    st.divider()

    st.markdown("### Issue Type Breakdown")

    if "ISSUE_TYPE" in oracle_report.columns:

        issue_counts = (
            oracle_report["ISSUE_TYPE"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        issue_counts.columns = ["Issue Type", "Count"]

        col1, col2 = st.columns([1, 1])

        with col1:

            st.dataframe(
                issue_counts,
                use_container_width=True,
                hide_index=True
            )

        with col2:

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.bar(
                issue_counts["Issue Type"],
                issue_counts["Count"]
            )

            ax.set_ylabel("Number of Issues")
            ax.set_title("Oracle Issue Distribution")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.25)

            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    st.markdown("### Complete Oracle Quality Report")

    st.dataframe(
        oracle_report,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 5: DOWNLOAD REPORTS
# ==========================================

elif page == "Download Reports":

    st.subheader("Download Project Reports")

    st.markdown(
        '<p class="section-note">Download the analysis results '
        'and generated project report.</p>',
        unsafe_allow_html=True
    )

    st.markdown("### CSV Reports")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="Download Final Analysis CSV",
            data=analysis.to_csv(index=False).encode("utf-8"),
            file_name="final_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Oracle Quality Report",
            data=oracle_report.to_csv(index=False).encode("utf-8"),
            file_name="quality_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Students CSV",
            data=students.to_csv(index=False).encode("utf-8"),
            file_name="students.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="Download ML Anomaly Report",
            data=ml_report.to_csv(index=False).encode("utf-8"),
            file_name="ml_anomaly_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Analysis Summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="analysis_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Marks CSV",
            data=marks.to_csv(index=False).encode("utf-8"),
            file_name="marks.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="Download Attendance CSV",
            data=attendance.to_csv(index=False).encode("utf-8"),
            file_name="attendance.csv",
            mime="text/csv",
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


# ==========================================
# PAGE 6: ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.subheader("About the Project")

    st.markdown("""
    ### AI-Based Database Data-Quality Analyzer

    This project analyzes student database records to identify
    missing values, duplicate records, invalid marks, invalid
    attendance, and unusual data patterns.

    ### Technologies Used

    - **Oracle Database:** Stores student, marks, and attendance data.
    - **SQL:** Retrieves and validates database records.
    - **PL/SQL:** Automatically detects predefined data-quality issues.
    - **Python:** Processes exported CSV files.
    - **Pandas and NumPy:** Data cleaning and analysis.
    - **Scikit-learn:** Machine-learning anomaly detection.
    - **Isolation Forest:** Identifies unusual records.
    - **Streamlit:** Provides the interactive web dashboard.

    ### Project Workflow

    1. Student data is stored in Oracle Database.
    2. SQL and PL/SQL identify predefined data-quality issues.
    3. Oracle tables and reports are exported as CSV files.
    4. Python processes the CSV files.
    5. Isolation Forest detects unusual student records.
    6. Results are displayed in this Streamlit dashboard.

    ### Data Quality Checks

    - Missing student information
    - Duplicate student records
    - Invalid marks
    - Invalid attendance

    ### Machine Learning

    Isolation Forest is an unsupervised machine-learning algorithm.
    It identifies records that differ significantly from the
    overall data patterns.

    ### Important Note

    This dashboard uses CSV files exported from Oracle Database.
    It does not connect directly to Oracle in real time.

    To refresh the dashboard with updated data, export the latest
    Oracle CSV files, rerun the Python analysis, and replace the
    corresponding CSV files in the GitHub repository.
    """)

    st.divider()

    st.markdown("### Project Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Students Analyzed", total_students)
    col2.metric("Oracle Issues Found", oracle_issues)
    col3.metric("ML Anomalies Found", ml_anomalies)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#718096; padding:10px;">
        <p><b>AI-Based Database Data-Quality Analyzer</b></p>
        <p>DBMS Lab Project | Oracle · Python · Machine Learning · Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
