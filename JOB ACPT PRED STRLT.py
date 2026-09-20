import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "cleaned_job_placement_dataset.csv"
    ) 

    return data


@st.cache_resource
def load_model():

    model = joblib.load(
        "job_acceptance_model.pkl"
    )

    return model


df = load_data()
model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎯 Job Acceptance Prediction System"
)

st.write(
    "HR Analytics Dashboard for Job Placement and Acceptance Prediction"
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_candidates = len(df)

placement_rate = (
    (df["status"] == "Placed").mean()
    * 100
)

job_acceptance_rate = placement_rate

average_interview_score = (
    df["interview_average"].mean()
)

average_skills_match = (
    df["skills_match_percentage"].mean()
)


# ============================================================
# KPI DISPLAY
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Candidates",
    f"{total_candidates:,}"
)

col2.metric(
    "Placement Rate",
    f"{placement_rate:.2f}%"
)

col3.metric(
    "Job Acceptance Rate",
    f"{job_acceptance_rate:.2f}%"
)


col4, col5, col6 = st.columns(3)

col4.metric(
    "Average Interview Score",
    f"{average_interview_score:.2f}"
)

col5.metric(
    "Average Skills Match %",
    f"{average_skills_match:.2f}%"
)

col6.metric(
    "Offer Dropout Rate",
    "N/A"
)


st.divider()


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("Filters")


selected_tier = st.sidebar.selectbox(
    "Company Tier",
    ["All"] +
    sorted(
        df["company_tier"]
        .dropna()
        .unique()
        .tolist()
    )
)


if selected_tier != "All":

    filtered_df = df[
        df["company_tier"] == selected_tier
    ]

else:

    filtered_df = df.copy()


# ============================================================
# PLACEMENT DISTRIBUTION
# ============================================================

st.subheader(
    "📌 Placement Status Distribution"
)

status_counts = (
    filtered_df["status"]
    .value_counts()
)

st.bar_chart(status_counts)


# ============================================================
# COMPANY TIER ANALYSIS
# ============================================================

st.subheader(
    "🏢 Placement Rate by Company Tier"
)

tier_analysis = pd.crosstab(
    filtered_df["company_tier"],
    filtered_df["status"],
    normalize="index"
) * 100

st.bar_chart(tier_analysis)


# ============================================================
# EXPERIENCE ANALYSIS
# ============================================================

st.subheader(
    "💼 Experience Category vs Placement"
)

experience_analysis = pd.crosstab(
    filtered_df["experience_category"],
    filtered_df["status"],
    normalize="index"
) * 100

st.bar_chart(experience_analysis)


# ============================================================
# COMPETITION ANALYSIS
# ============================================================

st.subheader(
    "📈 Competition Level vs Placement"
)

competition_analysis = pd.crosstab(
    filtered_df["competition_level"],
    filtered_df["status"],
    normalize="index"
) * 100

st.bar_chart(competition_analysis)


# ============================================================
# SKILLS MATCH ANALYSIS
# ============================================================

st.subheader(
    "🎓 Skills Match Level"
)

skills_analysis = (
    filtered_df["skills_match_level"]
    .value_counts()
)

st.bar_chart(skills_analysis)


# ============================================================
# INTERVIEW PERFORMANCE
# ============================================================

st.subheader(
    "🗣️ Interview Performance Category"
)

interview_analysis = (
    filtered_df[
        "interview_performance_category"
    ]
    .value_counts()
)

st.bar_chart(interview_analysis)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.header(
    "🔮 Individual Job Acceptance Prediction"
)

st.write(
    "Enter candidate details to predict job acceptance."
)


col1, col2 = st.columns(2)


with col1:

    age_years = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        sorted(df["gender"].unique())
    )

    ssc_percentage = st.number_input(
        "SSC Percentage",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    hsc_percentage = st.number_input(
        "HSC Percentage",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    degree_percentage = st.number_input(
        "Degree Percentage",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    degree_specialization = st.selectbox(
        "Degree Specialization",
        sorted(
            df[
                "degree_specialization"
            ].unique()
        )
    )

    technical_score = st.number_input(
        "Technical Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    aptitude_score = st.number_input(
        "Aptitude Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    communication_score = st.number_input(
        "Communication Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    skills_match_percentage = st.number_input(
        "Skills Match %",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    certifications_count = st.number_input(
        "Certifications Count",
        min_value=0,
        max_value=20,
        value=2
    )

    internship_experience = st.selectbox(
        "Internship Experience",
        sorted(
            df[
                "internship_experience"
            ].unique()
        )
    )

    years_of_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=30,
        value=1
    )


with col2:

    career_switch_willingness = st.selectbox(
        "Career Switch Willingness",
        sorted(
            df[
                "career_switch_willingness"
            ].unique()
        )
    )

    relevant_experience = st.selectbox(
        "Relevant Experience",
        sorted(
            df[
                "relevant_experience"
            ].unique()
        )
    )

    previous_ctc_lpa = st.number_input(
        "Previous CTC (LPA)",
        min_value=0.0,
        value=5.0
    )

    expected_ctc_lpa = st.number_input(
        "Expected CTC (LPA)",
        min_value=0.0,
        value=7.0
    )

    company_tier = st.selectbox(
        "Company Tier",
        sorted(
            df["company_tier"].unique()
        )
    )

    job_role_match = st.selectbox(
        "Job Role Match",
        sorted(
            df["job_role_match"].unique()
        )
    )

    competition_level = st.selectbox(
        "Competition Level",
        sorted(
            df["competition_level"].unique()
        )
    )

    bond_requirement = st.selectbox(
        "Bond Requirement",
        sorted(
            df["bond_requirement"].unique()
        )
    )

    notice_period_days = st.number_input(
        "Notice Period (Days)",
        min_value=0.0,
        value=30.0
    )

    layoff_history = st.selectbox(
        "Layoff History",
        sorted(
            df["layoff_history"].unique()
        )
    )

    employment_gap_months = st.number_input(
        "Employment Gap (Months)",
        min_value=0.0,
        value=0.0
    )

    relocation_willingness = st.selectbox(
        "Relocation Willingness",
        sorted(
            df["relocation_willingness"].unique()
        )
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Job Acceptance"
):

    input_data = pd.DataFrame({

        "age_years": [age_years],

        "gender": [gender],

        "ssc_percentage": [
            ssc_percentage
        ],

        "hsc_percentage": [
            hsc_percentage
        ],

        "degree_percentage": [
            degree_percentage
        ],

        "degree_specialization": [
            degree_specialization
        ],

        "technical_score": [
            technical_score
        ],

        "aptitude_score": [
            aptitude_score
        ],

        "communication_score": [
            communication_score
        ],

        "skills_match_percentage": [
            skills_match_percentage
        ],

        "certifications_count": [
            certifications_count
        ],

        "internship_experience": [
            internship_experience
        ],

        "years_of_experience": [
            years_of_experience
        ],

        "career_switch_willingness": [
            career_switch_willingness
        ],

        "relevant_experience": [
            relevant_experience
        ],

        "previous_ctc_lpa": [
            previous_ctc_lpa
        ],

        "expected_ctc_lpa": [
            expected_ctc_lpa
        ],

        "company_tier": [
            company_tier
        ],

        "job_role_match": [
            job_role_match
        ],

        "competition_level": [
            competition_level
        ],

        "bond_requirement": [
            bond_requirement
        ],

        "notice_period_days": [
            notice_period_days
        ],

        "layoff_history": [
            layoff_history
        ],

        "employment_gap_months": [
            employment_gap_months
        ],

        "relocation_willingness": [
            relocation_willingness
        ],

        # Feature engineered columns
        "academic_average": [
            (
                ssc_percentage
                + hsc_percentage
                + degree_percentage
            ) / 3
        ],

        "experience_category": [
            (
                "Fresher"
                if years_of_experience == 0
                else
                "Junior"
                if years_of_experience <= 2
                else "Senior"
            )
        ],

        "academic_performance_band": [
            (
                "Low"
                if (
                    (
                        ssc_percentage
                        + hsc_percentage
                        + degree_percentage
                    ) / 3
                ) < 60
                else
                "Medium"
                if (
                    (
                        ssc_percentage
                        + hsc_percentage
                        + degree_percentage
                    ) / 3
                ) < 75
                else "High"
            )
        ],

        "skills_match_level": [
            (
                "Low"
                if skills_match_percentage < 60
                else
                "Medium"
                if skills_match_percentage < 80
                else "High"
            )
        ],

        "interview_average": [
            (
                technical_score
                + aptitude_score
                + communication_score
            ) / 3
        ],

        "interview_performance_category": [
            (
                "Low"
                if (
                    (
                        technical_score
                        + aptitude_score
                        + communication_score
                    ) / 3
                ) < 60
                else
                "Medium"
                if (
                    (
                        technical_score
                        + aptitude_score
                        + communication_score
                    ) / 3
                ) < 80
                else "High"
            )
        ]
    })


    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    st.subheader(
        "Prediction Result"
    )

    if prediction == 1:

        st.success(
            "✅ Predicted Status: Job Accepted"
        )

    else:

        st.warning(
            "❌ Predicted Status: Job Rejected"
        )


    st.metric(
        "Acceptance Probability",
        f"{probability * 100:.2f}%"
    )