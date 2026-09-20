# ============================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EDA + DATA CLEANING + FEATURE ENGINEERING + ML
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

import joblib
import os


# ============================================================
# 1. LOAD DATASET
# ============================================================

job_pred = pd.read_csv("C:/Users/V MANINDRA KUMAR/Desktop/HR_Job_Placement_Dataset.csv")

df = pd.DataFrame(job_pred)

# ============================================================
# LOAD CLEANED DATASET
# ============================================================


job_pred_clean = pd.read_csv("C:/Users/V MANINDRA KUMAR/Desktop/JOB PLACEMENT CLEANED.csv")

df = pd.DataFrame(job_pred_clean)

print("\n================ DATASET LOADED ================\n")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA UNDERSTANDING
# ============================================================

print("\n================ DATA TYPES ================\n")
print(df.dtypes)

print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ STATISTICAL SUMMARY ================\n")
print(df.describe(include="all").T)

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

print("\n================ DUPLICATES ================\n")
print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove leading/trailing spaces from categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].astype("string").str.strip()


# Standardize inconsistent categorical values

# Gender
df["gender"] = df["gender"].replace({
    "male": "Male",
    "female": "Female"
})

# Internship
df["internship_experience"] = df["internship_experience"].replace({
    "yes": "Yes",
    "no": "No"
})


# Remove duplicate rows
before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicates removed:", before_duplicates - after_duplicates)


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())


categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:

    if df[col].isnull().sum() > 0:

        mode_value = df[col].mode()[0]

        df[col] = df[col].fillna(mode_value)


print("\n================ MISSING VALUES AFTER CLEANING ================\n")
print(df.isnull().sum())


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Experience Category
# ------------------------------------------------------------

def experience_category(years):

    if years == 0:
        return "Fresher"

    elif years <= 2:
        return "Junior"

    else:
        return "Senior"


df["experience_category"] = df["years_of_experience"].apply(
    experience_category
)


# ------------------------------------------------------------
# Academic Performance Band
# ------------------------------------------------------------

df["academic_average"] = (
    df["ssc_percentage"]
    + df["hsc_percentage"]
    + df["degree_percentage"]
) / 3


def academic_band(score):

    if score < 60:
        return "Low"

    elif score < 75:
        return "Medium"

    else:
        return "High"


df["academic_performance_band"] = df[
    "academic_average"
].apply(academic_band)


# ------------------------------------------------------------
# Skills Match Level
# ------------------------------------------------------------

def skills_level(score):

    if score < 60:
        return "Low"

    elif score < 80:
        return "Medium"

    else:
        return "High"


df["skills_match_level"] = df[
    "skills_match_percentage"
].apply(skills_level)


# ------------------------------------------------------------
# Interview Performance Category
# ------------------------------------------------------------

df["interview_average"] = (
    df["technical_score"]
    + df["aptitude_score"]
    + df["communication_score"]
) / 3


def interview_category(score):

    if score < 60:
        return "Low"

    elif score < 80:
        return "Medium"

    else:
        return "High"


df["interview_performance_category"] = df[
    "interview_average"
].apply(interview_category)


# ============================================================
# 6. TARGET VARIABLE
# ============================================================

# Project document defines:
# Placed     -> Job Accepted (1)
# Not Placed -> Job Rejected (0)

df["target"] = df["status"].map({
    "Placed": 1,
    "Not Placed": 0
})


print("\n================ TARGET DISTRIBUTION ================\n")
print(df["status"].value_counts())

print("\nTarget percentages:")
print(df["status"].value_counts(normalize=True) * 100)


# ============================================================
# 7. EDA
# ============================================================

os.makedirs("plots", exist_ok=True)


# ------------------------------------------------------------
# Status Distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="status")

plt.title("Job Placement Status Distribution")
plt.xlabel("Status")
plt.ylabel("Number of Candidates")

plt.tight_layout()
plt.savefig("plots/status_distribution.png")
plt.show()


# ------------------------------------------------------------
# Interview Score vs Placement
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="status",
    y="interview_average"
)

plt.title("Interview Score vs Job Acceptance")
plt.xlabel("Placement Status")
plt.ylabel("Average Interview Score")

plt.tight_layout()
plt.savefig("plots/interview_vs_acceptance.png")
plt.show()


# ------------------------------------------------------------
# Skills Match vs Placement
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="status",
    y="skills_match_percentage"
)

plt.title("Skills Match Percentage vs Job Acceptance")
plt.xlabel("Placement Status")
plt.ylabel("Skills Match Percentage")

plt.tight_layout()
plt.savefig("plots/skills_vs_acceptance.png")
plt.show()


# ------------------------------------------------------------
# Company Tier vs Acceptance
# ------------------------------------------------------------

company_acceptance = pd.crosstab(
    df["company_tier"],
    df["status"],
    normalize="index"
) * 100

print("\nCompany Tier vs Placement:")
print(company_acceptance)


company_acceptance.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Placement Rate by Company Tier")
plt.xlabel("Company Tier")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("plots/company_tier_acceptance.png")
plt.show()


# ------------------------------------------------------------
# Experience vs Placement
# ------------------------------------------------------------

experience_acceptance = pd.crosstab(
    df["experience_category"],
    df["status"],
    normalize="index"
) * 100

print("\nExperience vs Placement:")
print(experience_acceptance)


experience_acceptance.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Placement Rate by Experience Category")
plt.xlabel("Experience Category")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("plots/experience_acceptance.png")
plt.show()


# ------------------------------------------------------------
# Competition Level vs Placement
# ------------------------------------------------------------

competition_acceptance = pd.crosstab(
    df["competition_level"],
    df["status"],
    normalize="index"
) * 100

print("\nCompetition Level vs Placement:")
print(competition_acceptance)


competition_acceptance.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Placement Rate by Competition Level")
plt.xlabel("Competition Level")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("plots/competition_acceptance.png")
plt.show()


# ------------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14, 10))

sns.heatmap(
    numeric_df.corr(),
    annot=False,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.show()


# ============================================================
# 8. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

# Do NOT use:
# status -> original target
# target -> encoded target
#
# Also exclude derived categorical columns for the first model
# because their purpose is mainly analytical/EDA.

drop_columns = [
    "status",
    "target"
]

X = df.drop(columns=drop_columns)

y = df["target"]


# ============================================================
# 9. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\n================ NUMERIC FEATURES ================\n")
print(numeric_features)

print("\n================ CATEGORICAL FEATURES ================\n")
print(categorical_features)


# ============================================================
# 10. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 11. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ============================================================
# 12. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = logistic_model.predict(X_test)

logistic_probability = logistic_model.predict_proba(
    X_test
)[:, 1]


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_probability
)


print("\n================ LOGISTIC REGRESSION ================\n")

print(
    "Accuracy:",
    round(logistic_accuracy, 4)
)

print(
    "ROC-AUC:",
    round(logistic_auc, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


# ============================================================
# 13. RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


rf_predictions = random_forest_model.predict(
    X_test
)

rf_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)


print("\n================ RANDOM FOREST ================\n")

print(
    "Accuracy:",
    round(rf_accuracy, 4)
)

print(
    "ROC-AUC:",
    round(rf_auc, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Job Rejected",
        "Job Accepted"
    ]
)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix.png"
)

plt.show()


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

model_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],

    "ROC_AUC": [
        logistic_auc,
        rf_auc
    ]

})


print("\n================ MODEL COMPARISON ================\n")

print(model_results)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

rf_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)

rf_model = (
    random_forest_model
    .named_steps["model"]
)


feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)


feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": rf_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n================ TOP FEATURES ================\n")

print(
    feature_importance.head(20)
)


plt.figure(figsize=(10, 7))

top_features = feature_importance.head(15)

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Features Influencing Job Placement Prediction"
)

plt.tight_layout()

plt.savefig(
    "plots/feature_importance.png"
)

plt.show()


# ============================================================
# 17. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "cleaned_job_placement_dataset.csv",
    index=False
)


# ============================================================
# 18. SAVE MACHINE LEARNING MODEL
# ============================================================

joblib.dump(
    random_forest_model,
    "job_acceptance_model.pkl"
)


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

model_results.to_csv(
    "model_results.csv",
    index=False
)


feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)


print("\n================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================================")

print("\nGenerated files:")

print("1. cleaned_job_placement_dataset.csv")
print("2. job_acceptance_model.pkl")
print("3. model_results.csv")
print("4. feature_importance.csv")
print("5. plots/ folder")