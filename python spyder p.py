
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


BASE = Path(__file__).resolve().parent

DATA_FILE = Path(
    r"C:\Users\kp\Desktop\StudentsPerformance.csv"
)

GRAPH_DIR = BASE / "graphs"
RESULT_DIR = BASE / "results"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Original dataset shape:", df.shape)
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

if len(df) < 1000:
    raise ValueError("Dataset must contain at least 1000 rows.")

required_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )


df.columns = df.columns.str.strip()

duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()


text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].replace("", np.nan)
    df[col] = df[col].fillna("unknown")

score_columns = [
    "math score",
    "reading score",
    "writing score"
]

for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[~df[col].between(0, 100), col] = np.nan
    df[col] = df[col].fillna(df[col].median())

for col in text_columns:
    df[col] = df[col].str.title()

print("\n========== DATA CLEANING ==========")
print("Duplicate rows removed:", duplicates)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)

df.to_csv(
    RESULT_DIR / "cleaned_student_data.csv",
    index=False
)


df["average score"] = df[score_columns].mean(axis=1)
df["total score"] = df[score_columns].sum(axis=1)

df["performance category"] = pd.cut(
    df["average score"],
    bins=[-1, 39, 59, 79, 100],
    labels=["Needs Improvement", "Average", "Good", "Excellent"]
)

df.to_csv(
    RESULT_DIR / "student_data_with_features.csv",
    index=False
)

print("\n========== FEATURE ENGINEERING ==========")
print("Average score:", round(df["average score"].mean(), 2))
print("\nPerformance category counts:")
print(df["performance category"].value_counts())


plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="average score",
    bins=20,
    kde=True
)
plt.title("Distribution of Average Student Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "01_score_distribution.png")
plt.show()


gender_scores = df.groupby(
    "gender", observed=True
)["average score"].mean().sort_values(ascending=False)

print("\nObjective 2: Average score by gender")
print(gender_scores.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(
    data=df,
    x="gender",
    y="average score",
    errorbar=None
)
plt.title("Average Score by Gender")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "02_gender_performance.png")
plt.show()


parent_scores = df.groupby(
    "parental level of education",
    observed=True
)["average score"].mean().sort_values(ascending=False)

print("\nObjective 3: Average score by parental education")
print(parent_scores.round(2))

plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="parental level of education",
    y="average score",
    errorbar=None
)
plt.title("Average Score by Parental Education")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "03_parental_education.png")
plt.show()


prep_scores = df.groupby(
    "test preparation course",
    observed=True
)["average score"].mean().sort_values(ascending=False)

print("\nObjective 4: Average score by test preparation")
print(prep_scores.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(
    data=df,
    x="test preparation course",
    y="average score",
    errorbar=None
)
plt.title("Average Score by Test Preparation")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "04_test_preparation.png")
plt.show()


lunch_scores = df.groupby(
    "lunch", observed=True
)["average score"].mean().sort_values(ascending=False)

print("\nObjective 5: Average score by lunch type")
print(lunch_scores.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(
    data=df,
    x="lunch",
    y="average score",
    errorbar=None
)
plt.title("Average Score by Lunch Type")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "05_lunch_comparison.png")
plt.show()


plt.figure(figsize=(7, 5))
sns.heatmap(
    df[score_columns].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Between Exam Scores")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "06_score_correlation.png")
plt.show()


feature_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

X = df[feature_columns]
y = df["average score"]

categorical_features = feature_columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []
trained_models = {}

print("\n========== MACHINE LEARNING RESULTS ==========")

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("\nModel:", name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON TABLE ==========")
print(results_df.round(3).to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nBest-fit model based on lowest RMSE:", best_name)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE",
    errorbar=None
)
plt.title("Comparison of Machine Learning Models")
plt.ylabel("RMSE (Lower Is Better)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "07_model_comparison.png")
plt.show()


best_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({
    "Actual Average Score": y_test.to_numpy(),
    "Predicted Average Score": best_predictions
})

prediction_results.to_csv(
    RESULT_DIR / "best_model_predictions.csv",
    index=False
)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=y_test.to_numpy(),
    y=best_predictions
)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Actual vs Predicted Scores: " + best_name)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "08_actual_vs_predicted.png")
plt.show()


with open(
    RESULT_DIR / "project_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("STUDENT PERFORMANCE PREDICTION PROJECT\n\n")
    file.write(f"Rows after cleaning: {len(df)}\n")
    file.write(f"Duplicate rows removed: {duplicates}\n")
    file.write(
        f"Overall mean average score: "
        f"{df['average score'].mean():.2f}\n\n"
    )

    file.write("Average score by gender:\n")
    file.write(gender_scores.round(2).to_string())
    file.write("\n\nAverage score by parental education:\n")
    file.write(parent_scores.round(2).to_string())
    file.write("\n\nAverage score by test preparation:\n")
    file.write(prep_scores.round(2).to_string())
    file.write("\n\nAverage score by lunch type:\n")
    file.write(lunch_scores.round(2).to_string())

    file.write("\n\nMODEL COMPARISON:\n")
    file.write(results_df.round(3).to_string(index=False))

    file.write("\n\nCONCLUSION:\n")
    file.write(
        f"The model with the lowest test RMSE was {best_name}. "
        "Performance was evaluated using MAE, RMSE and R-squared. "
        "The predictions use student background variables only. "
        "The findings show associations, not proof of causation. "
        "Model performance should be interpreted using the measured "
        "test results and the limitations of the dataset."
    )