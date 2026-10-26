# Student Performance Prediction Using Machine Learning

## Project Overview

This project analyzes student performance data using Python, Data Analysis, and Machine Learning techniques. It explores how factors such as gender, parental education, lunch type, and test preparation are associated with students' mathematics, reading, and writing scores.

## Objectives

- Clean and preprocess the student dataset.
- Perform Exploratory Data Analysis (EDA).
- Visualize score distributions and performance patterns.
- Create additional features from student scores.
- Train and compare multiple Machine Learning models.
- Evaluate models using MAE, RMSE, and R².

## Dataset

**Source:** Kaggle – Students Performance in Exams
[https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

The dataset contains 1,000 student records and 8 original columns.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Load the dataset.
2. Clean and preprocess the data.
3. Perform exploratory data analysis.
4. Generate visualizations.
5. Perform feature engineering.
6. Train and compare Machine Learning models.
7. Evaluate model performance.
8. Save graphs and results.

## Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Key Findings

- The dataset's overall average score was approximately 67.77.
- Students who completed test preparation had a higher average score in this dataset.
- Average scores varied across parental education levels.
- Students with standard lunch had a higher average score in this dataset.
- Linear Regression achieved the lowest RMSE among the four models tested.

## Evaluation Metrics

The models were evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

## Project Structure

- `student_performance.py` – Python project code
- `StudentsPerformance.csv` – Original dataset
- `graphs/` – Generated visualizations
- `results/` – Cleaned data, model comparisons, predictions, and project summary

## How to Run

1. Install Python.
2. Install the required libraries:
   `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place the dataset in the location specified in the Python script.
4. Run the Python script.

## Conclusion

This project demonstrates how data analysis and Machine Learning can be used to explore student performance patterns and compare prediction models. The results also show that predicting student scores remains challenging with the available features.

## Author

Student Performance Prediction using Machine Learning – Academic Project
