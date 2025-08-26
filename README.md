# OCD Patient Dataset: Demographics & Clinical Data Analysis

## 📌 Project Overview
This project explores and models the **OCD Patient Dataset**, which contains demographic and clinical data of 1500 individuals diagnosed with Obsessive-Compulsive Disorder (OCD).

We perform:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Machine Learning Modeling (classification of medications)
- Insights & Visualizations

## 📂 Project Structure
```
OCD-Patient-Analysis/
│── data/                   # dataset (ocd_patient_dataset.csv)
│── notebooks/              # Jupyter notebooks for EDA & visualization
│── src/                    # source code (ML models, preprocessing)
│── README.md               # project documentation
│── requirements.txt        # dependencies
│── .gitignore              # ignored files
```

## ⚙️ Tools & Libraries
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- ML models: XGBoost, LightGBM, CatBoost, RandomForest, Logistic Regression
- Visualization: Plotly, Missingno

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd OCD-Patient-Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run EDA Notebook:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```
4. Train ML models:
   ```bash
   python src/modeling.py
   ```

## 📊 Dataset
The dataset includes:
- Demographics: Age, Gender, Ethnicity, Education, Marital Status
- Clinical Data: OCD duration, Y-BOCS scores, diagnoses, medications
- Mental health comorbidities: Depression, Anxiety

## 🔗 Reference
[Understanding OCD Dataset Analysis](https://github.com/TeniOT/Understanding-Obsessive-Compulsive-Disorder-OCD-Trends-A-Data-Analysis)
