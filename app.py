import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="OCD Patient Dataset Explorer", layout="wide")

st.title("🧠 OCD Patient Dataset - Analysis & Insights")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/ocd_patient_dataset.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Sidebar filters
st.sidebar.header("Filter Data")
gender = st.sidebar.multiselect("Select Gender", df["Gender"].unique())
ethnicity = st.sidebar.multiselect("Select Ethnicity", df["Ethnicity"].unique())

filtered_df = df.copy()
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]
if ethnicity:
    filtered_df = filtered_df[filtered_df["Ethnicity"].isin(ethnicity)]

st.subheader("Filtered Data")
st.write(f"Rows: {filtered_df.shape[0]}")
st.dataframe(filtered_df)

# Plotting
st.subheader("📊 Visualizations")
fig, ax = plt.subplots()
sns.histplot(filtered_df["Age"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x="Gender", ax=ax2)
st.pyplot(fig2)

st.write("✅ More plots and ML results can be added here...")
