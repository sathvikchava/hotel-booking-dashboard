import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Hotel Booking Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data3.csv')
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    return df

df = load_data()

# --- Feature Engineering ---
if 'age_group' not in df.columns:
    df['age_group'] = pd.cut(df['customer_age'], bins=[0, 25, 35, 50, 65, 100],
                             labels=['<25', '25–35', '35–50', '50–65', '65+'])

if 'stay_group' not in df.columns:
    df['stay_group'] = pd.cut(df['stay_nights'], bins=[0, 2, 5, 10, 30],
                              labels=['1–2 nights', '3–5 nights', '6–10 nights', '10+ nights'])

if 'discount_level' not in df.columns:
    df['discount_level'] = pd.cut(df['discount_pct'], bins=[0, 10, 30, 100],
                                  labels=['Low', 'Medium', 'High'])

if 'comp_rate_level' not in df.columns:
    df['comp_rate_level'] = pd.qcut(df['competitor_rate'], q=3,
                                    labels=['Low', 'Medium', 'High'])

# ================================
# SECTION 1: Introduction
# ================================
st.title("Hotel Booking Analytics Dashboard")

st.markdown("""
### Problem Statement

This project focuses on improving profitability for a major hotel chain by shrinking the cancellations and no shows.The goal is to identify traits and key predictors of unreliable bookings and provide insights for making smarter decisions on pricing, marketing and operations. 
### Objectives
            
- Analyse the patterns in booking to identify the predictor variables for cancellations and no-shows.
- Understand key factors influencing the revenue.
- Provide actionable recomendations from the analysis.
""")

# ================================
# SECTION 2: Data Exploration
# ================================
st.header("Data Exploration")

# --- 2 Columns: Age Histogram + Heatmap ---
col1, col2 = st.columns(2)

# Histogram: Customer Age
with col1:
    st.subheader("Customer Age Distribution")
    fig_age, ax_age = plt.subplots(figsize=(6, 4))
    sns.histplot(df['customer_age'], bins=30, kde=True, color='skyblue', ax=ax_age)
    ax_age.set_title('Age Distribution', fontsize=12)
    ax_age.set_xlabel('Age')
    ax_age.set_ylabel('Count')
    st.pyplot(fig_age)

# --- Correlation Heatmap with Annotations ---
with col2:
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,              # <--- show values inside cells
        fmt=".2f",
        cmap='coolwarm',
        ax=ax_corr,
        square=True,
        cbar_kws={"shrink": 0.6},
        annot_kws={"size": 8}    # smaller font for compact display
    )
    ax_corr.set_title("Numerical Feature Correlations", fontsize=12)
    st.pyplot(fig_corr)


# --- Top Correlations Section ---
st.subheader("Top 5 Correlated Variables with Target Columns")

# Encode categorical outcome
df['target_class_encoded'] = df['target_class'].astype('category').cat.codes
numeric_df['target_class_encoded'] = df['target_class_encoded']

# Compute correlations
corr_with_class = numeric_df.corr()['target_class_encoded'].drop('target_class_encoded')
top_class_corr = corr_with_class.abs().sort_values(ascending=False).head(5).reset_index()
top_class_corr.columns = ['Variable', 'Correlation with target_class']

corr_with_value = numeric_df.corr()['target_value'].drop('target_value')
top_value_corr = corr_with_value.abs().sort_values(ascending=False).head(5).reset_index()
top_value_corr.columns = ['Variable', 'Correlation with target_value']

col3, col4 = st.columns(2)
with col3:
    st.write("### `target_class` Correlations")
    st.dataframe(top_class_corr, use_container_width=True)

with col4:
    st.write("### `target_value` Correlations")
    st.dataframe(top_value_corr, use_container_width=True)
# --- Two Columns: KDE Plot and Booking Outcome % Chart ---
col5, col6 = st.columns(2)

# --- KDE Plot: Lead Time vs Avg Daily Rate ---
with col5:
    st.subheader("Lead Time vs Avg Daily Rate")
    sample_df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
    joint = sns.jointplot(
        data=sample_df,
        x='lead_time_days',
        y='avg_daily_rate',
        hue='target_class',
        kind='kde',
        palette='Set2',
        fill=True
    )
    joint.fig.set_size_inches(6, 5)
    st.pyplot(joint.fig)

# --- Booking Outcome % by Variable ---
with col6:
    st.subheader("Booking Outcome % by Categorical Variable")
    categorical_cols = ['channel', 'market_segment', 'weekend_flag', 'age_group', 'discount_level']
    selected_col = st.selectbox("Select a variable:", categorical_cols, key='category_chart')

    normalized = pd.crosstab(df[selected_col], df['target_class'], normalize='index') * 100
    normalized = normalized.transpose()

    custom_colors = {'Cancelled': '#FFA726', 'Completed': '#66BB6A', 'NoShow': '#EF5350'}
    bar_colors = [custom_colors.get(outcome, "#999999") for outcome in normalized.index]

    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    x = np.arange(len(normalized.columns))
    bar_width = 0.2

    for i, outcome in enumerate(normalized.index):
        values = normalized.iloc[i]
        ax_bar.bar(x + i * bar_width, values, width=bar_width, label=outcome, color=bar_colors[i])
        for xi, val in zip(x, values):
            if val > 3:
                ax_bar.text(xi + i * bar_width, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax_bar.set_title(f'Booking Outcome % by {selected_col}', fontsize=13)
    ax_bar.set_xlabel(selected_col)
    ax_bar.set_ylabel('Percentage')
    ax_bar.set_xticks(x + bar_width)
    ax_bar.set_xticklabels(normalized.columns, rotation=45)
    ax_bar.legend(title='Outcome', loc='upper right')
    st.pyplot(fig_bar)

# ================================
# SECTION 3: Key Insights
# ================================
st.header("💡 Key Insights")

st.markdown("""
- Weekend booking are more prone to no-show or cancellations.
- High discounts leads to more cancellations.
- Online Travel Agency channel had the lowest completion rate , 19.4%.
- "target_value" is most correlated with "stay_nights" and "taget_class" is most correlated with "lead_time_days".
""")

# ================================
# SECTION 4: Recommendations
# ================================
st.header("Recommendations")

st.markdown("""
- Promote direct or corporate booking channel.
- Provide high discount to customers based on a loyalty.
- Consider rigid cancellations rules during weekends.
- Prioritize bookings with longer durations as they can lead to higher income.
""")

st.markdown("""
Submitted by Sathvik Chava, Sumanjali Banjara, Sravani Enuganti.
""")
