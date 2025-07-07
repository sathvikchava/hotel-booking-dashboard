import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
 
# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data3.csv')
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    return df
 
df = load_data()
 
# --- Optional Binning ---
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
 
# --- Main Title ---
st.title("Hotel Booking Analytics Dashboard")

# ================================
# SECTION 1: Grouped Bar Chart
# ================================
st.subheader("Booking Outcome % by Categorical Variable")
 
# Variable selector
categorical_cols = [
    'channel', 'market_segment',
    'weekend_flag', 'age_group',
    'discount_level'
]
 
selected_col = st.selectbox("Select a categorical variable:", categorical_cols)
 
# Calculate % breakdown
normalized = pd.crosstab(df[selected_col], df['target_class'], normalize='index') * 100
normalized = normalized.transpose()
 
# Color map
custom_colors = {
    'Cancelled': '#FFA726',
    'Completed': '#66BB6A',
    'NoShow': '#EF5350'
}
bar_colors = [custom_colors.get(outcome, "#999999") for outcome in normalized.index]
 
# Plot
fig2, ax2 = plt.subplots(figsize=(12, 6))
x = np.arange(len(normalized.columns))
bar_width = 0.2
 
for i, outcome in enumerate(normalized.index):
    values = normalized.iloc[i]
    ax2.bar(x + i * bar_width, values, width=bar_width, label=outcome, color=bar_colors[i])
    for xi, val in zip(x, values):
        if val > 3:
            ax2.text(xi + i * bar_width, val + 1, f'{val:.1f}%',
                     ha='center', va='bottom', fontsize=9)
 
ax2.set_title(f'Booking Outcome % by {selected_col}', fontsize=16)
ax2.set_xlabel(selected_col, fontsize=12)
ax2.set_ylabel('Percentage', fontsize=12)
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(normalized.columns, rotation=45)
ax2.legend(title='Booking Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
 
st.pyplot(fig2)
 
# ================================
# SECTION 2: Correlation Heatmap
# ================================
st.subheader("Correlation Heatmap of Numerical Features")
 
# Compute correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
 
# Plot heatmap
fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax3)
ax3.set_title("Correlation Heatmap", fontsize=16)
st.pyplot(fig3)
 
st.subheader("Top 5 Correlated Variables with Target Columns")
 
# Encode target_class for correlation (e.g., Completed=0, Cancelled=1, NoShow=2)
df['target_class_encoded'] = df['target_class'].astype('category').cat.codes
 
# Combine relevant numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
numeric_df['target_class_encoded'] = df['target_class_encoded']
 
# Correlation with target_class
corr_with_class = numeric_df.corr()['target_class_encoded'].drop(['target_class_encoded'])
top_class_corr = corr_with_class.abs().sort_values(ascending=False).head(5).reset_index()
top_class_corr.columns = ['Variable', 'Correlation with target_class']
 
# Correlation with target_value
corr_with_value = numeric_df.corr()['target_value'].drop(['target_value'])
top_value_corr = corr_with_value.abs().sort_values(ascending=False).head(5).reset_index()
top_value_corr.columns = ['Variable', 'Correlation with target_value']
 
# Display
st.write("### Top 5 Correlations with `target_class`")
st.dataframe(top_class_corr)
 
st.write("### Top 5 Correlations with `target_value`")
st.dataframe(top_value_corr)

# ================================
# SECTION 3: KDE ointplot: Lead Time vs Avg Daily Rate 
# ================================

st.subheader("Lead Time vs Avg Daily Rate (KDE by Booking Outcome)")

# Sample to speed up rendering (optional)
sample_df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df

# Create Seaborn JointPlot
joint = sns.jointplot(
    data=sample_df,
    x='lead_time_days',
    y='avg_daily_rate',
    hue='target_class',
    kind='kde',
    palette='Set2',
    fill=True
)


# Show in Streamlit
st.pyplot(joint.fig)
# ================================
# SECTION 4:Histogram of Customer Age
# ================================
st.subheader("Customer Age Distribution")

fig_age, ax_age = plt.subplots(figsize=(10, 5))
sns.histplot(df['customer_age'], bins=30, kde=True, color='skyblue', ax=ax_age)
ax_age.set_title('Distribution of Customer Age')
ax_age.set_xlabel('Customer Age')
ax_age.set_ylabel('Frequency')

st.pyplot(fig_age)
