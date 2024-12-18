# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Matplotlib backend fix for Windows
import matplotlib
matplotlib.use("Agg")

# Initialize Streamlit app
def main():
    st.title("Statistical Application")

    # Sidebar for navigation
    menu = ["Data Upload", "Data Processing", "Descriptive Statistics", "Inferential Statistics", "Regression Analysis", "Clustering", "Advanced Features"]
    choice = st.sidebar.selectbox("Select an Option", menu)

    if choice == "Data Upload":
        data_upload()
    elif choice == "Data Processing":
        data_processing()
    elif choice == "Descriptive Statistics":
        descriptive_statistics()
    elif choice == "Inferential Statistics":
        inferential_statistics()
    elif choice == "Regression Analysis":
        regression_analysis()
    elif choice == "Clustering":
        clustering_visualizations()
    elif choice == "Advanced Features":
        advanced_features()

# Data Upload Function
def data_upload():
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.session_state['data'] = data
            st.dataframe(data.head())
            st.success("Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

# Data Processing Function
def data_processing():
    st.header("Data Processing and Wrangling")

    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Initial Dataset")
        st.dataframe(data.head())

        # Handle missing values
        if st.checkbox("Remove Missing Values"):
            data = data.dropna()
            st.success("Missing values removed.")

        # Rename columns
        if st.checkbox("Rename Columns"):
            col_mapping = {}
            for col in data.columns:
                new_name = st.text_input(f"Rename column '{col}'", value=col)
                if new_name != col:
                    col_mapping[col] = new_name
            if col_mapping:
                data.rename(columns=col_mapping, inplace=True)
                st.success("Columns renamed.")

        # Filter rows
        if st.checkbox("Filter Rows"):
            filter_col = st.selectbox("Select Column to Filter", data.columns)
            filter_val = st.text_input("Value to Filter by")
            if filter_val:
                data = data[data[filter_col].astype(str).str.contains(filter_val)]
                st.success("Rows filtered.")

        st.session_state['data'] = data
        st.write("Processed Dataset")
        st.dataframe(data.head())
    else:
        st.error("Please upload a dataset first.")

# Descriptive Statistics Function
def descriptive_statistics():
    st.header("Descriptive Statistics")

    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Summary Statistics")
        st.write(data.describe())

        st.write("Data Distribution")
        numeric_columns = data.select_dtypes(include=np.number).columns
        if not numeric_columns.any():
            st.warning("No numeric columns found for distribution.")
        else:
            for column in numeric_columns:
                try:
                    fig, ax = plt.subplots()
                    sns.histplot(data[column], kde=True, ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Failed to plot {column}: {e}")
    else:
        st.error("Please upload a dataset first.")

# Inferential Statistics Function
def inferential_statistics():
    st.header("Inferential Statistics")

    if 'data' in st.session_state:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=np.number).columns
        if len(numeric_columns) < 2:
            st.error("Dataset needs at least two numeric columns for analysis.")
        else:
            test_type = st.selectbox("Select Test", ["T-test", "Chi-Square Test", "ANOVA"])

            if test_type == "T-test":
                col1 = st.selectbox("Select First Column", numeric_columns, key="t1")
                col2 = st.selectbox("Select Second Column", numeric_columns, key="t2")
                if st.button("Run T-test"):
                    t_stat, p_value = ttest_ind(data[col1], data[col2])
                    st.write(f"T-statistic: {t_stat}, P-value: {p_value}")

            elif test_type == "Chi-Square Test":
                cat_col1 = st.selectbox("Select First Categorical Column", data.columns, key="chi1")
                cat_col2 = st.selectbox("Select Second Categorical Column", data.columns, key="chi2")
                if st.button("Run Chi-Square Test"):
                    contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
                    chi2, p, dof, _ = chi2_contingency(contingency_table)
                    st.write(f"Chi-Square: {chi2}, P-value: {p}, Degrees of Freedom: {dof}")

            elif test_type == "ANOVA":
                selected_columns = st.multiselect("Select Columns", numeric_columns, default=numeric_columns[:3], key="anova")
                if st.button("Run ANOVA") and len(selected_columns) >= 2:
                    f_stat, p_value = f_oneway(*(data[col] for col in selected_columns))
                    st.write(f"F-statistic: {f_stat}, P-value: {p_value}")
    else:
        st.error("Please upload a dataset first.")

# Regression Analysis Function
def regression_analysis():
    st.header("Regression Analysis")

    if 'data' in st.session_state:
        data = st.session_state['data']
        columns = data.select_dtypes(include=np.number).columns.tolist()

        if len(columns) < 2:
            st.error("Dataset needs at least two numeric columns for regression.")
        else:
            x_column = st.selectbox("Select Independent Variable", columns)
            y_column = st.selectbox("Select Dependent Variable", columns)

            if st.button("Run Linear Regression"):
                try:
                    valid_data = data[[x_column, y_column]].dropna()
                    if valid_data.empty:
                        st.error("No valid data available after removing missing values.")
                        return
                    X = valid_data[[x_column]]
                    y = valid_data[y_column]
                    model = LinearRegression()
                    model.fit(X, y)
                    st.write(f"Intercept: {model.intercept_:.2f}")
                    st.write(f"Coefficient: {model.coef_[0]:.2f}")
                    fig, ax = plt.subplots()
                    ax.scatter(X, y, color='blue')
                    ax.plot(X, model.predict(X), color='red')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a dataset first.")

# Clustering Visualizations Function
def clustering_visualizations():
    st.header("Clustering Visualizations")

    if 'data' in st.session_state:
        data = st.session_state['data']
        columns = data.select_dtypes(include=np.number).columns.tolist()

        if len(columns) < 2:
            st.error("Dataset needs at least two numeric columns for clustering.")
        else:
            x_column = st.selectbox("Select X-Axis Variable", columns)
            y_column = st.selectbox("Select Y-Axis Variable", columns)
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)

            if st.button("Run K-Means Clustering"):
                try:
                    X = data[[x_column, y_column]].dropna()
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                    data['Cluster'] = kmeans.fit_predict(X)
                    fig = px.scatter(data, x=x_column, y=y_column, color=data['Cluster'].astype(str))
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a dataset first.")

# Advanced Features Function
def advanced_features():
    st.header("Advanced Features")

    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Principal Component Analysis (PCA)")
        numeric_columns = data.select_dtypes(include=np.number).columns
        selected_columns = st.multiselect("Select Columns for PCA", numeric_columns)
        if len(selected_columns) >= 2 and st.button("Run PCA"):
            pca = PCA(n_components=2)
            components = pca.fit_transform(data[selected_columns].dropna())
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            fig = px.scatter(pca_df, x='PC1', y='PC2')
            st.plotly_chart(fig)

        st.write("Time-Series Analysis")
        date_column = st.selectbox("Select Date Column", data.columns)
        value_column = st.selectbox("Select Value Column", numeric_columns)
        if date_column and value_column and st.button("Run Time-Series Analysis"):
            ts_data = data[[date_column, value_column]].dropna()
            ts_data[date_column] = pd.to_datetime(ts_data[date_column])
            ts_data.set_index(date_column, inplace=True)
            fig, ax = plt.subplots()
            ts_data[value_column].plot(ax=ax)
            st.pyplot(fig)
    else:
        st.error("Please upload a dataset first.")

if __name__ == '__main__':
    main()
