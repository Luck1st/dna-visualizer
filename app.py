!pip install streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Streamlit app title
st.title("Genetic Data Analysis App")

# Load data (adjust path if necessary)
df_train = None # Initialize variables to None
df_test = None

try:
    df_train = pd.read_csv('genetic_data_train.csv')
    df_test = pd.read_csv('genetic_data_test.csv')
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("Error: One or both of the CSV files were not found. Please make sure 'genetic_data_train.csv' and 'genetic_data_test.csv' are in the same directory.")
    st.stop() # Stop the app if data loading fails
except pd.errors.ParserError:
    st.error("Error: There was an issue parsing one or both of the CSV files. Please check the CSV file format.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading: {e}")
    st.stop()

# Only proceed if data loading was successful (st.stop() was not called)
if df_train is not None and df_test is not None:
    # Display data head
    st.header("Data Head")
    st.subheader("Training Data Head")
    st.dataframe(df_train.head())
    st.subheader("Test Data Head")
    st.dataframe(df_test.head())

    # Data Exploration
    st.header("Data Exploration")

    st.subheader("Shape of DataFrames")
    st.write("Shape of df_train:", df_train.shape)
    st.write("Shape of df_test:", df_test.shape)

    st.subheader("Data Types")
    st.write("Data types of df_train:")
    st.write(df_train.dtypes)
    st.write("Data types of df_test:")
    st.write(df_test.dtypes)

    st.subheader("Missing Values")
    st.write("Missing values in df_train:")
    st.write(df_train.isnull().sum())
    st.write("Missing values in df_test:")
    st.write(df_test.isnull().sum())

    st.subheader("Distribution of 'Ancestry' in Training Data")
    st.write(df_train['Ancestry'].value_counts())
    fig_ancestry, ax_ancestry = plt.subplots(figsize=(10, 6))
    df_train['Ancestry'].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen', 'gold'], ax=ax_ancestry)
    ax_ancestry.set_title('Distribution of Ancestry in Training Data')
    ax_ancestry.set_xlabel('Ancestry')
    ax_ancestry.set_ylabel('Frequency')
    st.pyplot(fig_ancestry)

    # Data Analysis
    st.header("Data Analysis")

    st.subheader("Descriptive Statistics")
    st.write("Descriptive statistics for df_train:")
    st.dataframe(df_train.describe(exclude=['object']))
    st.write("Descriptive statistics for df_test:")
    st.dataframe(df_test.describe(exclude=['object']))

    st.subheader("Correlation Matrix of Numerical Features (df_train)")
    numerical_features = df_train.select_dtypes(include=np.number)
    correlation_matrix = numerical_features.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Numerical Features (df_train)')
    st.pyplot(fig_corr)

    st.subheader("Correlation between Numerical Features and 'Ancestry'")
    try:
        ancestry_correlation = df_train.groupby('Ancestry').agg(['mean', 'std'])
        st.dataframe(ancestry_correlation)
    except Exception as e:
        st.warning(f"Could not compute group-wise statistics: {e}")

    # Data Visualization
    st.header("Data Visualization")

    st.subheader("Distribution of Numerical Features")
    numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    if numerical_cols:
        for col in numerical_cols:
            st.subheader(f"Distribution of {col}")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            sns.histplot(df_train[col], kde=True, color='skyblue', label='Train', ax=ax_dist)
            sns.histplot(df_test[col], kde=True, color='salmon', label='Test', alpha=0.7, ax=ax_dist)
            ax_dist.set_title(f'Distribution of {col}')
            ax_dist.legend()
            st.pyplot(fig_dist)
            plt.close(fig_dist) # Close figure to prevent resource warnings

    st.subheader("Box Plots of Numerical Features")
    if numerical_cols:
        for col in numerical_cols:
            st.subheader(f"{col} Box Plot")
            fig_boxplot, ax_boxplot = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df_train, y=col, color='skyblue', showfliers=True, ax=ax_boxplot)
            ax_boxplot.set_title(f'{col} Box Plot (Train)')
            st.pyplot(fig_boxplot)
            plt.close(fig_boxplot) # Close figure

            fig_boxplot_test, ax_boxplot_test = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df_test, y=col, color='salmon', showfliers=True, ax=ax_boxplot_test)
            ax_boxplot_test.set_title(f'{col} Box Plot (Test)')
            st.pyplot(fig_boxplot_test)
            plt.close(fig_boxplot_test) # Close figure


    st.subheader("Scatter Plots of Numerical Features vs. 'Ancestry'")
    if numerical_cols:
        for col in numerical_cols:
            st.subheader(f"{col} vs. Ancestry")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_train, x=col, y='Ancestry', hue='Ancestry', palette='viridis', ax=ax_scatter)
            ax_scatter.set_title(f'{col} vs. Ancestry')
            st.pyplot(fig_scatter)
            plt.close(fig_scatter) # Close figure