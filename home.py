import streamlit as st
import pandas as pd
import numpy as np
import time  # For progress bar

# Set up the title
st.title("ASQ:EX Data Analysis Tool - Beta Version")

# Display welcome message at startup
st.markdown(
    "### Welcome to the ASQ:EX Data Analysis Tool (Beta Version)\n"
    "Please use the **buttons in the left sidebar** to navigate through different sections of the tool."
)

# Sidebar: Upload Excel File First
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# Sidebar: Navigation Buttons
st.sidebar.header("Options")
data_info_btn = st.sidebar.button("Data Information")
data_processing_btn = st.sidebar.button("Processing")
descriptive_btn = st.sidebar.button("Descriptives")
cutoffs_btn = st.sidebar.button("Cutoffs")

# Initialize session state to store cleaned data
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# If a file is uploaded
if uploaded_file:
    # Read the Excel file (all sheets)
    df = pd.read_excel(uploaded_file, sheet_name=None)

    # Let the user select a sheet if multiple exist
    sheet_name = st.selectbox("Select a sheet", df.keys())
    df = df[sheet_name]  # Load selected sheet into a DataFrame

    # When "Data Information" is clicked -> Show dataset details
    if data_info_btn:
        st.subheader("Dataset Information")
        st.write(f"✅ **Rows:** {df.shape[0]}")
        st.write(f"✅ **Columns:** {df.shape[1]}")
        st.subheader("Column Names")
        st.write(df.columns.tolist())

    # If "Processing" button is clicked -> Process data
    if data_processing_btn:
        st.subheader("Processing Data...")

        # Function to clean data
        def clean_data(df):
            # Remove fully empty rows and columns
            df.dropna(how="all", axis=0, inplace=True)  # Remove empty rows
            df.dropna(how="all", axis=1, inplace=True)  # Remove empty columns
            df.columns = df.columns.str.lower()  # Convert all column names to lowercase
            df.replace(".", np.nan, inplace=True)  # Convert all dots (".") to NaN

            # Apply missing value replacement rules row-wise
            for i in range(len(df)):
                row = df.iloc[i, :].values  # Get row values as an array
                first_non_nan = np.where(~pd.isna(row))[0][0] if np.any(~pd.isna(row)) else None
                last_non_nan = np.where(~pd.isna(row))[0][-1] if np.any(~pd.isna(row)) else None

                if first_non_nan is not None:
                    row[:first_non_nan] = 2  # Replace all NaNs before first number with 2
                if last_non_nan is not None:
                    row[last_non_nan + 1:] = 0  # Replace all NaNs after last number with 0
                for j in range(first_non_nan + 1, last_non_nan):  # Iterate only between first & last number
                    if pd.isna(row[j]):
                        row[j] = 2  # Replace middle NaNs with 2
                df.iloc[i, :] = row

            # Compute Total Score (Sum Numeric Columns with Quotes in Their Names)
            numeric_columns_with_quotes = [col for col in df.select_dtypes(include=["number"]).columns if '"' in col]
            if numeric_columns_with_quotes:
                df["total"] = df[numeric_columns_with_quotes].sum(axis=1)
            else:
                df["total"] = np.nan  # If no matching columns found, assign NaN

            return df

        # Clean the data and store it in session state
        st.session_state.cleaned_df = clean_data(df.copy())

        # Display cleaned data preview
        st.subheader("Cleaned Data Preview")
        st.write(st.session_state.cleaned_df.head())  # Show first few rows

        # Provide option to download cleaned data
        cleaned_file = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Cleaned Data", data=cleaned_file, file_name="cleaned_data.csv", mime="text/csv")

    # If "Descriptive" button is clicked -> Show two list boxes and compute descriptive table
    if descriptive_btn:
        if st.session_state.cleaned_df is not None:
            st.subheader("Descriptive Analysis")

            # Identify non-numeric and numeric columns
            non_numeric_columns = st.session_state.cleaned_df.select_dtypes(exclude=["number"]).columns.tolist()
            numeric_columns = st.session_state.cleaned_df.select_dtypes(include=["number"]).columns.tolist()

            # List boxes for user selection
            selected_non_numeric = st.selectbox("Select a non-numeric (grouping) variable:", ["None"] + non_numeric_columns)
            selected_numeric = st.selectbox("Select a numeric variable:", ["None"] + numeric_columns)

            # Ensure the table is always present
            if selected_numeric != "None":
                if selected_non_numeric == "None":
                    # Compute statistics for the entire dataset
                    descriptive_table = st.session_state.cleaned_df[selected_numeric].describe().to_frame()
                else:
                    # Compute statistics grouped by the selected non-numeric variable
                    descriptive_table = st.session_state.cleaned_df.groupby(selected_non_numeric)[selected_numeric].describe()

                # Display descriptive statistics
                st.subheader("Descriptive Statistics")
                st.write(descriptive_table)
            else:
                # Placeholder message when no variables are selected
                st.subheader("Descriptive Statistics")
                st.write("⚠️ Please select variables to generate descriptive statistics.")

        else:
            st.warning("Please process the data first by clicking 'Processing'.")

    # If "Cutoffs" button is clicked -> Do Nothing (Placeholder)
    if cutoffs_btn:
        st.subheader("Cutoffs (Coming Soon)")
        st.write("🚧 This section will be implemented soon.")

else:
    st.warning("Please upload an Excel file to proceed.")
