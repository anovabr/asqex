import streamlit as st
import pandas as pd
import numpy as np
import time  # For progress bar
import plotly.graph_objs as go
import plotly.express as px

# Set up the title
st.title("ASQ:EX Data Analysis Tool - Beta Version")

# Add a new session state variable to track current view
if "current_view" not in st.session_state:
    st.session_state.current_view = "welcome"

# Display welcome message at startup
if "welcome_displayed" not in st.session_state:
    st.session_state.welcome_displayed = False

if not st.session_state.welcome_displayed:
    st.markdown(
        "### Welcome to the ASQ:EX Data Analysis Tool (Beta Version) 🚀\n"
        "This tool helps with data processing and analysis.\n\n"
        "🔹 **Upload your dataset** using the sidebar.\n"
        "🔹 **Click the buttons on the left sidebar** to navigate through different sections."
    )
    st.session_state.welcome_displayed = True

# Function to set current view
def set_view(view):
    st.session_state.current_view = view

# Sidebar: Upload Excel File First
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# Sidebar: Navigation Buttons
st.sidebar.header("Options")
data_info_btn = st.sidebar.button("Data Information", on_click=set_view, args=("data_info",))
data_processing_btn = st.sidebar.button("Processing", on_click=set_view, args=("processing",))
descriptive_btn = st.sidebar.button("Descriptives", on_click=set_view, args=("descriptives",))
cutoffs_btn = st.sidebar.button("Cutoffs", on_click=set_view, args=("cutoffs",))

# Initialize session state to store cleaned data
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# If a file is uploaded
if uploaded_file:
    # Read the Excel file (all sheets)
    df = pd.read_excel(uploaded_file, sheet_name=None)

    # Let the user select a sheet if multiple exist
    sheet_name = st.selectbox("Select a sheet", list(df.keys()))
    df = df[sheet_name]  # Load selected sheet into a DataFrame

    # View handling
    if st.session_state.current_view == "data_info":
        st.subheader("Dataset Information")
        st.write(f"✅ **Rows:** {df.shape[0]}")
        st.write(f"✅ **Columns:** {df.shape[1]}")
        st.subheader("Column Names")
        st.write(df.columns.tolist())

    # Processing view
    elif st.session_state.current_view == "processing":
        st.subheader("Processing Data...")

        # Show cleaning explanation
        st.info(
            "**The data will be cleaned as follows:**\n"
            "- **Fully empty rows and columns will be removed**.\n"
            "- **All dots (.)** will be treated as **missing values**.\n"
            "- **Missing values before the first number** will be replaced with **2**.\n"
            "- **Missing values after the last number** will be replaced with **0**.\n"
            "- **Missing values between numbers** will be replaced with **2**.\n"
            "- **A 'total' variable will be created** by summing numeric columns that contain `\"` (quotes) in their names."
        )

        # Function to clean data with a progress bar
        def clean_data(df):
            progress = st.progress(0)  # Initialize progress bar

            # Step 1: Remove fully empty rows and columns
            df.dropna(how="all", axis=0, inplace=True)  # Remove empty rows
            df.dropna(how="all", axis=1, inplace=True)  # Remove empty columns
            progress.progress(20)  # Update progress

            # Step 2: Convert all column names to lowercase
            df.columns = df.columns.str.lower()
            progress.progress(40)  # Update progress

            # Step 3: Convert all dots (".") to NaN
            df.replace(".", np.nan, inplace=True)
            progress.progress(60)  # Update progress

            # Step 4: Apply missing value replacement rules row-wise
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

            progress.progress(80)  # Update progress

            # Step 5: Compute Total Score (Sum Numeric Columns with Quotes in Their Names)
            numeric_columns_with_quotes = [col for col in df.select_dtypes(include=["number"]).columns if '"' in col]
            if numeric_columns_with_quotes:
                df["total"] = df[numeric_columns_with_quotes].sum(axis=1)
            else:
                df["total"] = np.nan  # If no matching columns found, assign NaN

            progress.progress(100)  # Processing complete
            time.sleep(0.5)  # Small delay before removing the progress bar

            return df

        # Clean the data and store it in session state
        st.session_state.cleaned_df = clean_data(df.copy())

        # Display cleaned data preview
        st.subheader("Cleaned Data Preview")
        st.write(st.session_state.cleaned_df.head())  # Show first few rows

        # Provide option to download cleaned data
        cleaned_file = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Cleaned Data", data=cleaned_file, file_name="cleaned_data.csv", mime="text/csv")

    # Descriptives view
    elif st.session_state.current_view == "descriptives":
        # Ensure cleaned data exists
        if st.session_state.get('cleaned_df') is not None:
            st.subheader("Descriptive Analysis")

            # Identify non-numeric and numeric columns
            cleaned_df = st.session_state.cleaned_df
            non_numeric_columns = cleaned_df.select_dtypes(exclude=["number"]).columns.tolist()
            numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

            # Create columns for dropdowns
            col1, col2 = st.columns(2)

            with col1:
                # Unique key for non-numeric selector to prevent resets
                selected_non_numeric = st.selectbox(
                    "Select a non-numeric (grouping) variable:", 
                    ["None"] + non_numeric_columns,
                    key="non_numeric_group_select"
                )

            with col2:
                # Unique key for numeric selector to prevent resets
                selected_numeric = st.selectbox(
                    "Select a numeric variable:", 
                    ["None"] + numeric_columns,
                    key="numeric_variable_select"
                )

            # Descriptive statistics container
            st.subheader("Descriptive Statistics")
            
            # Check if a numeric variable is selected
            if selected_numeric != "None":
                try:
                    # Two different computation paths
                    if selected_non_numeric == "None":
                        # Simple descriptive stats for a single numeric column
                        descriptive_table = cleaned_df[selected_numeric].describe().to_frame()
                        descriptive_table.columns = [selected_numeric]
                        st.dataframe(descriptive_table)
                    else:
                        # Grouped descriptive stats
                        descriptive_table = cleaned_df.groupby(selected_non_numeric)[selected_numeric].describe()
                        st.dataframe(descriptive_table)
                
                except Exception as e:
                    st.error(f"Error generating descriptive statistics: {str(e)}")
            else:
                st.warning("Please select a numeric variable for analysis.")

    # Cutoffs view
    elif st.session_state.current_view == "cutoffs":
        # Ensure cleaned data exists
        if st.session_state.get('cleaned_df') is not None:
            st.subheader("Cutoff Analysis")

            # Prepare the data
            cleaned_df = st.session_state.cleaned_df

            # Identify non-numeric columns
            non_numeric_columns = cleaned_df.select_dtypes(exclude=["number"]).columns.tolist()

            # Create columns for dropdowns
            col1, col2 = st.columns(2)

            with col1:
                # Grouping variable selector
                selected_non_numeric = st.selectbox(
                    "Select a non-numeric (grouping) variable:", 
                    ["None"] + non_numeric_columns,
                    key="cutoff_non_numeric_select"
                )

            with col2:
                # Variable selector (currently only 'total')
                selected_numeric = st.selectbox(
                    "Select a numeric variable:", 
                    ["total"],  # Currently only 'total'
                    key="cutoff_numeric_select"
                )

            # Compute statistics
            if selected_numeric:
                # Prepare the dataframe for analysis
                if selected_non_numeric == "None":
                    # No grouping
                    analysis_df = cleaned_df
                else:
                    # Group by the selected non-numeric variable
                    analysis_df = cleaned_df

                # Compute statistics
                total_var = analysis_df[selected_numeric]
                
                # Descriptive statistics
                mean = total_var.mean()
                std = total_var.std()
                
                # Percentiles
                percentiles = {
                    '10%': total_var.quantile(0.1),
                    '25%': total_var.quantile(0.25),
                    '50%': total_var.quantile(0.5),
                    '75%': total_var.quantile(0.75),
                    '90%': total_var.quantile(0.9)
                }

                # Prepare statistics table
                stats_data = {
                    'Statistic': ['Mean', 'Standard Deviation'] + list(percentiles.keys()),
                    'Value': [mean, std] + list(percentiles.values())
                }
                stats_df = pd.DataFrame(stats_data)
                
                # Display statistics table
                st.subheader("Descriptive Statistics")
                st.dataframe(stats_df)

                # Compute observations below different cutoffs
                cutoffs_data = {
                    'Cutoff': [
                        'Below Mean - 2 SD', 
                        'Below Mean - 1 SD', 
                        'Below 10th Percentile', 
                        'Below 25th Percentile'
                    ],
                    'Number of Observations': [
                        sum(total_var < (mean - 2*std)),
                        sum(total_var < (mean - std)),
                        sum(total_var < percentiles['10%']),
                        sum(total_var < percentiles['25%'])
                    ]
                }
                cutoffs_df = pd.DataFrame(cutoffs_data)
                
                # Display cutoffs table
                st.subheader("Observations Below Cutoffs")
                st.dataframe(cutoffs_df)

                # Plotly Density Plot
                st.subheader("Density Plot")
                
                # Create density plot
                fig = go.Figure()
                
                # Add density trace
                fig.add_trace(go.Histogram(
                    x=total_var, 
                    name='Density',
                    histnorm='density',
                    opacity=0.7
                ))
                
                # Add vertical lines for key statistics
                fig.add_shape(
                    type='line',
                    x0=mean, x1=mean,
                    y0=0, y1=10,
                    line=dict(color='red', width=2, dash='dash'),
                    name='Mean'
                )
                
                # Add percentile lines
                percentile_colors = {
                    '10%': 'green',
                    '25%': 'blue',
                    '50%': 'purple',
                    '75%': 'orange',
                    '90%': 'brown'
                }
                
                for perc_name, perc_value in percentiles.items():
                    fig.add_shape(
                        type='line',
                        x0=perc_value, x1=perc_value,
                        y0=0, y1=10,
                        line=dict(color=percentile_colors[perc_name], width=2),
                        name=f'{perc_name} Percentile'
                    )
                
                # Customize layout
                fig.update_layout(
                    title='Density Distribution with Statistical Markers',
                    xaxis_title=selected_numeric,
                    yaxis_title='Density',
                    barmode='overlay'
                )
                
                # Display the plot
                st.plotly_chart(fig)

            else:
                st.warning("Please select a variable for analysis.")

else:
    st.warning("Please upload an Excel file to proceed.")