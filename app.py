# app.py

# Import all necessary libraries for the Streamlit web application.
import streamlit as st
import pandas as pd
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
import os # Import the 'os' library to handle file paths.

# Import the core logic functions from the separate allocation_model.py file.
from allocation_model import get_allocations_from_model, run_allocation_model, calculate_smoothness

# ==================================================================================================
# SECTION 1: DATA LOADING AND INITIAL SETUP
# This section loads the data using a relative path, making it compatible with cloud deployment.
# The code assumes the data file is in a subdirectory named 'data'.
# ==================================================================================================
try:
    # Use a relative path to the data file.
    # The path has been corrected from 'raw data' to 'data' to match your file structure.
    data_path = os.path.join(os.path.dirname(__file__), 'raw data', 'Occupation_Major_Mapping_Singapore.xlsx')
    df_excel = pd.read_excel(data_path, sheet_name='Sheet2')
    st.success(f"Data loaded successfully from: {data_path}")
except Exception as e:
    st.error(f"Error loading data from the specified path. Please check the file path: {e}")
    # Stop the app execution if the data cannot be loaded.
    st.stop()


# ==================================================================================================
# SECTION 2: CORE ALLOCATION LOGIC FUNCTIONS
# This section contains the backend functions that perform the calculations and optimization.
# These functions are now imported from the allocation_model.py file.
# ==================================================================================================


# ==================================================================================================
# SECTION 3: STREAMLIT APP LAYOUT AND INTERFACE
# This section defines the web page's structure, user interaction elements, and displays results.
# ==================================================================================================

# Set up the Streamlit page configuration.
st.set_page_config(
    page_title="LP Student Allocation Model",
    page_icon="🎓",
    layout="wide"
)

# --- App Header ---
st.title("University Allocation Optimization Model")
st.write("This application uses a linear programming model to optimize student allocations based on various parameters.")
st.markdown("---")

st.header("1. Configure and Run the Model")

# --- Sidebar for User Inputs ---
st.sidebar.header("Model Parameters")

# Create sliders and number inputs for the user to define model parameters.
total_students = st.sidebar.number_input(
    "Total Students",
    min_value=1,
    value=7000,
    step=100
)

budget_in_millions = st.sidebar.number_input(
    "Max Budget (in millions)",
    min_value=1,
    value=500,
    step=10
)
max_budget = budget_in_millions * 1_000_000

# Let the user choose the target distribution model.
st.sidebar.subheader("Target Distribution Model")
distribution_options = ["Power Law", "Sigmoid", "Exponential Decay"]
selected_model_name = st.sidebar.radio(
    "Choose the distribution model:",
    options=distribution_options
)
# Convert the user's choice to an integer for use in the backend functions.
choice = distribution_options.index(selected_model_name) + 1

# Provide sliders for the model constants (alpha, k, decay_rate).
st.sidebar.markdown("---")
st.sidebar.subheader("Model Constants")
col1, col2 = st.sidebar.columns(2)
with col1:
    alpha = st.sidebar.slider("Power Law (alpha)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
with col2:
    k = st.sidebar.slider("Sigmoid (k)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
decay_rate = st.sidebar.slider("Exponential Decay (decay_rate)", min_value=0.0, max_value=0.1, value=0.01, step=0.001)

st.markdown("---")

# A button to trigger the model run.
if st.button("Run Optimization Model"):
    # Display a spinner while the model is solving.
    with st.spinner("Solving the linear programming problem..."):
        result_df, summary = run_allocation_model(
            df_excel=df_excel,
            total_students=total_students,
            choice=choice,
            alpha=alpha,
            k=k,
            decay_rate=decay_rate,
            max_budget=max_budget
        )

    # Show a success message and key summary metrics.
    st.success("Optimization complete! Status: " + summary["status"])
    st.markdown("---")

    st.write("### Allocation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students Allocated", f"{summary['total_students_allocated']:,}")
    col2.metric("Total Budget Used", f"${summary['total_budget_used']:,}")
    col3.metric("Selected Model", summary['selected_model_name'])
    
    st.write("---")
    st.write("### Allocation Tables")
    
    # Display allocation tables side-by-side.
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Major Allocations")
        st.dataframe(result_df[['Rank', 'Major', 'Minimum Students', 'Target Students', 'Allocated Students']].head(10).set_index('Rank'))
        
    with col2:
        st.subheader("Branch-Level Summary")
        branch_summary = summary['branch_summary'].copy()
        branch_summary.index.name = 'Department'
        branch_summary['Total Cost'] = branch_summary['Total Cost'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(branch_summary)
    
    st.markdown("---")
    st.write("### Allocation Visualizations")

    # Create and display a bar chart of student allocations per major.
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    result_df.set_index('Major')['Allocated Students'].plot(kind='bar', ax=ax1, color='mediumseagreen', edgecolor='black')
    ax1.set_title(f'Student Allocations per Major (Sorted by Rank) using {summary["selected_model_name"]}', fontsize=14)
    ax1.set_ylabel('Number of Allocated Students')
    ax1.tick_params(axis='x', rotation=90, labelsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)

    st.write("---")
    st.write("### Model Comparison: Target Allocations")

    # Get the theoretical allocations for all three models to plot them.
    ranks = result_df['Rank']
    alloc_power = get_allocations_from_model(df_excel, total_students, 1, alpha, k, decay_rate)
    alloc_sigmoid = get_allocations_from_model(df_excel, total_students, 2, alpha, k, decay_rate)
    alloc_exp = get_allocations_from_model(df_excel, total_students, 3, alpha, k, decay_rate)

    # Create and display a line chart comparing the allocation curves.
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    if choice == 1:
        ax2.plot(ranks, alloc_power, label='Power Law (Selected)', color='red', linewidth=2.5)
        ax2.plot(ranks, alloc_sigmoid, '--', color='gray', label='Sigmoid Model')
        ax2.plot(ranks, alloc_exp, '-', color='gray', label='Exponential Decay')
    elif choice == 2:
        ax2.plot(ranks, alloc_sigmoid, label='Sigmoid Model (Selected)', color='red', linewidth=2.5)
        ax2.plot(ranks, alloc_power, '--', color='gray', label='Power Law')
        ax2.plot(ranks, alloc_exp, '-', color='gray', label='Exponential Decay')
    else:
        ax2.plot(ranks, alloc_exp, label='Exponential Decay (Selected)', color='red', linewidth=2.5)
        ax2.plot(ranks, alloc_power, '--', color='gray', label='Power Law')
        ax2.plot(ranks, alloc_sigmoid, '-', color='gray', label='Sigmoid Model')

    ax2.set_title("Comparison of Allocation Curves Across Models", fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xlabel("Rank", fontsize=12)
    ax2.set_ylabel("Allocated Students (Log Scale)", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    st.write("---")
    st.write("### Model Smoothness Metrics")

    # Display smoothness tables side-by-side.
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Theoretical Model Smoothness")
        theory_data = {
            'Power Law': alloc_power,
            'Sigmoid': alloc_sigmoid, 
            'Exponential': alloc_exp
        }
        
        theory_table = []
        for model_name, allocation in theory_data.items():
            max_d, mean_d, median_d = calculate_smoothness(allocation)
            theory_table.append({
                'Model': model_name,
                'Max Drop (%)': f"{max_d:.2f}",
                'Mean Drop (%)': f"{mean_d:.2f}",
                'Median Drop (%)': f"{median_d:.2f}"
            })
        st.dataframe(pd.DataFrame(theory_table).set_index('Model'))
    
    with col2:
        st.write("#### Actual Allocation Smoothness")
        result_df['Drop Ratio'] = result_df['Allocated Students'].pct_change().abs()
        actual_max_drop = result_df['Drop Ratio'].max() * 100
        actual_mean_drop = result_df['Drop Ratio'].mean() * 100
        actual_median_drop = result_df['Drop Ratio'].median() * 100
        
        actual_smoothness_df = pd.DataFrame({
            'Metric': ['Max Drop Ratio (%)', 'Mean Drop Ratio (%)', 'Median Drop Ratio (%)'],
            'Value': [f"{actual_max_drop:.2f}", f"{actual_mean_drop:.2f}", f"{actual_median_drop:.2f}"]
        }).set_index('Metric')
        st.dataframe(actual_smoothness_df)
