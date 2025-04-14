# allocation_model.py

# Import the necessary libraries for the core logic.
import pandas as pd
from pulp import *
import numpy as np

# ==================================================================================================
# SECTION 1: CORE ALLOCATION LOGIC FUNCTIONS
# This section contains the reusable backend functions that perform the calculations and optimization.
# These functions do not rely on a web framework and can be imported into other applications.
# ==================================================================================================

def get_allocations_from_model(df, total_students, choice, alpha, k, decay_rate):
    """
    Calculates the theoretical target student allocations for different models (Power Law, Sigmoid,
    or Exponential Decay). This is used for plotting and comparison.

    Args:
        df (pd.DataFrame): The input DataFrame containing major ranks.
        total_students (int): The total number of students to be allocated.
        choice (int): An integer representing the selected distribution model (1, 2, or 3).
        alpha (float): The decay constant for the Power Law model.
        k (float): The growth constant for the Sigmoid model.
        decay_rate (float): The constant for the Exponential Decay model.

    Returns:
        np.array: An array of target student allocations based on the chosen model.
    """
    df_copy = df.copy()
    ranks = df_copy['Rank']
    
    if choice == 1: # Power Law Model
        # Allocates students with a steep drop-off, favoring higher ranks.
        target = (total_students * (ranks ** -alpha) / (ranks ** -alpha).sum()).round()
    elif choice == 2: # Sigmoid Model
        # Creates an S-shaped curve for allocation, with a more gradual transition.
        midpoint = ranks.median()
        target = (total_students / (1 + np.exp(-k * (ranks - midpoint)))).round()
    else: # Exponential Decay Model
        # Allocates students with a continuous, exponential decline from the top ranks.
        raw = total_students * np.exp(-decay_rate * ranks)
        target = (raw / raw.sum() * total_students).round()
    
    return target.values

def run_allocation_model(df_excel, total_students, choice, alpha, k, decay_rate, max_budget):
    """
    Executes the full linear programming optimization model. It takes user parameters,
    sets up the LP problem with all constraints, solves it, and returns the results.
    
    Args:
        df_excel (pd.DataFrame): The input DataFrame from the uploaded Excel file.
        total_students (int): The total number of students to allocate.
        choice (int): The selected distribution model (1=Power, 2=Sigmoid, 3=Exponential).
        alpha (float): The constant for the Power Law model.
        k (float): The constant for the Sigmoid model.
        decay_rate (float): The constant for the Exponential Decay model.
        max_budget (int): The maximum total university budget.

    Returns:
        tuple: A tuple containing the final results DataFrame and a dictionary of summary metrics.
    """
    df_sheet = df_excel.copy()
    
    # --- 1. Define Model Parameters ---
    branch_costs = {
        'Technology': 50000, 'Science': 50000, 'Healthcare': 60000,
        'Engineering': 40000, 'Arts': 40000, 'Other': 30000
    }
    
    df_sheet = df_sheet.sort_values(by='Rank', ascending=True)
    top45_budget = 10_000_000 
    rest_budget = 5_000_000
    
    df_sheet['Major Budget'] = df_sheet['Rank'].apply(lambda x: top45_budget if x <= 45 else rest_budget)
    df_sheet['Student Cost'] = df_sheet['Branch'].map(branch_costs)
    df_sheet['Maximum Students'] = (df_sheet['Major Budget'] / df_sheet['Student Cost']).astype(int)
    df_sheet['Minimum Students'] = 1 

    # --- 2. Set the Target Distribution ---
    if choice == 1:
        name = 'Power Law'
        df_sheet['Power Law'] = (total_students * (df_sheet['Rank'] ** -alpha) / (df_sheet['Rank'] ** -alpha).sum())
        df_sheet['Target Students'] = df_sheet['Power Law'].round().astype(int)
    elif choice == 2:
        name = 'Sigmoid Model'
        midpoint = df_sheet['Rank'].median()
        df_sheet['Sigmoid Target'] = df_sheet['Rank'].apply(lambda rank: total_students / (1 + np.exp(-k * (rank - midpoint))))
        df_sheet['Target Students'] = df_sheet['Sigmoid Target'].round().astype(int)
    else:
        name = 'Exponential Decay'
        scale_factor = total_students / sum(np.exp(-decay_rate * df_sheet['Rank']))
        df_sheet['Exponential Decay'] = df_sheet['Rank'].apply(lambda rank: scale_factor * np.exp(-decay_rate * rank))
        df_sheet['Target Students'] = df_sheet['Exponential Decay'].round().astype(int)

    # --- 3. Define and Solve the Linear Programming Problem ---
    model = LpProblem("Student_Allocation", LpMinimize)
    majors = df_sheet['Major'].tolist()
    student_vars = LpVariable.dicts("Students", majors, lowBound=0, cat='Integer')
    deviation_neg = LpVariable.dicts("Neg_Dev", majors, lowBound=0)
    deviation_pos = LpVariable.dicts("Pos_Dev", majors, lowBound=0)
    
    model += lpSum([deviation_pos[m] + deviation_neg[m] for m in majors]), "Minimise_Deviation"
    
    model += lpSum([student_vars[m] for m in majors]) == total_students
    model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] for m in majors]) <= max_budget
    
    for m in majors:
        model += student_vars[m] >= df_sheet.loc[df_sheet['Major'] == m, 'Minimum Students'].values[0]
        model += student_vars[m] <= df_sheet.loc[df_sheet['Major'] == m, 'Maximum Students'].values[0]
    
    arts_majors = df_sheet[df_sheet['Branch'] == 'Arts']['Major'].tolist()
    eng_majors = df_sheet[df_sheet['Branch'] == 'Engineering']['Major'].tolist()
    model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] for m in arts_majors]) <= \
             lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] for m in eng_majors]), "Arts_leq_Engineering"

    health_majors = df_sheet[df_sheet['Branch'] == 'Healthcare']['Major'].tolist()
    tech_majors = df_sheet[df_sheet['Branch'] == 'Technology']['Major'].tolist()
    model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] for m in health_majors]) >= \
             lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] for m in tech_majors]), "Healthcare_geq_Technology"

    for i in range(1, len(majors)):
        prev_major = majors[i-1]
        curr_major = majors[i]
        prev_rank = df_sheet.loc[df_sheet['Major'] == prev_major, 'Rank'].values[0]
        if prev_rank <= 45:
            model += student_vars[curr_major] >= 0.8 * student_vars[prev_major]
        else:
            model += student_vars[curr_major] >= 0.5 * student_vars[prev_major]

    for m in majors:
        target = df_sheet.loc[df_sheet['Major'] == m, 'Target Students'].values[0]
        model += student_vars[m] - target == deviation_pos[m] - deviation_neg[m]

    model.solve(PULP_CBC_CMD(msg=False))

    # --- 4. Process and Return Results ---
    df_sheet['Allocated Students'] = df_sheet['Major'].apply(
        lambda x: int(value(student_vars[x]))
    )
    df_sheet['Total Cost'] = df_sheet['Allocated Students'] * df_sheet['Student Cost']
    df_sheet['Allocated Students'] = df_sheet['Allocated Students'].astype(int)

    summary = {
        "status": LpStatus[model.status],
        "total_students_allocated": df_sheet['Allocated Students'].sum(),
        "total_budget_used": df_sheet['Total Cost'].sum(),
        "selected_model_name": name,
        "branch_summary": df_sheet.groupby('Branch').agg({
            'Allocated Students': 'sum',
            'Total Cost': 'sum',
        }).sort_values('Allocated Students', ascending=False)
    }

    return df_sheet, summary

def calculate_smoothness(allocation):
    """Helper function to calculate smoothness metrics."""
    drop_ratios = []
    for i in range(1, len(allocation)):
        if allocation[i-1] > 0:
            ratio = abs((allocation[i] - allocation[i-1]) / allocation[i-1])
            drop_ratios.append(ratio)
    
    if not drop_ratios:
        return 0, 0, 0
    
    max_drop = max(drop_ratios) * 100
    mean_drop = sum(drop_ratios) * 100 / len(drop_ratios)
    median_drop = sorted(drop_ratios)[len(drop_ratios)//2] * 100
    
    return max_drop, mean_drop, median_drop
