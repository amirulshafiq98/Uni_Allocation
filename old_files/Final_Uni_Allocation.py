import pandas as pd
from pulp import *
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Occupation_Major_Mapping_Singapore.xlsx'
df_sheet = pd.read_excel(data, sheet_name='Sheet2')
pd.set_option('display.max_columns', None)

# 1. Define parameters
# 1.1. Cost per student in each branch
branch_costs = {
    'Technology': 50000,
    'Science': 50000,
    'Healthcare': 60000,
    'Engineering': 40000,
    'Arts': 40000,
    'Other': 30000
}

# 1.2. Budgeting
df_sheet = df_sheet.sort_values(by='Rank', ascending=True)
top45_budget = 10_000_000  # Budget for each top 45 major
rest_budget = 5_000_000    # Budget for each remaining major
max_budget = 500_000_000   # Total university budget
total_students = 7000 # Number of students admitted to the university

# 1.3. Assign budget and max students
df_sheet['Major Budget'] = df_sheet['Rank'].apply(lambda x: top45_budget if x <= 45 else rest_budget)
df_sheet['Student Cost'] = df_sheet['Branch'].map(branch_costs)
df_sheet['Maximum Students'] = (df_sheet['Major Budget'] / df_sheet['Student Cost']).astype(int)

# 1.4. Define target distribution the constants - Controls steepness (lower = flatter)
alpha = 0.5 
k = 0.1 
decay_rate = 0.01  

# 1.5. User input to choose the model
while True:
    try:
       print('Enter 1: Power Law, 2: Sigmoid, 3: Exponential Decay')
       choice = int(input('Choose the distribution model: ')) 
       if choice in [1, 2, 3]:
           break
       else:
            print('Invalid choice. Please enter 1, 2, or 3.')
    except ValueError:
        print('Invalid input. Please enter a number.')

# --- Define target distribution
if choice == 1:
    print('Power Law model selected.')
    name = 'Power Law'
    # --- Power Law Model 
    df_sheet['Power Law'] = (total_students * (df_sheet['Rank'] ** -alpha) / (df_sheet['Rank'] ** -alpha).sum())
    df_sheet['Target Students'] = df_sheet['Power Law'].round().astype(int) # Set Target Students to Power Law

elif choice == 2:
    print('Sigmoid model selected.')
    name = 'Sigmoid Model'
    # --- Sigmoid Model
    midpoint = df_sheet['Rank'].median()
    df_sheet['Sigmoid Target'] = df_sheet['Rank'].apply(
        lambda rank: total_students / (1 + np.exp(-k * (rank - midpoint)))
    )
    df_sheet['Target Students'] = df_sheet['Sigmoid Target'].round().astype(int) # Set Target Students to Sigmoid

else:
    print('Exponential Decay model selected.')
    name = 'Exponential Decay'
    # --- Exponential Decay
    scale_factor = total_students / sum(np.exp(-decay_rate * df_sheet['Rank']))
    df_sheet['Exponential Decay'] = df_sheet['Rank'].apply(
        lambda rank: scale_factor * np.exp(-decay_rate * rank)
    )
    df_sheet['Target Students'] = df_sheet['Exponential Decay'].round().astype(int) # Set Target Students to Exponential Decay

# 1.6. Define the model
model = LpProblem("Student_Allocation", LpMinimize)

# 1.7. Define variables
majors = df_sheet['Major'].tolist()
student_vars = LpVariable.dicts("Students", majors, lowBound=0, cat='Integer')

# 1.8. Define deviation variables (to minimize difference from target)
deviation_neg = LpVariable.dicts("Neg_Dev", majors, lowBound=0)
deviation_pos = LpVariable.dicts("Pos_Dev", majors, lowBound=0)

# 1.9. Objective function
model += lpSum([deviation_pos[m] + deviation_neg[m] for m in majors]), "Minimise_Deviation"

# 2. Constraints
# 2.1. Total students
model += lpSum([student_vars[m] for m in majors]) == total_students

# 2.2. Budget constraint
model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] 
               for m in majors]) <= max_budget

# 2.3. Min/max students per major
for m in majors:
    model += student_vars[m] >= df_sheet.loc[df_sheet['Major'] == m, 'Minimum Students'].values[0]
    model += student_vars[m] <= df_sheet.loc[df_sheet['Major'] == m, 'Maximum Students'].values[0]

# 2.4. Add Branch-Level Budget Constraints ---
branch_groups = df_sheet.groupby('Branch')['Student Cost'].sum()

# --- Total Arts budget <= Total Engineering budget
arts_majors = df_sheet[df_sheet['Branch'] == 'Arts']['Major'].tolist()
eng_majors = df_sheet[df_sheet['Branch'] == 'Engineering']['Major'].tolist()

model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] 
               for m in arts_majors]) <= \
         lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] 
               for m in eng_majors]), "Arts_leq_Engineering"

# --- Total Healthcare budget >= Total Tech budget
health_majors = df_sheet[df_sheet['Branch'] == 'Healthcare']['Major'].tolist()
tech_majors = df_sheet[df_sheet['Branch'] == 'Technology']['Major'].tolist()

model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] 
               for m in health_majors]) >= \
         lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] 
               for m in tech_majors]), "Healthcare_geq_Technology"

# 2.5. Gradual decrease with different ratios for top and bottom ranks
for i in range(1, len(majors)):
    prev_major = majors[i-1]
    curr_major = majors[i]
    prev_rank = df_sheet.loc[df_sheet['Major'] == prev_major, 'Rank'].values[0]
    
    if prev_rank <= 45:  # For top 45 ranks
        model += student_vars[curr_major] >= 0.8 * student_vars[prev_major]
    else:  # For the rest
        model += student_vars[curr_major] >= 0.5 * student_vars[prev_major]

# 2.6. Deviation constraints
for m in majors:
    target = df_sheet.loc[df_sheet['Major'] == m, 'Target Students'].values[0]
    model += student_vars[m] - target == deviation_pos[m] - deviation_neg[m]

# 2.7. Solve
model.solve(PULP_CBC_CMD(msg=True))
print('====================================')
print("Status:", LpStatus[model.status])

# 2.8. Add results to DataFrame
df_sheet['Allocated Students'] = df_sheet['Major'].apply(
    lambda x: int(value(student_vars[x]))
)

# 3. Output
# 3.1. Top 25 and bottom 25 dataframes
print("\nTop 25 Majors:")
print(df_sheet[['Rank', 'Major', 'Minimum Students', 'Target Students', 'Allocated Students']].head(25))

print("\nBottom 25 Majors:")
print(df_sheet[['Rank', 'Major', 'Minimum Students', 'Target Students', 'Allocated Students']].tail(25))

# 3.2. Print summary
print("\n=== Allocation Summary ===")
df_sheet['Total Cost'] = df_sheet['Allocated Students'] * df_sheet['Student Cost']
print(f"Total Students Allocated: {df_sheet['Allocated Students'].sum()}")
print(f"Total Budget Used: ${df_sheet['Total Cost'].sum():,}")

# 3.3. Summary by branch
branch_summary = df_sheet.groupby('Branch').agg({
    'Allocated Students': 'sum',
    'Total Cost': 'sum',
}).sort_values('Allocated Students', ascending=False)

# --- Format the branch summary with commas
formatted_branch = branch_summary.reset_index()
formatted_branch['Allocated Students'] = formatted_branch['Allocated Students'].apply(lambda x: f"{x:,}")
formatted_branch['Total Cost'] = formatted_branch['Total Cost'].apply(lambda x: f"${x:,.0f}")

# --- Print the branch summary
print("\nBranch Summary:")
print((formatted_branch).set_index('Branch'))

# 3.4. Create the histogram for each major and its allocation
plt.figure(figsize=(14, 6))
plt.bar(df_sheet['Major'], df_sheet['Allocated Students'], color='mediumseagreen', edgecolor='black')
plt.title(f'Student Allocations per Major (Sorted by Rank) using {name}', fontsize=14)
plt.xlabel('Major', fontsize=12)
plt.ylabel('Number of Allocated Students', fontsize=12)
plt.xticks(rotation=90, fontsize=8)  # Rotate x labels for readability
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3.5. Allocation Curves for Each Model (Power Law, Sigmoid, Exponential Decay) 
def get_allocations_from_model(df, model_choice):
    df_copy = df.copy()
    ranks = df_copy['Rank']
    
    if model_choice == 1:
        target = (total_students * (ranks ** -alpha) / (ranks ** -alpha).sum()).round()
    elif model_choice == 2:
        midpoint = ranks.median()
        target = (total_students / (1 + np.exp(-k * (ranks - midpoint)))).round()
    else:
        raw = total_students * np.exp(-decay_rate * ranks)
        target = (raw / raw.sum() * total_students).round()
    
    return target.values

# 3.6. Calculate All 3 Allocated Curves
alloc_power = get_allocations_from_model(df_sheet, 1)
alloc_sigmoid = get_allocations_from_model(df_sheet, 2)
alloc_exp = get_allocations_from_model(df_sheet, 3)

ranks = df_sheet['Rank']

# 3.7. Verify smoothness with separate tables for theory and actual
print("\nTheoretical Model Smoothness:")

# --- Calculate metrics for theoretical models
theory_data = {
    'Power Law': alloc_power,
    'Sigmoid': alloc_sigmoid, 
    'Exponential': alloc_exp
}

theory_table = []
# --- Iterate through each model and calculate drop ratios
for model_name, allocation in theory_data.items():
    # --- Calculate drop ratios
    drop_ratios = []
    for i in range(1, len(allocation)):
        if allocation[i-1] > 0:  # Avoid division by zero
            ratio = abs((allocation[i] - allocation[i-1]) / allocation[i-1])
            drop_ratios.append(ratio)
    
    # --- Calculate stats
    max_drop = max(drop_ratios) * 100 if drop_ratios else 0
    mean_drop = sum(drop_ratios) * 100 / len(drop_ratios) if drop_ratios else 0
    median_drop = sorted(drop_ratios)[len(drop_ratios)//2] * 100 if drop_ratios else 0
    
    # --- Add to table
    theory_table.append({
        'Model': model_name,
        'Max Drop (%)': f"{max_drop:.2f}",
        'Mean Drop (%)': f"{mean_drop:.2f}",
        'Median Drop (%)': f"{median_drop:.2f}"
    })

# --- Print theoretical model table
print(pd.DataFrame(theory_table).set_index('Model'))
print(f"\nSelected Model: {name}")

# --- Calculate and display actual allocation smoothness
print("=== Actual Allocation Smoothness ===")
df_sheet['Drop Ratio'] = df_sheet['Allocated Students'].pct_change().abs()
print(f"Max Drop Ratio Between Adjacent Ranks: {df_sheet['Drop Ratio'].max() * 100:.2f} %")
print(f"Mean Drop Ratio Between Adjacent Ranks: {df_sheet['Drop Ratio'].mean() * 100:.2f} %")
print(f"Median Drop Ratio Between Adjacent Ranks: {df_sheet['Drop Ratio'].median() * 100:.2f} %")

# 3.8. Plot with Selected in Red, Others in Gray
plt.figure(figsize=(12, 6))

if choice == 1:
    plt.plot(ranks, alloc_power, label='Power Law (Selected)', color='red', linewidth=2.5)
    plt.plot(ranks, alloc_sigmoid, '--', color='gray', label='Sigmoid Model')
    plt.plot(ranks, alloc_exp, '-', color='gray', label='Exponential Decay')
elif choice == 2:
    plt.plot(ranks, alloc_sigmoid, label='Sigmoid Model (Selected)', color='red', linewidth=2.5)
    plt.plot(ranks, alloc_power, '--', color='gray', label='Power Law')
    plt.plot(ranks, alloc_exp, '-', color='gray', label='Exponential Decay')
else:
    plt.plot(ranks, alloc_exp, label='Exponential Decay (Selected)', color='red', linewidth=2.5)
    plt.plot(ranks, alloc_power, '--', color='gray', label='Power Law')
    plt.plot(ranks, alloc_sigmoid, '-', color='gray', label='Sigmoid Model')

# --- Aesthetic settings for the plot
plt.title("Comparison of Allocation Curves Across Models", fontsize=14)
plt.yscale('log')  # Log scale for better visibility of differences
plt.xlabel("Rank", fontsize=12)
plt.ylabel("Allocated Students (Log)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Export to Excel
output_path = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Student_Allocation.xlsx'
try:
    df_sheet.to_excel(output_path, index=False)
    print(f"Results exported to: {output_path}")
except Exception as e:
    print(f"Error exporting to Excel: {e}")
