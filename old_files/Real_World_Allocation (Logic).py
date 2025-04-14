import pandas as pd
from pulp import *
import matplotlib.pyplot as plt

# Load data
data = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Occupation_Major_Mapping_Singapore.xlsx'
df_sheet = pd.read_excel(data, sheet_name='Sheet2')

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

# 1.3. Assign budget and max students
df_sheet['Major Budget'] = df_sheet['Rank'].apply(lambda x: top45_budget if x <= 45 else rest_budget)
df_sheet['Student Cost'] = df_sheet['Branch'].map(branch_costs)
df_sheet['Max Students'] = (df_sheet['Major Budget'] / df_sheet['Student Cost']).astype(int)

# 1.4. Define target distribution (power law for gradual decrease)
total_students = 7000 # Number of students admitted to the university
alpha = 0.5  # Controls steepness (lower = flatter distribution)
df_sheet['Target Students'] = (total_students * (df_sheet['Rank'] ** -alpha) / 
                              (df_sheet['Rank'] ** -alpha).sum()).round().astype(int)

# 1.5. Optimisation with PuLP
model = LpProblem("Student_Allocation", LpMinimize)

# 1.6. Variables
majors = df_sheet['Major'].tolist()
student_vars = LpVariable.dicts("Students", majors, lowBound=0, cat='Integer')

# 1.7. Deviation variables (to minimize difference from target)
deviation_pos = LpVariable.dicts("Pos_Dev", majors, lowBound=0)
deviation_neg = LpVariable.dicts("Neg_Dev", majors, lowBound=0)

# 1.8. Objective function
model += lpSum([deviation_pos[m] + deviation_neg[m] for m in majors])

# 2. Constraints
# 2.1. Total students
model += lpSum([student_vars[m] for m in majors]) == total_students

# 2.2. Budget constraint
model += lpSum([student_vars[m] * df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] 
               for m in majors]) <= max_budget

# 2.3. Min/max students per major
for m in majors:
    model += student_vars[m] >= df_sheet.loc[df_sheet['Major'] == m, 'Minimum Students'].values[0]
    model += student_vars[m] <= df_sheet.loc[df_sheet['Major'] == m, 'Max Students'].values[0]

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

# 2.6. Deviation from target
for m in majors:
    target = df_sheet.loc[df_sheet['Major'] == m, 'Target Students'].values[0]
    model += student_vars[m] - target == deviation_pos[m] - deviation_neg[m]

# # Optional: 2.7. Prioritise majors with higher ranks first before applying constraints
# for i in range(len(majors)-1):
#     if df_sheet.iloc[i]['Rank'] < df_sheet.iloc[i+1]['Rank']:
#         model += student_vars[majors[i]] >= student_vars[majors[i+1]]

# 2.8. Solve
model.solve(PULP_CBC_CMD(msg=True))
print("Status:", LpStatus[model.status])

# 2.9. Add results to DataFrame
df_sheet['Allocated Students'] = df_sheet['Major'].apply(
    lambda x: int(value(student_vars[x]))
)

# 3. Output
# 3.1. Top 25 and bottom 25 dataframes
print("\nTop 25 Majors:")
print(df_sheet[['Rank', 'Major', 'Target Students', 'Allocated Students']].head(25))

print("\nBottom 25 Majors:")
print(df_sheet[['Rank', 'Major', 'Target Students', 'Allocated Students']].tail(25))

# 3.2. Print summary
print("\n=== Allocation Summary ===")
df_sheet['Total Cost'] = df_sheet['Allocated Students'] * df_sheet['Student Cost']
print(f"Total Students Allocated: {df_sheet['Allocated Students'].sum()}")
print(f"Total Budget Used: ${df_sheet['Total Cost'].sum():,}")

# 3.3. Verify smoothness
print("\n=== Drop Smoothness ===")
df_sheet['Drop Ratio'] = df_sheet['Allocated Students'].pct_change().abs()
print(f"Max Drop Ratio Between Adjacent Ranks: {df_sheet['Drop Ratio'].max() * 100:.2f} %")
print(f"Mean Drop Ratio Between Adjacent Ranks: {df_sheet['Drop Ratio'].mean() * 100:.2f} %")
print(f"Median Drop Ratio Between Adjacent Ranks: {df_sheet['Drop Ratio'].median() * 100:.2f} %")

# 3.4. Summary by branch
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
print(formatted_branch)

# 3.5. Create the histogram for each major and its allocation
plt.figure(figsize=(14, 6))
plt.bar(df_sheet['Major'], df_sheet['Allocated Students'], color='mediumseagreen', edgecolor='black')
plt.title('Allocated Students per Major (Sorted by Rank)', fontsize=14)
plt.xlabel('Major', fontsize=12)
plt.ylabel('Number of Allocated Students', fontsize=12)
plt.xticks(rotation=90, fontsize=8)  # Rotate x labels for readability
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # 4. Export to Excel
# output_path = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Corrected_Student_Allocation.xlsx'
# try:
#     df_sheet.to_excel(output_path, index=False)
#     print(f"Results exported to: {output_path}")
# except Exception as e:
#     print(f"Error exporting to Excel: {e}")
