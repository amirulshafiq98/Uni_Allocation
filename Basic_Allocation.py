import pandas as pd
from pulp import *

# Load data
data = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Occupation_Major_Mapping_Singapore.xlsx'
df_sheet = pd.read_excel(data, sheet_name='Sheet2')

# Cost per student in each branch
branch_costs = {
    'Technology': 50000,
    'Science': 50000,
    'Healthcare': 60000,
    'Engineering': 40000,
    'Business': 40000,
    'Arts': 30000,
    'Other': 30000
}

# Budgeting for university
df_sheet = df_sheet.sort_values(by='Rank', ascending=True)
top45_budget = 10_000_000  # Budget for each top 45 major (using underscores for readability)
rest_budget = 5_000_000    # Budget for each remaining major
max_budget = 500_000_000  # Total university budget

# Assign budget based on rank
df_sheet['Major Budget'] = df_sheet['Rank'].apply(lambda x: top45_budget if x <= 45 else rest_budget)

# Map student costs to their respective branches
df_sheet['Student Cost'] = df_sheet['Branch'].map(branch_costs)

# Calculate maximum students possible based on budget
df_sheet['Max Students'] = (df_sheet['Major Budget'] / df_sheet['Student Cost']).astype(int)

# Points allocation
df_sheet['Points'] = 1 / df_sheet['Rank']

# Integer Linear Programming Model
model = LpProblem("Student_Allocation", LpMaximize)

# Define Variables 
majors = df_sheet['Major'].tolist()
student_var = LpVariable.dicts("Students", majors, lowBound=0, cat='Integer')

# Objective Function
model += lpSum([df_sheet.loc[df_sheet['Major'] == m, 'Points'].values[0] * student_var[m] for m in majors]), "Weighted_Student_Allocation"

# Total University Budget Constraint
model += lpSum([df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0] * student_var[m] for m in majors]) <= max_budget, "Total_Budget"

# Individual Major Budget and Minimum/Maximum Constraints
for m in majors:
    # Budget constraint - can't exceed the major's budget
    student_cost = df_sheet.loc[df_sheet['Major'] == m, 'Student Cost'].values[0]
    major_budget = df_sheet.loc[df_sheet['Major'] == m, 'Major Budget'].values[0]
    max_students = int(major_budget / student_cost)
    model += student_var[m] <= max_students, f"Budget_Constraint_{m}"
    
    # Minimum student constraint - must meet the minimum requirement
    min_students = df_sheet.loc[df_sheet['Major'] == m, 'Minimum Students'].values[0]
    model += student_var[m] >= min_students, f"Min_Students_{m}"

# Add a constraint for total number of students if desired
target_students = 5000 # Example: University wants to admit 5000 students
model += lpSum([student_var[m] for m in majors]) == target_students, "Total_Students_Target"

# Solve the model
model.solve(PULP_CBC_CMD(msg=False))
print("Status:", LpStatus[model.status])

# Display results
if model.status == 1:  # If optimal solution found
    allocations = {m: int(student_var[m].value()) for m in majors}
    df_sheet['Allocated Students'] = df_sheet['Major'].map(allocations)
    
    # Calculate final costs
    df_sheet['Total Cost'] = df_sheet['Allocated Students'] * df_sheet['Student Cost']
    
    # Format numbers with commas
    print(f"\nTotal students allocated: {df_sheet['Allocated Students'].sum():,}")
    print(f"Total university cost: ${df_sheet['Total Cost'].sum():,.0f}")
    
    # Display allocation results with Rank as index
    result_df = df_sheet[['Rank', 'Major', 'Branch', 'Minimum Students', 'Allocated Students', 'Total Cost']]
    result_df = result_df.set_index('Rank').sort_index()
    
    # Format the numeric columns in the dataframe for display with commas
    formatted_result = result_df.copy()
    formatted_result['Minimum Students'] = formatted_result['Minimum Students'].apply(lambda x: f"{x:,}")
    formatted_result['Allocated Students'] = formatted_result['Allocated Students'].apply(lambda x: f"{x:,}")
    formatted_result['Total Cost'] = formatted_result['Total Cost'].apply(lambda x: f"${x:,.0f}")

    # Format number columns to show commas
    pd.options.display.float_format = '{:,.0f}'.format
    
    # Print the formatted result
    print("\nStudent Allocation Results:")
    print(formatted_result)
    
    # Summary by branch - fixing the display issue
    branch_summary = df_sheet.groupby('Branch').agg({
        'Allocated Students': 'sum',
        'Total Cost': 'sum',
    }).sort_values('Allocated Students', ascending=False)
    
    # Format the branch summary with commas
    formatted_branch = branch_summary.reset_index()
    formatted_branch['Allocated Students'] = formatted_branch['Allocated Students'].apply(lambda x: f"{x:,}")
    formatted_branch['Total Cost'] = formatted_branch['Total Cost'].apply(lambda x: f"${x:,.0f}")
    
    # Print the branch summary
    print("\nBranch Summary:")
    print(formatted_branch)

    # Create an Excel writer object
    output_file = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Student_Allocation_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save the main results to the first sheet
        result_df.to_excel(writer, sheet_name='Allocation Results')
        
        # Save the branch summary to another sheet
        branch_summary.reset_index().to_excel(writer, sheet_name='Branch Summary', index=False)
        
        # Optionally save the original data with all calculated columns
        df_sheet.to_excel(writer, sheet_name='Complete Data', index=False)
    
    print(f"\nResults exported to {output_file}")
else:
    print("No optimal solution found. Check if constraints can be satisfied.")
