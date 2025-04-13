import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = 'c:/Users/amiru/Desktop/Portfolio/Coding/WEF Jobs/Occupation_Major_Mapping_Singapore.xlsx'
df_excel = pd.read_excel(data, sheet_name='Sheet1')

def major_allocations(df, students, total_majors=125):
    major = []
    rank = []
    for i in range(total_majors):
        major.append(df.iloc[i, 1])
        rank.append(df.iloc[i, 0])

    # Point allocation based on rank
    points = []
    for r in rank:
        if r != 0:
            point = 1 / r
            points.append(point)
        else:
            points.append(0)
    sum_points = sum(points)

    # Calculate the allocation for each major
    allocations = []
    for p in points:
        allocation = (students * p) / sum_points
        allocations.append(int(allocation))

    # Create a DataFrame to display the results
    df = pd.DataFrame({
        'Rank': rank,
        'Major': major,
        'Allocation': allocations,
    })

    # Set the index to 'Rank' and sort the DataFrame by 'Rank'
    df = df.set_index('Rank')
    df = df.sort_values(by='Rank', ascending=True)
    sum_allocations = sum(allocations)

    # Calculate the difference and redistribute allocations
    difference = students - sum_allocations
    if difference > 0:
        percent = df['Allocation'].max() * 0.05
        filtered = df[df['Allocation'] <= int(percent)]
        num_major = len(filtered)
        # Distribute the difference among majors with allocations <= percent
        if num_major > 0:
            bottom_majors = filtered.index.tolist()
            bottom_majors.sort(reverse = True) # Sort in descending order
            
            distributed = 0
            i = 0
            while distributed < difference:
                # Add 1 student to the current major
                current_major = bottom_majors[i]
                df.loc[current_major, 'Allocation'] += 1
                distributed += 1
                
                # Move to next major (loop back to start if needed)
                i = (i + 1) % num_major
    return df

while True:
    try:
        students = int(input("Enter the number of students: "))
        if students > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Display the student allocations
result_table = major_allocations(df_excel, students)
print('Final Major Allocations Table:')
print(result_table)
    
