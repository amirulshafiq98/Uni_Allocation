<p align="center">
  <img src="https://c1.wallpaperflare.com/preview/294/960/948/university-student-graduation-photo-hats.jpg">
</p>

# Project Background [Steamlit Link](https://uniallocation.streamlit.app/)
After reading the WEF report from [January 2025](https://reports.weforum.org/docs/WEF_Future_of_Jobs_Report_2025.pdf), I was curious as to what jobs would be impacted by this new AI age. Looking through the report, I found out what jobs were expected to grow in the next 5 years and which ones are likely to be automated by AI. I then wondered: 'Do universities consider such reports when deciding how to allocate admissions for their courses?'. After some short amount of digging, the answer was inconclusive. In the context of Singapore (where I am from), the Ministry of Education (MOE) determines student intake for each course based on economic factors and course popularity. While this approach does address course allocation, it raises further questions:

- What is the precise methodology employed by the ministry to determine student numbers for each course?
- Does the allocated education budget influence the intake numbers for in-demand courses?
- In the event of an increased overall intake compared to the previous year, how would this additional capacity be distributed across a university's various courses?

These questions were what motivated me to start this project, which aims to explore a method for allocating university admissions that reflects real-world demand. A key concern for universities is ensuring their graduates enter a relevant and receptive job market. While some data is unknown to me as I am merely a soon-to-be graduate, we can at least simulate this situation to maximise the student allocation to produce graduates that are employable.

# Data Structure
The data was sourced from a YouTuber named [Tina Huang](https://www.youtube.com/watch?v=x2x8Ww7Es4s&t=43s) who compiled all of the jobs mentioned in the report into an Excel file. I then manually removed the jobs that did not require a degree and ranked them based on the same report. To tailor this analysis to Singapore, I used ChatGPT to generate a table of relevant degree programs offered in the country that was able to do the jobs mentioned in the report. Subsequently, I added minimum student numbers for each course and their respective branches of study. This was done to simulate a more official structure and facilitate coding. The first 20 majors in the [Excel file](https://github.com/amirulshafiq98/Uni_Allocation/blob/main/Occupation_Major_Mapping_Singapore.xlsx) can be seen below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b734c52-9e48-47e5-b444-8659759c9389">
</p>

# Executive Summary [Steamlit Link](https://uniallocation.streamlit.app/)
### Overview:

![Histogram](https://github.com/user-attachments/assets/bd2d4948-cdae-4585-93eb-b648c80f74b7)

Prior to coding, I researched the average cost per student for various courses. These costs were averaged across branches of study for simplification during coding (e.g., Engineering: $40,000, Technology: $50,000, Science: $50,000). My research indicated an approximate intake of 7000 students per cohort based on numbers I could find from NTU and NUS. I then estimated a minimum student number per course per cohort, resulting in the final Excel file used for reference as seen in the previous section.

In addition to these values, I incorporated constraints such as a maximum university budget of $500,000,000 and per-course budget limits (Top courses: $10,000,000; Other courses: $5,000,000). With these parameters defined, I plotted the student allocations for each major in a histogram to visualise the gradual decline in allocations across the various courses.

### Code:
There are two main files that run the web app:

- `app.py:` Contains the Streamlit user interface and controls

- `allocation_model.py:` Holds all the core logic, including the run_allocation_model and calculate_smoothness functions

Python libraries used in this project are:
- pandas
- matplotlib
- pulp
- numpy
- openpyxl

1. As outlined in the overview, the initial step involved defining the parameters for this case study. Furthermore, I categorised the majors into 'Top courses' and 'Other courses' to enable a budget allocation strategy that mirrors real-world university practices. Subsequently, I mapped each major to its corresponding cost per student, derived from online research, as shown below.

![First](https://github.com/user-attachments/assets/092635f5-1b8b-49e0-9220-29e7c456b73c)

2. To manage the decline in student allocation across majors, I incorporated a few models - power law, sigmoid and the exponential decay as represented by the equations below:

**[Power law model](https://en.wikipedia.org/wiki/Power_law)**

For this model α=0.5 (0.0 ≤ α ≤ 1.0),

$$ Students_i = {Total \\,\\, Students\\,\\, × \\,\\,Rank_i^{-α} \over ∑_{j=1}^N Rank_j^{-α}} $$

Where:
- α = steepness parameter (lower = flatter distribution)
- N = total number of majors

**[Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function)**

For this model k=0.1 (0.0 ≤ k ≤ 1.0),

$$ Students_i = {Total \\,\\, Students\\,\\, \over 1+e^{-k(Rank_i-Midpoint)}} $$

Where:
- k = steepness parameter (lower = gentler transition)
- Midpoint = median rank (e.g. middle rank)

**[Exponential Decay](https://en.wikipedia.org/wiki/Exponential_decay)**

For this model λ=0.1 (0.00 ≤ λ ≤ 0.10),

$$ Students_i = {Total \\,\\, Students\\,\\, × \\,\\, e^{-λ\\,\\,×\\,\\,Rank_i} \over ∑_{j=1}^Ne^{-λ\\,\\,×\\,\\,Rank_j}} $$

Where:
- λ = decay rate (lower = slower decay)

Here are the equations translated into python:

![Second](https://github.com/user-attachments/assets/a656232f-1c5d-4956-af5d-f2107ab03958)

3. With the distribution models established, the next step involved defining variables, deviations, and the objective function.

The constraints are:
- Budget constraints
- Min/Max students per major
- Branch Constraints
- Set-ratios to mitigate significant drops in student allocation between consecutive majors

![Third](https://github.com/user-attachments/assets/1dbef253-dc7a-4a05-8ef3-a65f05d02b81)

4. The model was then solved, and the resulting allocation data was stored in a DataFrame

![Fourth](https://github.com/user-attachments/assets/ca8238f1-d22d-47ce-81bd-ab92ec64b0fc)

5. The outputted power law model Excel file can be downloaded here [Link](https://github.com/amirulshafiq98/Uni_Allocation/blob/main/raw%20data/Power%20Law_Student_Allocation.xlsx)

<p align="center">
  <img src="https://github.com/user-attachments/assets/94d322fa-0637-4dbc-95da-e64eab27b631">
</p>

6. A snippet of the streamlit web app code can be seem below running the logic seen above seamlessly allowing for a user friendly interface the full list of the streamlit code can be downloaded here [Link](https://github.com/amirulshafiq98/Uni_Allocation/blob/main/app.py)

![App](https://github.com/user-attachments/assets/9909c61c-dff9-464e-8e67-9d863d8380ae)

### Model Comparison

![Line Curve](https://github.com/user-attachments/assets/3801e73f-52a9-4d87-aaad-fe7108f34814)

While the power law model provides a general approach for gradual decline, the exponential decay and sigmoid curve models are also valid alternatives. As illustrated in the line graph above, the power law model's decline pattern more closely aligned with the trends suggested in the WEF report. Exponential decay exhibits a rapid initial decrease, while the sigmoid model tends to distribute allocations more towards the middle ranks, leading to a higher intake in moderately ranked courses. Overall, the **power law** model appears to be the most suitable, as it prioritises higher student intake in top-ranked courses before distributing allocations to other majors, potentially better preparing graduates for the current job market

### Branch Summary and Total Costs:

<p align="center">
  <img src="https://github.com/user-attachments/assets/352563bf-3e86-4f1a-a92d-eeac2b743789">
</p>

- As specified in the constraints, the healthcare branch was allocated a similar budget to technology but with 400 fewer students. This aligns with the real-world trend of an expanding healthcare sector in Singapore due to an ageing population, [requiring more experts in the field](https://www.channelnewsasia.com/singapore/allied-health-professionals-healthcare-nurses-hospitals-shortage-radiographers-pharmacists-3861226#:~:text=few%20private%20hospitals.-,The%20crunch%20has%20been%20getting%20worse%20since%20the%20COVID%2D19,the%20expansion%20of%20healthcare%20infrastructure.)
- As anticipated, the technology branch has the highest student allocation, reflecting the sector's projected [CAGR of 16.3% from 2024-2029](https://sbr.com.sg/information-technology/news/singapore-ict-market-grow-1903b-2028)
- With an estimated budget of $500,000,000 for a 7000-student intake, the total allocated expenditure for all 92 majors is $329,080,000, leaving a significant surplus for funding projects and scholarships, mirroring real-world university financial practices

### Drop Smoothness:

<p align="center">
  <img src="https://github.com/user-attachments/assets/746f0d41-ce9e-48c7-a915-19ddd2945239">
</p>

- The 'drop smoothness' metric indicates the magnitude of the decrease in student allocations between consecutive majors (based on rank)
- A high maximum drop smoothness is not necessarily negative, as the objective is to prioritize student intake in top-ranked courses before allocating to less popular ones
- The theoretical drop smoothness values suggest that the exponential decay model might be optimal under minimal constraints in the ILP optimiser. While not fully representative of real-world scenarios, this highlights the model's ability to allocate students based on the ranking of majors
- Another way to evaluate a model's allocation prowess is by seeing the mean drop smoothness. In the case of the power law model, the mean was 18.61% which is slightly higher than the exponential decay model (15.95%) and much lower than the sigmoid model (55.46%)

# Recommendations [Steamlit Link](https://uniallocation.streamlit.app/)
- The power law model is recommended as the best overall model due to its closer alignment with theoretical smoothing and its effectiveness in allocating a majority of students to top-ranked courses
- Given that the constants used to determine the decline are currently arbitrary (alpha = 0.5, decay_rate = 0.01, k = 0.1), future improvements could involve using machine learning to optimise these values for each model, leading to a more refined allocation spread before final model selection
- If the exponential decay model is preferred due to its theoretical accuracy, reducing the number of constraints might lead to a more optimal allocation spread
- Generally, the sigmoid model should not be considered unless there is a need to fill up the less popular courses

# Limitations [Steamlit Link](https://uniallocation.streamlit.app/)
- The branch-specific budget constraints were not entirely reflective of real-world scenarios (e.g., art majors typically have different budgets than engineering majors), partly due to the ILP model's requirement for less-than-or-equal/greater-than-or-equal equations for optimization
- My lack of expertise in machine learning prevented the development of a model to automatically determine optimal values for the decline constants. This is an area where collaboration with an ML engineer could be beneficial
- For this project, I utilised averages and generalised values to define the model parameters. Real-world scenarios are significantly more complex, involving numerous dynamic factors and potentially requiring additional constraints that could impact the model's usability
