import streamlit as st
import pulp as lp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====================
# MODEL (COPY YOUR OPTIMIZATION CODE HERE)
# ====================
# Paste your full optimization model code here.
# ====================
# MODEL
# ====================
workers = ["Worker1", "Worker2", "Worker3"]  # Define your workers
tasks = ["Task1", "Task2", "Task3"]  # Define your tasks
days = [1, 2, 3]  # Define your days

# Define maximum regular hours per day
max_regular_hours = 8

# Define worker availability (example data)
availability = {
    ("Worker1", 1): 1,
    ("Worker1", 2): 1,
    ("Worker1", 3): 1,
    ("Worker2", 1): 1,
    ("Worker2", 2): 1,
    ("Worker2", 3): 1,
    ("Worker3", 1): 1,
    ("Worker3", 2): 1,
    ("Worker3", 3): 1,
}

# Define task skill requirements (example data)
task_skill_req = {
    "Task1": {"Skill1": 1},
    "Task2": {"Skill2": 1},
    "Task3": {"Skill3": 1},
}

# Define worker skills (example data)
worker_skills = {
    "Worker1": ["Skill1"],
    "Worker2": ["Skill2"],
    "Worker3": ["Skill3"],
}

# Define maximum overtime hours per day
max_overtime = 4

# Define maximum workers per task
max_workers_per_task = 3  # Example value, adjust as needed

model = lp.LpProblem("Construction_Workforce_Optimization", lp.LpMinimize)

# Decision Variables
x = lp.LpVariable.dicts("Assign",
                        [(i, j, t) for i in workers for j in tasks for t in days],
                        cat="Binary")  # Assign worker i to task j on day t
y = lp.LpVariable.dicts("Hire",
                        [(j, t) for j in tasks for t in days],
                        lowBound=0, cat="Integer")  # Temp workers hired for task j on day t
o = lp.LpVariable.dicts("Overtime",
                        [(i, t) for i in workers for t in days],
                        cat="Binary")  # Overtime for worker i on day t
d = lp.LpVariable.dicts("Delay", tasks, lowBound=0, cat="Integer")  # Delay for task j

# Auxiliary variable for total hours worked by worker i on day t
z = lp.LpVariable.dicts("TotalHours",
                        [(i, t) for i in workers for t in days],
                        lowBound=0, cat="Continuous")

# Auxiliary variable for the product of x[i, j, t] and z[i, t]
w = lp.LpVariable.dicts("WorkedHours",
                        [(i, j, t) for i in workers for j in tasks for t in days],
                        lowBound=0, cat="Continuous")

# Auxiliary variable for the product of x[i, j, t] and o[i, t]
v = lp.LpVariable.dicts("OvertimeAssignment",
                        [(i, j, t) for i in workers for j in tasks for t in days],
                        lowBound=0, cat="Continuous")

# ====================
# OBJECTIVE FUNCTION
# ====================
# Define hourly wages for each worker
hourly_wage = {
    "Worker1": 20,
    "Worker2": 25,
    "Worker3": 30
}

# Minimize: Labor Cost + Hiring Cost + Overtime Cost + Delay Penalty
labor_cost = lp.lpSum(z[i, t] * hourly_wage[i] for i in workers for t in days)
hire_cost_total = lp.lpSum(y[j, t] * hire_cost for j in tasks for t in days)
overtime_cost = lp.lpSum(o[i, t] * hourly_wage[i] * max_overtime * (overtime_multiplier - 1)
                         for i in workers for t in days)
delay_penalty_total = lp.lpSum(d[j] * delay_penalty for j in tasks)

model += labor_cost + hire_cost_total + overtime_cost + delay_penalty_total

# ====================
# CONSTRAINTS
# ====================
# Linearization of v[i, j, t] = x[i, j, t] * o[i, t]
for i in workers:
    for j in tasks:
        for t in days:
            # Ensure v[i, j, t] <= o[i, t] (if x[i, j, t] == 0, then v[i, j, t] == 0)
            model += v[i, j, t] <= o[i, t]
            # Ensure v[i, j, t] <= M * x[i, j, t] (if x[i, j, t] == 1, then v[i, j, t] == o[i, t])
            model += v[i, j, t] <= x[i, j, t]
            # Ensure v[i, j, t] >= o[i, t] - M * (1 - x[i, j, t]) (if x[i, j, t] == 1, then v[i, j, t] == o[i, t])
            model += v[i, j, t] >= o[i, t] - (1 - x[i, j, t])

# Linearization of w[i, j, t] = x[i, j, t] * z[i, t]
for i in workers:
    for j in tasks:
        for t in days:
            # Ensure w[i, j, t] <= z[i, t] (if x[i, j, t] == 0, then w[i, j, t] == 0)
            model += w[i, j, t] <= z[i, t]
            # Ensure w[i, j, t] <= M * x[i, j, t] (if x[i, j, t] == 1, then w[i, j, t] == z[i, t])
            model += w[i, j, t] <= max_regular_hours * x[i, j, t]
            # Ensure w[i, j, t] >= z[i, t] - M * (1 - x[i, j, t]) (if x[i, j, t] == 1, then w[i, j, t] == z[i, t])
            model += w[i, j, t] >= z[i, t] - max_regular_hours * (1 - x[i, j, t])

for i in workers:
    for t in days:
        # Define z[i, t] as the total hours worked by worker i on day t
        model += z[i, t] == lp.lpSum(x[i, j, t] * max_regular_hours + v[i, j, t] * max_overtime for j in tasks)
       
        # Max hours per day (regular + overtime)
        model += z[i, t] <= max_regular_hours + o[i, t] * max_overtime
       
        # Overtime allowed only if assigned work
        model += o[i, t] <= lp.lpSum(x[i, j, t] for j in tasks)
       
        # Availability constraint
        for j in tasks:
            model += x[i, j, t] <= availability.get((i, t), 0)

for j in tasks:
    for t in days:
        # Skill requirement: At least req workers with skill s
        for s, req in task_skill_req[j].items():
            model += lp.lpSum(x[i, j, t] for i in workers if s in worker_skills[i]) + y[j, t] >= req
       
        # Max workers per task (permanent + temp)
        model += lp.lpSum(x[i, j, t] for i in workers) + y[j, t] <= max_workers_per_task

# Task dependency: Plumbing can't start before Electrical finishes (example)
model += lp.lpSum(x[i, "Plumbing", t] for i in workers for t in days) <= lp.lpSum(
    x[i, "Electrical", t] for i in workers for t in days
)

# ====================
# SOLVE & RESULTS
# ====================
model.solve()

print(f"Status: {lp.LpStatus[model.status]}")
print("Total Cost: $", round(lp.value(model.objective), 2))

# Print assignments
for t in days:
    print(f"Day {t}:")
    for j in tasks:
        assigned = [i for i in workers if lp.value(x[i, j, t]) >= 0.99]
        temps = lp.value(y[j, t])
        print(f"  {j}:")
        print(f"    Permanent: {', '.join(assigned) if assigned else 'None'}")
        print(f"    Temporary: {int(temps)} workers")

# Print overtime and delays
print("\nOvertime:")
for i in workers:
    for t in days:
        if lp.value(o[i, t]) >= 0.99:
            print(f"  {i} worked overtime on Day {t}")

print("\nDelays:")
for j in tasks:
    print(f"  {j}: {int(lp.value(d[j]))} days")
# Ensure the `model.solve()` function is called at the end of the model definition.

# ====================
# STREAMLIT DASHBOARD
# ====================

# Title
st.title("Construction Workforce Optimization Dashboard")

# Sidebar for Inputs
st.sidebar.header("Input Parameters")
hire_cost = st.sidebar.number_input("Hiring Cost ($/day)", value=70, min_value=0)
overtime_multiplier = st.sidebar.number_input("Overtime Multiplier", value=1.5, min_value=1.0)
delay_penalty = st.sidebar.number_input("Delay Penalty ($/day)", value=200, min_value=0)

# Solve Button
if st.sidebar.button("Run Optimization"):
    # Solve the model
    model.solve()

    # Display Status
    st.subheader("Optimization Status")
    st.write(f"Status: {lp.LpStatus[model.status]}")
    st.write(f"Total Cost: ${round(lp.value(model.objective), 2)}")

    # Prepare Data for Visualization
    assignments = []
    overtime_data = []
    delays = []

    for t in days:
        for j in tasks:
            assigned = [i for i in workers if lp.value(x[i, j, t]) >= 0.99]
            temps = lp.value(y[j, t])
            assignments.append({"Day": t, "Task": j, "Permanent Workers": ", ".join(assigned) if assigned else "None", "Temporary Workers": int(temps)})
        for i in workers:
            if lp.value(o[i, t]) >= 0.99:
                overtime_data.append({"Worker": i, "Day": t})
    for j in tasks:
        delays.append({"Task": j, "Delay (days)": int(lp.value(d[j]))})

    # Convert to DataFrames
    assignments_df = pd.DataFrame(assignments)
    overtime_df = pd.DataFrame(overtime_data)
    delays_df = pd.DataFrame(delays)

    # Visualizations
    st.subheader("Assignments Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    for task in tasks:
        task_data = assignments_df[assignments_df["Task"] == task]
        ax.bar(task_data["Day"], task_data["Temporary Workers"], label=f"{task} (Temp)")
        ax.bar(task_data["Day"], [len(w.split(",")) if w != "None" else 0 for w in task_data["Permanent Workers"]], bottom=task_data["Temporary Workers"], label=f"{task} (Perm)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Workers")
    ax.set_title("Worker Assignments by Task and Day")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Overtime Analysis")
    if not overtime_df.empty:
        overtime_counts = overtime_df.groupby("Worker").size().reset_index(name="Overtime Days")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(overtime_counts["Worker"], overtime_counts["Overtime Days"], color="orange")
        ax.set_xlabel("Worker")
        ax.set_ylabel("Overtime Days")
        ax.set_title("Overtime Days per Worker")
        st.pyplot(fig)
    else:
        st.write("No overtime recorded.")

    st.subheader("Task Delays")
    if not delays_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(delays_df["Task"], delays_df["Delay (days)"], color="red")
        ax.set_xlabel("Task")
        ax.set_ylabel("Delay (days)")
        ax.set_title("Delays by Task")
        st.pyplot(fig)
    else:
        st.write("No delays recorded.")

    # Detailed Tables
    st.subheader("Detailed Assignments")
    st.dataframe(assignments_df)

    st.subheader("Overtime Details")
    if not overtime_df.empty:
        st.dataframe(overtime_df)
    else:
        st.write("No overtime recorded.")

    st.subheader("Delays Details")
    if not delays_df.empty:
        st.dataframe(delays_df)
    else:
        st.write("No delays recorded.")
