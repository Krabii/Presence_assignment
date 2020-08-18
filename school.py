from pyomo.environ import *
import itertools
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.bdate_range('2020-09-07', '2020-09-18', freq='B')
weeks = dates.week.unique()
kids = {x: 'b' if x % 2 else 'g' for x in range(1, 25)}
pair_of_kids = [f for f in itertools.combinations(kids.keys(), 2)]

# Model Definition
model = ConcreteModel()

model.Assignments = Var(dates, kids.keys(), domain=Binary)
model.pairs = Var(dates, pair_of_kids, domain=Binary)
model.gender_balance = Var(domain=NonNegativeIntegers)
model.z = Var(domain=NonNegativeReals)
model.weekly_interaction = Var(weeks, pair_of_kids, domain=Binary)

# Maximize children's interaction
model.obj = Objective(expr=sum(model.weekly_interaction[week, pair] for week in weeks for pair in pair_of_kids) + model.gender_balance + model.z, sense=maximize)

model.constraints = ConstraintList()
# No more than 50% of kids are in school in a given day
for day in dates:
    model.constraints.add(
        sum(model.Assignments[day, kid] for kid in kids.keys()) <= 15)
    model.constraints.add(
        model.z <= sum(model.Assignments[day, kid] for kid in kids.keys()))


# Each kid must go to school at least twice in any given week but no more than 3
for kid in kids.keys():
    for week in dates.week.unique():
        model.constraints.add(sum(model.Assignments[day, kid] for day in dates if day.week == week) >= 2)
        model.constraints.add(sum(model.Assignments[day, kid] for day in dates if day.week == week) <= 3)

# Kids must interact at least once in the scheduling period
for pair in pair_of_kids:
    model.constraints.add(
        sum(model.pairs[day, pair] for day in dates) >= 1
    )

# Constraint to track the kids' interactions. If both are in school then sum == 2 then Boolean variable equals one,
# if only one or less are present then boolean variable equals zero
for day in dates:
    for pair in pair_of_kids:
        model.constraints.add(
            2 * model.pairs[day, pair] <= sum(model.Assignments[day, kid] for kid in pair)
        )

# Constraint to track weekly interaction.
# Same as above, only this time a single interaction within a week sets x variable to 1 otherwise 0
for week in weeks:
    for pair in pair_of_kids:
        model.constraints.add(
            1 * model.weekly_interaction[week, pair] <= sum(model.pairs[day, pair] for day in dates if day.week == week)
        )

# MinMax gender balance criterion
for day in dates:
    model.constraints.add(
        model.gender_balance <= sum(model.Assignments[day, kid] for kid in kids.keys() if kids[kid] == 'b')
    )


opt = SolverFactory('scip')
# results = opt.solve(model, tee=True)  # solve the model with the selected solver

# opt = SolverFactory('cplex', executable='/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux/cplex')
opt.options['ratio'] = 0.001
opt.options['sec'] = 1500
solver_manager = SolverManagerFactory('serial')  # Solve in neos server
results = solver_manager.solve(model, opt=opt, tee=True)


for i, d in enumerate(dates):
    if i % 5 == 0:
        print("")
    print(f"{d.date()} :", end=" ")
    for kid in kids.keys():
        if value(model.Assignments[d, kid]) > 0:
            print(f"{kid}", end=" ")
    print("")

print(f'Total Score is: {model.obj()}')
print(f'Total weekly interaction is: {sum(model.weekly_interaction[week, pair]() for week in weeks for pair in pair_of_kids)}')
print(f'Gender balance is: {model.gender_balance()}')
print(f'Least number of children: {model.z()}')

assinments, boys = {}, {}
for date in dates:
    assinments[str(date.date())] = sum([model.Assignments[date, kid]() for kid in kids.keys()])
    boys[str(date.date())] = sum([model.Assignments[date, kid]() for kid in kids.keys() if kids[kid] == 'b'])


fig, axs = plt.subplots(2)

axs[0].bar(list(boys.keys()), list(boys.values()))
axs[0].set_ylabel('Boys present in school')
axs[1].bar(list(assinments.keys()), list(assinments.values()))
axs[1].set_ylabel('Children present in school')
fig.suptitle('Gender equity and children in-school per day', fontsize=16)
plt.show()
