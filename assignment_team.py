from __future__ import print_function
import time
from ortools.linear_solver import pywraplp
import numpy as np
import names
from collections import OrderedDict


def main():
    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # number and names of teams
    nteam = 4
    teams = ['Exact Science', 'Life Science',
             'Environement', 'Humanities']

    # probability of each team preference
    pteams = [0.6, 0.15, 0.15, 0.1]

    # number of worker name and preference
    nworker = 40
    name, pref = [], []
    for i in range(nworker):
        name.append(names.get_full_name())
        pref.append(np.random.choice(4, 4, p=pteams, replace=False))

    # create the cost matrix
    cost = np.zeros((nteam, nworker))
    for iw in range(nworker):
        cost[:, iw] = pref[iw]

    # save a np copy and transform to list
    cost_array = cost
    cost = cost.tolist()

    # size of a worker i.e. 1 slot in the team
    worker_size = [1]*nworker

    # Maximum total of worker per team
    team_size_max = 10

    # Variables
    x = {}

    for i in range(nteam):
        for j in range(nworker):
            x[i, j] = solver.IntVar(0, 1, 'x[%i,%i]' % (i, j))

    # Constraints
    # The total size of the tasks each worker takes on is at most team_size_max.
    for i in range(nteam):
        solver.Add(solver.Sum([worker_size[j] * x[i, j]
                               for j in range(nworker)]) <= team_size_max)

    # Each task is assigned to at least one worker.

    for j in range(nworker):
        solver.Add(solver.Sum([x[i, j]
                               for i in range(nteam)]) >= 1)

    solver.Minimize(solver.Sum([cost[i][j] * x[i, j] for i in range(nteam)
                                for j in range(nworker)]))
    sol = solver.Solve()

    print('Minimum cost = ', solver.Objective().Value())
    fmt_name = "{name:15s}"
    for i in range(nteam):
        print('\n=== ', teams[i])
        for j in range(nworker):
            if x[i, j].solution_value() > 0:
                print('\t', fmt_name.format(name=name[j]), '\tCost = ', cost[i]
                      [j], '\tPref =', cost_array[:, j])


if __name__ == '__main__':
    main()
