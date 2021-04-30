from __future__ import print_function
import time
from ortools.linear_solver import pywraplp
import numpy as np
import names
from collections import OrderedDict
import pandas as pd


def create_random_dataframe():
    """Create a random data frame.

    Returns:
        pd.DataFrame: dataframe
    """
    # number and names of teams
    nteam = 4
    teams = ['Exact Science', 'Life Science',
             'Environement', 'Humanities']

    # probability of each team preference
    pteams = [0.5, 0.2, 0.2, 0.1]

    # number of worker name and preference
    nworker = 47
    workers_data = []

    # create the array
    for i in range(nworker):
        data = []
        data.append(names.get_full_name())
        pref = 10*np.random.choice(4, 4, p=pteams,
                                   replace=False)

        data += pref.tolist()
        workers_data.append(data)

    # return DF
    return pd.DataFrame(workers_data, columns=['Name']+teams)


def solve(df, team_size_max=15):
    """Solve the assignement problem

    Args:
        df (pd.DataFrame): input data
        team_size_max (int, optional): max size of the teams. Defaults to 15.
    """

    # init the solver
    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # convert the dataframe
    data = df.to_numpy()
    name = data[:, 0]
    cost = data[:, 1:].T

    # size of the problem
    nworker = len(name)
    nteam = cost.shape[0]

    # save a np copy and transform to list
    cost_array = cost
    cost = cost.tolist()

    # size of a worker i.e. 1 slot in the team
    worker_size = [1]*nworker

    # Maximum total of worker per team
    teams = df.columns.to_list()[1:]

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
    fmt_name = "{name:25s}"
    fmt_idx = "{idx: 3d}"
    for i in range(nteam):
        print('\n=== ', teams[i], ": ")
        iw = 1
        for j in range(nworker):
            if x[i, j].solution_value() > 0:
                print('\t', fmt_idx.format(idx=iw), '.', fmt_name.format(name=name[j]), '\tCost = ', cost[i]
                      [j], '\t', cost_array[:, j])
                iw += 1


if __name__ == '__main__':
    df = create_random_dataframe()
    solve(df, team_size_max=15)
