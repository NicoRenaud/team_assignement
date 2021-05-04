from __future__ import print_function
import time
from ortools.linear_solver import pywraplp
import numpy as np
import names
from collections import OrderedDict
import pandas as pd
from string import digits


def create_random_dataframe():
    """Create a random data frame.

    Returns:
        pd.DataFrame: dataframe
    """
    # number and names of teams
    nteam = 4
    teams = ['Natural Sciences & Engineering', 'Life Sciences',
             'Environment & Sustainability', 'Social Sciences & Humanities']

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


def create_dataframe(path_file, cost_first_choice=10, cost_second_choice=20, default_cost=100, only_engineer=False, debug=False):
    """Creates the dataframe from the excel sheet

    Args:
        path_file (str): excel file
    """

    def get_name(worker):
        "Get the full name of the worker"
        name, fam, part = worker[0], worker[1], worker[2]
        if not isinstance(part, str):
            return ' '.join([name, fam])
        else:
            return ' '.join([name, part, fam])

    def get_pref(worker, name):

        pref = [default_cost]*4
        first_choice = worker[6]
        second_choice = worker[7]

        if first_choice in sections:
            pref[sections[first_choice]] = cost_first_choice
        else:
            if debug:
                print(name, ' 1st choice (',
                      first_choice, ') not recognized')

        if second_choice in sections:
            pref[sections[second_choice]] = cost_second_choice
        else:
            if debug:
                print(name, ' 2nd choice (',
                      second_choice, ') not recognized')

        return pref

    sections = OrderedDict({'Natural Sciences & Engineering': 0,
                            'Life Sciences': 1,
                            'Environment & Sustainability': 2,
                            'Social Sciences & Humanities': 3})

    # import from excel
    all_data = pd.read_excel(path_file).to_numpy()
    workers_data = []

    for d in all_data:
        if only_engineer and 'Engineer' not in d[3]:
            continue
        data = []
        name = get_name(d)
        data.append(name)
        data += get_pref(d, name)
        workers_data.append(data)

    # return DF
    return pd.DataFrame(workers_data, columns=['Name']+list(sections.keys()))


def modify(df, name, section, cost):
    """change the cost in name/section

    Args:
        df (pandas.DataFrame) : dataframe
        name ([str]): name of the worker
        section (str): name of the section
        cost (float): new cost
    """

    # get index
    idx = df[df["Name"] == name].index.tolist()[0]
    df.at[idx, section] = cost
    return df


def solve(df, team_size_max=15, debug=False):
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

    team_compo = {t: ['']*team_size_max for t in teams}
    for i in range(nteam):
        if debug:
            print('\n=== ', teams[i], ": ")
        iw = 1
        for j in range(nworker):
            if x[i, j].solution_value() > 0:
                if debug:
                    print('\t', fmt_idx.format(idx=iw), '.', fmt_name.format(name=name[j]), '\tCost = ', cost[i]
                          [j], '\t', cost_array[:, j])
                team_compo[teams[i]][iw-1] = name[j] + \
                    ' ' + str(cost[i][j])
                iw += 1
    return pd.DataFrame(team_compo), solver.Objective().Value()


def estimate_cost(team, pref):
    """Estiamte the cost of a given team compo."""
    total_cost = 0
    for team_name in list(team.columns):
        for eng in team[team_name]:
            if isinstance(eng, str):
                eng_name = eng.translate(
                    str.maketrans('', '', digits)).strip()
                cost = pref[pref['Name'] ==
                            eng_name][team_name].values
                if len(cost) > 0:
                    total_cost += cost
    print('Total cost = ', total_cost)
    return total_cost


if __name__ == '__main__':
    df = create_random_dataframe()
    df = create_dataframe('total_overview_fixed_lars.xlsx')
    team, cost = solve(df, team_size_max=15)
