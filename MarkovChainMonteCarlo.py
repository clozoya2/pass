"""
Author: Christian Lozoya, 2017
"""
import os
import random as random
import re

import numpy as np
import pandas as pd

FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'
TXT_EXT = '.txt'
NA_VALUES = ['N']


def raw_frq(data, vS, yrs, focus):
    matrix = count_matrix(data, pd.DataFrame(index=vS, columns=yrs), yrs)
    return frq(focus, matrix)


def raw_hst(data, vS, yrs, focus):
    if focus == "year":
        return data.T.as_matrix().astype(np.float64)
    elif focus == "state":
        return hst(data=data, matrix={s: [] for s in vS}, columns=yrs)


def markov_chain(columns, initial, matrix):
    """
    initial (initial state) is multiplied by matrix (transition matrix)
    as many times as the number of columns
    """
    markovChain = [initial]
    for i in range(len(columns) - 1):
        markovChain.append(pd.Series(markovChain[i].dot(matrix)))
    return concat_data(dataFrame=markovChain, columns=columns)


def frq(focus, matrix):
    """
    Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
    """
    matrix = matrix
    if focus == "year":
        frq = matrix.T
        frq = frq[frq.columns[::-1]]
        return normalize_rows(frq).as_matrix()
    elif focus == "state":
        return normalize_rows(matrix.T).T.as_matrix()
    else:
        print("Select focus")


def hst(data, matrix, columns):  # TODO bad method + needs generalizing (only works for freq vs state)
    print(data)
    print(matrix)
    hst = count_matrix(data=data, matrix=matrix, columns=columns, hst=True)
    hst = pd.DataFrame([hst[m] for m in hst], index=[m for m in hst])
    print(hst)
    hst.fillna(0, inplace=True)
    hst.sort_index(ascending=False, inplace=True)
    return hst.as_matrix().astype(np.float64)


def sample(simulation, vS, pV, iteration):
    """
    simulation: list
    vS: list of valid state names
    pV: probability vector
    iteration: current iteration
    returns a state vector
    """
    sum = 0
    randomNumber = random.uniform(0, 1)
    # Assign state if randomNumber is within its range
    for state, prob in zip(vS, pV):
        sum += prob
        if (sum >= randomNumber):
            simulation[iteration].append(state)
            return simulation, np.transpose((pd.Series([1 if s == state else 0 for s in vS], index=vS)))


def sim_frq(simulation, vS, yrs, focus):
    """
    Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
    """
    matrix = count_matrix(pd.DataFrame(simulation),
                          pd.DataFrame(index=vS, columns=yrs), yrs)
    return frq(focus, matrix)


def sim_hst(simulation, vS, yrs, time, focus):
    if focus == "year":
        return pd.DataFrame(simulation, index=[x for x in range(len(simulation))],
                            columns=[y for y in range(time)]).T.as_matrix().astype(np.float64)
    elif focus == "state":
        return hst(data=pd.DataFrame(simulation), matrix={s: [] for s in vS},
                   columns=yrs)


'---------------------------------------------------------------------'


def read_folder(dir, ext):
    """
    Collect all file paths with given extension (ext) inside a folder directory (dir)
    """
    paths = []
    for file in os.listdir(dir):
        if file.split('.')[-1].lower() == ext.lower():
            paths.append(file)
    return paths


def process_data(item, dir, ext, clean=False):
    """
    item is the item of interest (str)
    dir is the directory where the files of data are found (str)
    paths is all the paths of data in the directory (list of str)
    clean determines whether or not incomplete data is removed
    returns a list of pandas dataFrames and a list of numerics to name the files
    """
    data = []
    yrs = []
    paths = read_folder(dir, ext)
    for i, file in enumerate(paths):
        data.append(pd.read_csv(dir + '/' + file, usecols=[ID] + [item], na_values=NA_VALUES,
                                dtype={**{ID: str}, **{item: str}}, encoding=ENCODING))
        yrs.append(re.findall(r"[0-9]{2}(?=.txt)", file)[0])
        data[i].set_index(ID, inplace=True)
        data[i] = data[i][~data[i].index.duplicated()]
    data = concat_data(dataFrame=data, columns=yrs, index=item)
    if clean:
        data = clean_data(dataFrame=data, columns=yrs)
    return data, yrs


def concat_data(dataFrame, columns, index=None):
    """
    Concatenate list of pandas Series into a DataFrame
    dataFrame is a pandas DataFrame and columns are the columns in data
    index is used for concatenating only specific columns
    """
    if index:
        cnc = pd.concat([d[index] for d in dataFrame], axis=1)
    else:
        cnc = pd.concat([d for d in dataFrame], axis=1)
    cnc.columns = columns
    return cnc


def count_matrix(data, matrix, columns=None, hst=False):
    """
    Count data and store the count in matrix (pre-labeled DataFrame objects)
    columns is a list of column labels
    """
    data = data.astype('int8')
    if not hst:
        matrix.fillna(0, inplace=True)
    length = len(data.columns)
    for i, row in enumerate(data.iterrows()):
        for j, column in enumerate(data.loc[row[0], :]):
            if hst:
                matrix[data.iloc[i, j]].append(int(columns[j]) - 1991)
            elif columns:  # If columns are specified: row=the value in (i,j), column=column j
                matrix.loc[data.iloc[i, j], columns[j]] += 1
            else:  # If nothing is specified: transition matrix is assumed
                if (j < length - 1):
                    try:
                        if (int(data.iloc[i, j + 1]) <= int(data.iloc[i, j])):
                            matrix.loc[data.iloc[i, j], data.iloc[i, j + 1]] += 1
                    except Exception as e:
                        print(e)
    return matrix


def normalize_rows(dataFrame):
    """
    Normalize dataFrame by row
    """
    try:
        nrm = dataFrame.div(dataFrame.sum(axis=1), axis=0)
        nrm.fillna(0, inplace=True)
        return nrm
    except Exception as e:
        print(e)


def clean_data(dataFrame, columns):
    """
    Clears incomplete data resulting in regular matrix
    data is a pandas DataFrame and columns are the columns in data
    """
    cln = dataFrame
    for column in columns:
        cln = (cln[np.isfinite(cln[column].astype(float))])
    cln.columns = columns
    return cln


def get_transition_matrix(data, vS):
    return normalize_rows(count_matrix(data, pd.DataFrame(index=vS, columns=vS)))


def run_monte_carlo(data, matrix, vS, iS, iterations):
    simulation = []
    time = len(data.columns)
    for i in range(iterations):
        simulation.append([])
        cS = iS
        for t in range(time):
            if t != 0:
                # Multiply current state and transition matrix, resulting in a probability vector
                pV = cS.dot(np.linalg.matrix_power(matrix, t))
                simulation, cS = sample(simulation, vS, pV, i)
            else:
                for state, prob in zip(vS, cS):
                    if prob == 1:
                        simulation[i].append(state)
        simulation[i] = pd.Series(simulation[i])
    return simulation


dataDir = 'data'
item = 'DECK_COND_058'
clean = True
data, yrs = process_data(item=item, dir=dataDir, ext='txt', clean=clean)
vS = tuple(range(10))
iS = np.array([0 if i != 9 else 1 for i in range(10)])
Q = get_transition_matrix(data, vS)
simulation = run_monte_carlo(data, Q, vS, iS, iterations=100)
import matplotlib.pyplot as plt
c = pd.concat(simulation, axis=1).mean(axis=1)
plt.plot(c)
plt.show()
