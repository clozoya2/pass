"""
Author: Christian Lozoya, 2017
"""

import numpy as np
import pandas as pd
import random as random
import re

FOLDER = 'YEAR'
FILE = 'STATE_CODE_001'
ID = 'STRUCTURE_NUMBER_008'
LATITUDE = 'LAT_016'
LONGITUDE = 'LONG_017'
ENCODING = 'latin1'
TXT_EXT = '.txt'
NA_VALUES = ['N']


class Model:
    def __init__(self, item, files, clean=True, iterations=1, initialState=1, validStates=(0, 1)):
        try:
            self.iS = np.transpose((pd.Series(initialState, index=validStates)))
            self.vS = validStates
            self.data, self.yrs = Model.process_data(item=item, dir=files,
                                                     paths=read_folder(dir=files, ext=TXT_EXT),
                                                     clean=clean)
            self.sampler = None
        except:
            pass

    @staticmethod
    def process_data(item, dir, paths, clean=False):
        """
        item is the item of interest (str)
        dir is the directory where the files of data are found (str)
        paths is all the paths of data in the directory (list of str)
        clean determines whether or not incomplete data is removed
        returns a list of pandas dataFrames and a list of numerics to name the files
        """
        data = []
        yrs = []
        for i, file in enumerate(paths):
            data.append(pd.read_csv(dir + '/' + file, usecols=[ID] + [item], na_values=NA_VALUES,
                                    dtype={**{ID: str}, **{item: str}}, encoding=ENCODING))
            yrs.append(re.findall(r"[0-9]{2}(?=.txt)", file)[0])
            data[i].set_index(ID, inplace=True)
        data = concat_data(dataFrame=data, columns=yrs, index=item)
        if clean:
            data = clean_data(dataFrame=data, columns=yrs)
        return data, yrs

    @staticmethod
    def count_matrix(data, matrix, columns=None, hst=False):
        """
        Count data and store the count in matrix (pre-labeled DataFrame objects)
        columns is a list of column labels
        """
        if not hst: matrix.fillna(0, inplace=True)
        for i, row in enumerate(data.iterrows()):
            for j, column in enumerate(data.loc[row[0], :]):
                if hst:
                    matrix[data.iloc[i, j]].append(int(columns[j]) - 1991)

                elif columns:  # If columns are specified: row=the value in (i,j), column=column j
                    matrix.loc[data.iloc[i, j], columns[j]] += 1

                else:  # If nothing is specified: transition matrix is assumed
                    if (j < len(data.iloc[i, :]) - 1):
                        try:
                            if (int(data.iloc[i, j + 1]) <= int(data.iloc[i, j])):
                                matrix.loc[data.iloc[i, j], data.iloc[i, j + 1]] += 1
                        except:
                            pass

        return matrix

    @staticmethod
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

    @staticmethod
    def hst(data, matrix, columns):  # TODO bad method + needs generalizing (only works for freq vs state)
        print(data)
        print(matrix)
        hst = Model.count_matrix(data=data, matrix=matrix, columns=columns, hst=True)
        hst = pd.DataFrame([hst[m] for m in hst], index=[m for m in hst])
        print(hst)
        hst.fillna(0, inplace=True)
        hst.sort_index(ascending=False, inplace=True)
        return hst.as_matrix().astype(np.float64)

    def raw_frq(self, focus):
        matrix = Model.count_matrix(self.data, pd.DataFrame(index=self.vS, columns=self.yrs), self.yrs)
        return Model.frq(focus, matrix)

    def raw_hst(self, focus):
        if focus == "year":
            return self.data.T.as_matrix().astype(np.float64)
        elif focus == "state":
            return Model.hst(data=self.data, matrix={s: [] for s in self.vS}, columns=self.yrs)


class MarkovChain(Model):
    name = "Markov Chain "

    def run(self):
        self.countMatrix = Model.count_matrix(self.data, pd.DataFrame(index=self.vS, columns=self.vS))
        self.matrix = normalize_rows(self.countMatrix)  # Transition Matrix
        self.pdf = MarkovChain.markov_chain(self.yrs, self.iS, self.matrix)
        self.sampler = MonteCarlo.MonteCarlo(self)
        self.sampler.run()

    @staticmethod
    def markov_chain(columns, initial, matrix):
        """
        initial (initial state) is multiplied by matrix (transition matrix)
        as many times as the number of columns
        """
        markovChain = [initial]
        for i in range(len(columns) - 1):
            markovChain.append(pd.Series(markovChain[i].dot(matrix)))
        return concat_data(dataFrame=markovChain, columns=columns)


class MonteCarlo(Model):
    name = "Monte Carlo "

    def run(self):
        """cS is the current state"""
        self.model = self.parent
        self.simulation = []
        self.time = len(self.model.data.columns)
        for i in range(self.model.iterations):
            self.simulation.append([])
            cS = self.model.iS
            for t in range(self.time):
                if t != 0:
                    # Multiply current state and transition matrix, resulting in a probability vector
                    pV = cS.dot(self.model.matrix)
                    cS = self.sample(self.model.vS, pV, i, cS)
                else:
                    for state, prob in zip(self.model.vS, cS):
                        if prob == 1:
                            self.simulation[i].append(state)
            self.simulation[i] = pd.Series(self.simulation[i])

    def sample(self, vS, pV, iteration, cS):
        """
        vS is a list of valid state names
        pV is a probability vector
        iteration is the current iteration
        cS is the current state
        returns a state vector
        """
        sum = 0
        # Generate random number
        randomNumber = random.uniform(0, 1)
        # Assign state if randomNumber is within its range
        for state, prob in zip(vS, pV):
            sum += prob
            if (sum >= randomNumber):
                self.simulation[iteration].append(state)
                return np.transpose((pd.Series([1 if s == state else 0 for s in vS], index=vS)))

    def sim_frq(self, focus):
        """
        Count state transitions in data and store the count in matrix (pre-labeled DataFrame objects)
        """
        matrix = Model.count_matrix(pd.DataFrame(self.simulation),
                                    pd.DataFrame(index=self.model.vS, columns=self.model.yrs), self.model.yrs)
        return Model.frq(focus, matrix)

    def sim_hst(self, focus):
        if focus == "year":
            return pd.DataFrame(self.simulation, index=[x for x in range(len(self.simulation))],
                                columns=[y for y in range(self.time)]).T.as_matrix().astype(np.float64)
        elif focus == "state":
            return Model.hst(data=pd.DataFrame(self.simulation), matrix={s: [] for s in self.model.vS},
                             columns=self.model.yrs)


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


def read_folder(dir, ext):
    """
    Collect all file paths with given extension (ext) inside a folder directory (dir)
    """
    paths = []
    for file in os.listdir(dir):
        if file[-4:].lower() == ext.lower():
            paths.append(file)
    return paths


def concat_data(dataFrame, columns, index=None):
    """
    Concatenate list of pandas Series into a DataFrame
    dataFrame is a pandas DataFrame and columns are the columns in data
    index is used for concatenating only specific columns
    """
    if index:
        cnc = pd.DataFrame(pd.concat([d[index] for d in dataFrame], axis=1))
    else:
        cnc = pd.DataFrame(pd.concat([d for d in dataFrame], axis=1))
    cnc.columns = columns
    return cnc


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
