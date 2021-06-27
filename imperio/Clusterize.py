'''
Created with love by Sigmoid
​
@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from .erorrs import NotAClusteringAlgorithm

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

class ClusterizeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, algorithm, index : list = None) -> None:
        '''
        Constructor of the Clusterize class
        :param algorithm: Object
            Algorith that will do the clusterization
        :param column_index: list, default = None
            A parameter that specifies the list of indexes of columns that should be transformed.
        '''
        self.index = index

        if issubclass(algorithm.__class__, sklearn.base.ClusterMixin):
            self.algorithm = algorithm
        else:
            raise NotAClusteringAlgorithm("Wasn't passed a clustering algorithm")

    def fit(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict):
        '''
            The fit function of the MeanEncoding, fits up the model.
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: MeanEncoding
            The fitter MeanEncoding object.
        '''
        #Fitting algorithm on data
        self.shape = X.shape
        if self.index:
            self.algorithm.fit(X[:, self.index])
        else:
            self.algorithm.fit(X)
        return self

    def transform(self, X: 'np.array', **fit_params) -> 'np.array':
        '''
            The transform function of the MeanEncoding, transforms the passed data..
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        #get predicted clusters and put them as a new column
        if self.shape[1] == X.shape[1]:
            self.X_copy = X.copy()
            if self.index:
                cluster = self.algorithm.predict(self.X_copy[:,self.index])
            else:
                cluster = self.algorithm.predict(self.X_copy)

            new_X = np.column_stack([self.X_copy, cluster])
        else:
            raise ValueError(f'Was passed an array with {X.shape[1]} features, while where required {self.shape[1]} features')
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        '''
            Function that fits and transform the data
        :param X: 2-d numpy array
            A parameter that stores the data set without the target vector.
        :param y: 1-d numpy array
            A parameter that stores the target vector.
        :param fit_params: dict
            Additional fit parameters.
        :return: 2-d numpy array
            The transformed 2-d numpy array.
        '''
        return self.fit(X).transform(X)

    def apply(self, df : 'pd.DataFrame', target: str, columns : list = None) -> 'pd.DataFrame':
        '''
            This function allows applying the transformer on certain columns of a data frame.
        :param df: pandas DataFrame
            The pandas DataFrame on which the transformer should be applied.
        :param target: string
             The target name of the value that will be predicted
        :param columns: list
            The list if the names of columns on which the transformers should be applyed.
        :return: pandas DataFrame
            The new pandas DataFrame with transformed columns.
        '''
        # Checking if columns aren't set as None.
        if columns is not None:
            # Checking if passed columns exist in the passed DataFrame.   
            columns_difference = set(columns) - set(df.columns)
            if len(columns_difference) != 0:
                raise ValueError(f"Columns {', '.join(list(columns_difference))} are not present in the passed data frame")
            elif target in columns:
                raise ValueError(f"Target column {target} was passed as a feature name")
            else:
                # Setting upe the categorical index
                columns_list = list(df.columns)
                self.index = []
                for col in columns:
                    self.index.append(columns_list.index(col))
                # Removing the target name from the columns list.v
                columns_list.remove(target)
        else:
            columns_list = list(df.columns)
            columns_list.remove(target)
            self.index = None
        # Transforming the data frame.
        df_copy = df.copy()
        df_copy['Cluster'] = 0
        columns_list_new = columns_list.copy()
        columns_list_new.append('Cluster')
        df_copy[columns_list_new] = self.fit_transform(df_copy[columns_list].values)
        df_copy[target] = df[target]
        return df_copy