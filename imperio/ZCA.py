'''
Created with love by Sigmoid
​
@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

class ZCATransformer(BaseEstimator, TransformerMixin):

    def __init__(self, index : list = None, eps : float = 10e-5) -> None:
        '''
        Constructor of the ZCA class
        :param column_index: list, default = None
            A parameter that specifies the list of indexes of columns that should be transformed.
        '''
        self.index = index
        self.eps = eps

    def fit(self, X : 'np.array', y : 'np.array', **fit_params : dict):
        '''
            The fit function of the ZCA, fits up the model.
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: BocCoxTransformer
            The fitter ZCA object.
        '''
        self.shape = X.shape
        return self

    def transform(self, X : 'np.array', **fit_params : dict):
        '''
            The transform function of the ZCA, transforms the passed data..
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        if self.shape[1] == X.shape[1]:

            #Prepare data to modify
            self.X_copy = X.copy()
            if self.index is not None:
                self.X_copy = self.X_copy[:,self.index]

            #   covariance matrix
            cov = np.dot(self.X_copy.T, self.X_copy)
            #   d = (lambda1, lambda2, ..., lambdaN)
            d, E = np.linalg.eigh(cov)
            #   D = diag(d) ^ (-1/2)
            D = np.diag(1. / np.sqrt(d + self.eps))
            #   W_zca = E * D * E.T
            W = np.dot(np.dot(E, D), E.T)

            X_white = np.dot(self.X_copy, W)
        else:
            raise ValueError(f'Was passed an array with {X.shape[1]} features, while where required {self.shape[1]} features')

        return X_white

    def fit_transform(self, X, y=None, **fit_params):
        '''
            The transform function of the ZCA, transforms the passed data..
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame, default = None
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        return self.fit(X, y).transform(X)
    
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
        df_copy[columns_list] = self.fit_transform(df_copy[columns_list].values)
        df_copy[target] = df[target]
        return df_copy