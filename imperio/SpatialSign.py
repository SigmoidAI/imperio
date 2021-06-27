'''
Created with love by Sigmoid

@Author - Nichita Novitchi - novitchi.nichita@isa.utm.md
'''

# Importing all libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

class SpatialSignTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, index : list = None) -> None:
        '''
            The SpatialSignTransformer constructor.
        :param index: list, default = None
            A parameter that specifies the list of indexes of categorical columns that should be transformed.
        '''
        self.index = index
        
    def fit(self, X: 'np.array', y: 'np.array', **fit_params : dict) -> 'SpatialSignTransformer':
        '''
            The fit function of the SpatialSignTransformer, fits up the model.
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: SpatialSignTransformer
            The fitter SpatialSignTransformer object.
        '''
        # Saving the shape of the train data.
        self.shape = X.shape
        return self
             
    def transform (self, X : 'np.array', **fit_params : dict) -> 'np.array':
        '''
            The transform function of the SpatialSignTransformer, transforms the passed data..
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        # Making a copy of the data set.
        X_copy = X.copy()

        # Transformation is applied on the feature matrix.
        if self.shape[1] == X.shape[1]:
            # If list of indexes is present then the transformation is applied to the passed columns.
            if self.index:
                X_copy[:, self.index] = normalize(X[:, self.index], norm='l2')
                return X_copy
            else:
                # Else the transformation is applied to the whole matrix.
                return normalize(X_copy, norm='l2')
        else:
            raise ValueError(
                f'Was passed an array with {X.shape[1]} feature, while where required {self.shape[1]} features')

    def fit_transform(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict) -> 'np.array':
        '''
            The transform function of the SpatialSignTransformer, transforms the passed data..
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame, default = None
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        # Fitting and applying the transformation on the same matrix.
        return self.fit(X, y).transform(X)

    def apply(self, df: 'pd.DataFrame', target: str, columns: list = None) -> 'pd.DataFrame':
        '''
            This function allows applying the transformer on certain columns of a data frame.
        :param df: pandas DataFrame
            The pandas DataFrame on which the transformer should be applied.
        :param columns: list
            The list if the names of columns on which the transformers should be applied.
        :param target: str
            The target column.
        :return: pandas DataFrame
            The new pandas DataFrame with transformed columns.
        '''
        # Checking if columns aren't set as None.
        if columns is not None:
            # Checking if passed columns exist in the passed DataFrame.
            columns_difference = set(columns) - set(df.columns)
            if len(columns_difference) != 0:
                raise ValueError(
                    f"Columns {', '.join(list(columns_difference))} are not present in the passed data frame")
            elif target in columns:
                raise ValueError(f"Target column {target} was passed as a feature name")
            else:
                # Setting upe the categorical index
                columns_list = list(df.columns)
                self.index = []
                for col in columns:
                    self.index.append(columns_list.index(col))
                # Removing the target name from the columns list.
                columns_list.remove(target)
        else:
            # Else we are applying the transformation on the whole data frame.
            columns_list = list(df.columns)
            columns_list.remove(target)
            self.index = None
        # Transforming the data frame.
        df_copy = df.copy()
        df_copy[columns_list] = self.fit_transform(df_copy[columns_list].values)
        df_copy[target] = df[target]
        return df_copy