'''
Created with love by Sigmoid
​
@Author - Păpăluță Vasile - papaluta.vasile@isa.utm.md
'''
# Importing all needed libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, l : float = 0.5, index = None) -> None:
        '''
            The constructor of the BoxCoxTransformer class.
        :param l: float, default = 0.5
            The lambda parameter used by Box-Cox Algorithm to choose the transformation applied on the data.
        '''
        self.l = l
        self.index = index

    def fit(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict) -> 'BoxCoxTransformer':
        '''
            The fit function of the BoxCoxTransformer, fits up the model.
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: BocCoxTransformer
            The fitter BoxCoxTransformer object.
        '''
        # Saving the fitting array shape.
        self.shape = X.shape
        return self

    def __apply_change(self, value : float) -> float:
        '''
            The functions that transforms a scalar value.
        :param value: float
            The value that should be transformed.
        :return: float
            The transformed value by BoxCoxTransformer.
        '''
        return np.log(value) if self.l == 0 else (self.l ** (-1)) * (value ** self.l - 1)

    def transform(self, X : 'np.array',  **fit_params : dict) -> 'np.array':
        '''
            The transform function of the BoxCoxTransformer, transforms the passed data..
        :param X: 2-d numpy array
            The 2-d numpy array that represents the feature matrix.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        if self.shape[1] == X.shape[1]:
            if 'target' in fit_params:
                # If target is passed as a argument then, if it is set to True then y array is changed.
                if fit_params['target']:
                    return np.array([self.__apply_change(yi) for yi in y])
                else:
                    # If passed argument target is passed as False then transformation is applyed on the feature matrix.
                    return np.array([np.array([self.__apply_change(xi) for xi in X[:, i]]) for i in range(len(X[0]))])
            else:
                if self.index:
                    # If target isn't passed as a argument then transformation is applyed on the feature matrix.
                    X_copy = X.copy()
                    for i in self.index:
                        if i >= len(X[0]) or i < 0:
                            raise ValueError('Passed index list contains a invalid index!')
                        X_copy[:, i] = self.__apply_change(X_copy[:, i])
                    return X_copy
                else:
                    return np.array([self.__apply_change(xi) for xi in X])
        else:
            raise ValueError(
                f'Was passed an array with {X.shape[1]} feature, while where required {self.shape[1]} features')

    def fit_transform(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict):
        '''
            The transform function of the BoxCoxTransformer, transforms the passed data..
        :param X: 2-d numpy array
            The 2-d numpy array that represents the feature matrix.
        :param y: 1-d numpy array, default = None
            The 1-d numpy array that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        if 'target' in fit_params:
            return self.fit(X, y).transform(X, target = fit_params['target'])
        else:
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