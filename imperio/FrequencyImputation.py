'''
Created with love by Sigmoid

@Author - Clefos Alexandru - clefos.alexandru@isa.utm.md
'''

# Importing all libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyImputationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, index='auto', min_int_freq=5):
        '''
            Setting up the algorithm
        :param index: list, default = 'auto'
            A parameter that specifies the list of indexes of categorical columns that should be transformed.
        :param min_int_freq: int, default = 5
            A parameter that indicates the number of minimal values in a categorical column for the transformer
            to be applied.
        '''
        self.index = index
        self.min_int_freq = min_int_freq

    def __categorical_check(self, X : 'np.array') -> 'np.array':
        '''
            This function converts matrix to integer.
        :param X: np.array
            The matrix to convert.
        :return: np.array
            The matrix converted to X.
        '''
        return X.astype('int64')

    def fit(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict) -> 'FrequencyImputationTransformer':
        '''
            Fit function
        :param X: 2-d numpy array
            A parameter that stores the data set without the target vector.
        :param y: 1-d numpy array, default = None
            A parameter that stores the target vector.
        :param fit_params: dict
            Additional fit parameters.
        :return: FrequencyImputationTransformer
            The fitted transformer.
        '''
        # Saving the fitting array shape.
        self.shape = X.shape

        # Creating a dictionary with all mappers.
        self.__mappers = dict()

        # Defining the list with categorical data types.
        categorical = ['int0', 'int8', 'int16', 'int32', 'int64']

        # Checking the value of __categorical_index.
        if self.index == 'auto':
            # If index is set as default, transformer finds all the categorical columns
            # for which mapper should be created.
            for i in range(len(X[0])):
                if str(X[0, i].dtype) in categorical and len(set(X[:, i])) < self.min_int_freq and len(set(X[:, i])) != 2:
                    self.__mappers[i] = dict()

                    # Mapping out all the categorical values in the selected column.
                    for element in np.unique(X[:, i]):
                        self.__mappers[i][element] = len(X[:, i][np.where(X[:, i] == element)]) / len(X)
        else:
            # If index is set by the user, transformer iterates through the passed list of indexes.
            for i in self.index:
                if i >= len(X[0]) or i < 0:
                    raise ValueError('Passed index list contains a invalid index!')
                self.__mappers[i] = dict()

                # Mapping out all the categorical values in the selected column.
                for element in np.unique(X[:, i]):
                    self.__mappers[i][element] = len(X[:, i][np.where(X[:, i] == element)]) / len(X)

        # Returning the fitted instance of FrequencyImputationTransformer class.
        return self

    def transform(self, X : 'np.array', **fit_params : dict) -> 'np.array':
        '''
            Function that transforms the new given data
        :param X: 2-d numpy array
            A parameter that stores the data set without the target vector.
        :param fit_params: dict
            Additional fit parameters.
        :return: 2-d numpy array
            The transformed 2-d numpy array.
        '''
        X_new = X.copy()
        # Converting the input matrix to int data type.
        X_new[:, list(self.__mappers.keys())] = self.__categorical_check(X_new[:, list(self.__mappers.keys())])

        # Checking if the new array passed to be transformed has the same number of column as the one fitted on
        if self.shape[1] == X.shape[1]:
            for key in self.__mappers:
                # If a new value is passed than it is replaced with 0.
                X_new[:, key] = [self.__mappers[key][value] if value in self.__mappers[key] else 0 for value in
                                 X_new[:, key]]
            return X_new.astype('float')
        else:
            raise ValueError(
                f'Was passed an array with {X.shape[1]} feature, while where required {self.shape[1]} features')

    def fit_transform(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict) -> 'np.array':
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
            self.index = 'auto'
        # Transforming the data frame.
        df_copy = df.copy()
        df_copy[columns_list] = self.fit_transform(df_copy[columns_list].values)
        df_copy[target] = df[target]
        return df_copy