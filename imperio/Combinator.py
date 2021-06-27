'''
Created with love by Sigmoid
​
@Author - Păpăluță Vasile - papaluta.vasile@isa.utm.md
'''
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_index : list =  None, num_index : list = None, cat_transformer = None, num_transformer = None) -> None:
        '''
            The combinator constructor.
        :param cat_index: list, default = None
            The list with indexes of categorical columns.
        :param num_index: list, default = None
            The list with the indexes of numerical indexes.
        :param cat_transformer: sklearn or imperio transformer
            The sklearn or imperio transformer to apply on categorical columns.
        :param num_transformer: sklearn or imperio transformer
            The sklearn or imperio transformer to apply on numerical columns.
        '''
        # Setting up the categorical and numerical class.
        self.cat_index = cat_index
        self.num_index = num_index

        # Checking the categorical transformer.
        if issubclass(cat_transformer.__class__, TransformerMixin):
            self.cat_transformer = cat_transformer
            self.cat_transformer.index = self.cat_index
        elif cat_transformer is None:
            pass
        else:
            raise ValueError("The categorical transformer isn't a scikit-learn or imperio transformer")

        # Checking the numerical transformer.
        if issubclass(num_transformer.__class__, TransformerMixin):
            self.num_transformer = num_transformer
            self.num_transformer.index = self.num_index
        elif num_transformer is None:
            pass
        else:
            raise ValueError("The categorical transformer isn't a scikit-learn or imperio transformer")

    def fit(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict):
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

        # Checking if any list of indexes for the categorical columns was passed.
        if self.cat_index:
            # Checking if indexes passed as categorical ones are relevant.
            if min(self.cat_index) >= 0 and max(self.cat_index) < len(X[0]):
                self.cat_transformer.fit(X.copy(), y)
            else:
                # Raising an error if indexes aren't relevant.
                raise ValueError("The categorical index list has a value lower than 0 or higher that the number of features")

        # Checking if any list of indexes for the numerical columns was passed.
        if self.num_index:
            # Checking if indexes passed as numerical ones are relevant.
            if min(self.num_index) >= 0 and max(self.num_index) < len(X[0]):
                self.num_transformer.fit(X.copy(), y)
            else:
                # Raising an error if indexes aren't relevant.
                raise ValueError("The numerical index list has a value lower than 0 or higher that the number of features")
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
        # Copying the data passed.
        X_copy = X.copy()

        # Checking if the new array passed to be transformed has the same number of column as the one fitted on
        if self.shape[1] == X.shape[1]:
            # Checking if any list of indexes for the categorical columns was passed.
            if self.cat_index:
                X_copy = self.cat_transformer.transform(X_copy)

            # Checking if any list of indexes for the numerical columns was passed.
            if self.num_index:
                X_copy = self.num_transformer.transform(X_copy)

            # Returning the transformed array.
            return X_copy
        else:
            raise ValueError(f'Was passed an array with {X.shape[1]} feature, while where required {self.shape[1]} features')

    def apply(self, df : 'pd.DataFrame', target : str, cat_columns : list = None, num_columns : list = None) -> 'pd.DataFrame':
        '''
            This function allows applying the transformer on certain columns of a data frame.
        :param df: pandas DataFrame
            The pandas DataFrame on which the transformer should be applied.
        :param target: str
            The name of the column that is the target for the prediction.
        :param cat_columns: list, default = None
            The names of the columns that are selected as categorical columns.
        :param num_columns: list, default = None
            The names of the columns that are selected as numerical columns.
        :return: pd.DataFrame
            The new changed data frame.
        '''
        # Calculating the intersection between the categorical and numerical columns.
        intersection = set(cat_columns).intersection(set(num_columns))

        # Calculating the difference between the categorical columns set and the total list of columns.
        cat_columns_difference = set(cat_columns) - set(df.columns)

        # Calculating the difference between the numerical columns set and the total list of columns.
        num_columns_difference = set(num_columns) - set(df.columns)

        # Checking if a column is in both numerical and categorical list of features.
        if len(intersection) > 0:
            raise ValueError(f"Columns: {', '.join(list(intersection))} are present in the categorical and numerical classes at the same time")

        # Checking if a categorical column isn't present in data frame.
        elif len(cat_columns_difference) > 0:
            raise ValueError(f"Columns: {', '.join(list(cat_columns_difference))} aren't columns of the passed data frame")

        # Checking if a numerical column isn't present in data frame.
        elif len(num_columns_difference) > 0:
            raise ValueError(f"Columns: {', '.join(list(cat_columns_difference))} aren't columns of the passed data frame")

        # Checking if the target column is in the categorical or numerical columns list.
        elif target in cat_columns or target in num_columns:
            raise ValueError("Can't change the target column")
        else:
            # Getting the list of columns.
            columns_list = list(df.columns)

            # Getting the indices of categorical and numerical columns.
            self.cat_index, self.num_index = [], []
            if cat_columns:
                for col in cat_columns:
                    self.cat_index.append(columns_list.index(col))
            if num_columns:
                for col in num_columns:
                    self.num_index.append(columns_list.index(col))

            # Getting the X and y arrays.
            X = df.drop(target, axis=1).values
            y = df[target].values

            # Transforming the numpy arrau.
            new_X = self.fit_transform(X, y)

            # Creating the new data frame.
            new_df = pd.DataFrame(new_X, columns=[column for column in columns_list if column != target])
            new_df[target] = y

            return new_df