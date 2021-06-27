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

class SmoothingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, alpha :float = 100, index = None, min_int_freq : int = 10) -> None:
        '''
        Constructor of theSmoothing class
        :param alpha: int, alpha>0, default = 100
            Number that controls the amount of regularization
        :param index: list, default = 'auto'
            A parameter that specifies the list of indexes of categorical columns that should be transformed.
        :param min_int_freq: int, default = 5
            A parameter that indicates the number of minimal values in a categorical column for the transformer
            to be applied.
        '''

        self.alpha = alpha
        self.index = index
        self.min_int_freq = min_int_freq

    def fit(self, X : 'np.array', y : 'np.array' = None, **fit_params : dict) -> 'SmoothingTransformer':
        '''
            The fit function of the BoxCoxTransformer, fits up the model.
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: SmoothingTransformer
            The fitter Smoothing object.
        '''
        self.shape = X.shape
        X_copy = X.copy()
        self.__mapper = {}
        #set columns which will be transformed if index list was not given
        if self.index is None:
            self.index = [i for i in range(X_copy.shape[1]) if isinstance(X_copy[0 ,i],str) and len(set(X_copy[:, i])) < self.min_int_freq and len(set(X_copy[:, i])) != 2]

        #Smoothing encoding, with penalization for categories with small counts
        for col in self.index:
            self.__mapper[col]={}
            #find all unique categories
            unique_values = np.unique(X_copy[:,col])
            target_count = np.zeros(X_copy.shape[0])
            target_average = np.zeros(X_copy.shape[0])

            #for every category, calculate count number and target mean and store it in array
            for unique in unique_values:
                unique_set = y[np.where(X_copy[:,col] == unique)]
                average = unique_set.sum()/len(unique_set)
                count = len(unique_set)
                target_count[X_copy[:,col] == unique] = count
                target_average[X_copy[:,col] == unique] = average

            #use created array, and average and count values for every category to create mapping
            for unique in unique_values:
                unique_set =  y[np.where(X_copy[:,col]==unique)]
                average = unique_set.sum()/len(unique_set)
                count = len(unique_set)
                mapping = (average*count+target_average.mean()*self.alpha)/(count*self.alpha)
                self.__mapper[col][unique]=mapping

        return self

    def transform(self, X : 'np.array', **fit_params : dict) -> 'np.array':
        '''
            The transform function of the MeanEncoding, transforms the passed data..
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        X_copy = X.copy()
        #Transform categorical values, using created mapping dictionary
        if self.shape[1] == X.shape[1]:
            for i in self.index:
                X_copy[:,i] = np.array([self.__mapper[i][val] for val in X_copy[:,i]])
        else:
            raise ValueError(f'Was passed an array with {X.shape[1]} features, while where required {self.shape[1]} features')
        return X_copy


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
        df_copy[columns_list] = self.fit_transform(df_copy[columns_list].values,df_copy[target].values)
        df_copy[target] = df[target]
        return df_copy