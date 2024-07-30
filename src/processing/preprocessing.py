import pandas as pd
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

def impute_missing(data, out_path=None, filename='data'):
    """
    Takes the data as pandas dataframe, imputes using the MICE algorithm, returns imputed dataset.
    :param data: dataset as dataframe
    :param out_path: Filepath to save csv with data to, optional
    :return: Imputed data as pandas dataframe
    """
    imputer = IterativeImputer(max_iter=100)
    data_np = imputer.fit_transform(data)
    # convert back to DataFrame
    data_imputed = pd.DataFrame(data_np, index=data.index, columns=data.columns)
    # save data
    if out_path != None:
        data_imputed.to_csv(os.path.join(out_path, f"02_{filename}_imputed.csv"))
    return data_imputed

def regress_confounders(df, age=None, gender=None, out_path=None, filename="data"):
    """
    Takes the data as pandas dataframe, regresses out age and/or gender, returns residuals as dataframe.
    :param data: dataset as dataframe
    :param out_path: Filepath to save csv with data to, optional
    :return: Residuals as pandas dataframe
    """
    # lib male as 0 and female as 1
    if gender is not None:
        gender[gender == 'male'] = 0.0
        gender[gender == 'female'] = 1.0
    # matrix of predictors age and/or gender
    if age is not None and gender is not None:
        x = np.column_stack((age, gender))
    elif age is None and gender is not None:
        x = np.array(gender)
        x = x.reshape(-1, 1)
    elif age is not None and gender is None:
        x = np.array(age)
        x = x.reshape(-1, 1)
    elif age is None and gender is None:
        raise Exception("Either age or gender must be given!")
    # z-score variables
    mean = df.mean().values
    sd = df.std().values
    y = ((df-mean)/sd)
    # fit regression model
    reg = LinearRegression().fit(x,y)
    # predict values based on model
    pred = reg.predict(x)
    # residuals: difference between actual and predicted values
    data_res = y-pred
    if out_path != None:
        data_res.to_csv(os.path.join(out_path, f"03_{filename}_res.csv"))
    return data_res

def z_score(data, out_path=None, filename='data'):
    """
    Z-scores data
    :param data: Dataset as dataframe
    :param out_path: Filepath to save csv with data to, optional
    :return: Dataset with z-scored values as pandas dataframe
    """
    # z-score variables
    mean = data.mean().values
    sd = data.std().values
    data_z = ((data-mean)/sd)
    if out_path != None:
        data_z.to_csv(os.path.join(out_path, f"05_{filename}_z.csv"))
    return data_z