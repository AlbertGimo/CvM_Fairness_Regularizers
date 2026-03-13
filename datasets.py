# code taken from https://github.com/liuhaixias1/Fair_dc

import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabulate import tabulate


class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()
    

def load_adult_data(path="../../dataset/adult", sensitive_attribute="sex"):
    column_names = ["age","workclass","fnlwgt","education","education_num",
                    "marital-status","occupation","relationship","race",
                    "sex","capital_gain","capital_loss","hours_per_week",
                    "native-country","target"]

    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "native-country", "education"]
    features_to_drop = ["fnlwgt"]

    df_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python")
    df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python", skiprows=1)

    df = pd.concat([df_train, df_test])
    df.drop(columns=features_to_drop, inplace=True)
    df.dropna(inplace=True)

    # df = pd.get_dummies(df, columns=categorical_features)

    if sensitive_attribute == "race":
        df = df[df["race"].isin(["White", "Black"])]
        s = df[sensitive_attribute][df["race"].isin(["White", "Black"])]
        s = (s == "White").astype(int).to_frame()
        categorical_features.append( "sex" )

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "Male").astype(int).to_frame()
        categorical_features.append( "race" )

    df["target"] = df["target"].replace({"<=50K.": 0, ">50K.": 1, ">50K": 1, "<=50K": 0})
    y = df["target"]

    X = df.drop(columns=["target", sensitive_attribute])
    # X = pd.get_dummies(X, columns=categorical_features)
    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s

def load_acs_data(path = '../../dataset/acs/raw', target_attr="income", 
                    sensitive_attribute="race", survey_year="2018", 
                    states=["CA"], horizon="1-Year",survey='person'):
    from folktables import (ACSDataSource, ACSEmployment, ACSIncome,
                            ACSMobility, ACSPublicCoverage, ACSTravelTime)
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey, root_dir=path)
    data = data_source.get_data(states=states, download=True)

    if target_attr == "income":
        features, labels, _ = ACSIncome.df_to_pandas(data)
        categorical_features = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP"]
    elif target_attr == "employment":
        features, labels, _ = ACSEmployment.df_to_pandas(data)
        categorical_features = ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM"]
    elif target_attr == "publiccoverage":
        features, labels, _ = ACSPublicCoverage.df_to_pandas(data)
        categorical_features = ['AGEP','SCHL','MAR','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER']
    elif target_attr == "mobility":
        features, labels, _ = ACSMobility.df_to_pandas(data)
        categorical_features = ['AGEP','SCHL','MAR','DIS','ESP','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','GCL','COW','ESR','WKHP','JWMNP','PINCP']
    elif target_attr == "traveltime":
        features, labels, _ = ACSTravelTime.df_to_pandas(data)
        categorical_features = ['AGEP','SCHL','MAR','DIS','ESP','MIG','RELP','PUMA','ST','CIT','OCCP','JWTR','POWPUMA','POVPIP']

    else:
        print( "[Main] Dataload Error" )
        exit(1)
    

    df = features
    y = labels.astype(np.int32)
    if sensitive_attribute == "sex":
        sensitive_attribute = "SEX"
        s = (df["SEX"] == 2).astype(np.int32).to_frame()
        categorical_features.append("RAC1P")

    elif sensitive_attribute == "race":
        sensitive_attribute = "RAC1P"
        s = (df["RAC1P"] == 1).astype(np.int32).to_frame()
        categorical_features.append("SEX")

    X = df.drop(columns=[sensitive_attribute])
    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    # print(string_cols)
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s


def datasetPreprocessing(path, dataset_name, split_list, seed,
                  sensitive_attribute=None, to_tensors=True, verbose=False):
    """
    Split and preprocess a tabular dataset. Return the training, validation, and test sets as PyTorch tensors.
    """
    task = "classification"
    if dataset_name == "adult":
        if sensitive_attribute is None:
            sensitive_attribute = "sex"
        X, y, s = load_adult_data(path=path+'adult', sensitive_attribute=sensitive_attribute)
    elif dataset_name == "acs":
        if sensitive_attribute is None:
            sensitive_attribute = "race"
        X, y, s = load_acs_data(path=path+'acs/raw', sensitive_attribute=sensitive_attribute)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented. Please choose from 'adult' or 'acs'.")
    
    # One-hot encoding for categorical features
    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # Performing training, validation, and test set partitioning
    if seed is not None:
        X_train, X_testvalid, y_train, y_testvalid, s_train, s_testvalid = train_test_split(
            X, y, s, train_size=split_list[0], stratify=y, random_state=seed)
        X_test, X_val, y_test, y_val, s_test, s_val = train_test_split(
            X_testvalid, y_testvalid, s_testvalid, train_size=split_list[1], stratify=y_testvalid,
            random_state=seed)
    else:
        X_train, X_testvalid, y_train, y_testvalid, s_train, s_testvalid = train_test_split(
            X, y, s, train_size=split_list[0], stratify=y)
        X_test, X_val, y_test, y_val, s_test, s_val = train_test_split(
            X_testvalid, y_testvalid, s_testvalid, train_size=split_list[1], stratify=y_testvalid)
    
    dataset_stats = {
        "dataset": dataset_name.upper(),
        "task": task,
        "num_features": X.shape[1],
        "num_classes": len(np.unique(y)),
        "num_sensitive": len(np.unique(s)),
        "num_samples": X.shape[0],
        "num_train": X_train.shape[0],
        "num_val": X_val.shape[0],
        "num_test": X_test.shape[0],
        "num_y1": (y.values == 1).sum(),
        "num_y0": (y.values == 0).sum(),
        "num_s1": (s.values == 1).sum(),
        "num_s0": (s.values == 0).sum(),
    }
    table = tabulate([(k, v) for k, v in dataset_stats.items()])


    # Standardization
    numerical_cols = X.select_dtypes("float32").columns
    if len(numerical_cols) > 0:
        # scaler = StandardScaler().fit(X[numerical_cols])
        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        scaler = StandardScaler().fit(X_train[numerical_cols])
        X_train[numerical_cols] = X_train[numerical_cols].pipe(scale_df, scaler)
        X_val[numerical_cols]   = X_val[numerical_cols].pipe(scale_df, scaler)
        X_test[numerical_cols]  = X_test[numerical_cols].pipe(scale_df, scaler)
    non_numerical_cols = X.select_dtypes(exclude="float32").columns
    if len(non_numerical_cols) > 0: 
        X_train[non_numerical_cols] = X_train[non_numerical_cols].astype("float32")
        X_val[non_numerical_cols] = X_val[non_numerical_cols].astype("float32")
        X_test[non_numerical_cols] = X_test[non_numerical_cols].astype("float32")

    if verbose:
        print("Dataset statistics:")
        print(table)
        # Debug
        print("[DEBUG] X_train.shape: ", X_train.shape, " y_train.shape: ", y_train.shape,
            " s_train.shape: ", s_train.shape)
    
    if to_tensors:
        # print(type(X_train.values[0]), type(y_train.values[0]), type(s_train.values[0]))
        # print(X_train.values[0], y_train.values[0], s_train.values[0])
        train_data = PandasDataSet(X_train, y_train, s_train)
        val_data = PandasDataSet(X_val, y_val, s_val)
        test_data = PandasDataSet(X_test, y_test, s_test) 
        return train_data, val_data, test_data, dataset_stats
        # X_train = torch.from_numpy(X_train.values).float()
        # y_train = torch.from_numpy(y_train.values).float().view(-1, 1)
        # s_train = torch.from_numpy(s_train.values).float()

        # X_val = torch.from_numpy(X_val.values).float()
        # y_val = torch.from_numpy(y_val.values).float().view(-1, 1)
        # s_val = torch.from_numpy(s_val.values).float()

        # X_test = torch.from_numpy(X_test.values).float()
        # y_test = torch.from_numpy(y_test.values).float().view(-1, 1)
        # s_test = torch.from_numpy(s_test.values).float()
    # return X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test, dataset_stats


def getDataLoader(dataset,batch_size, num_workers, pin_memory=True, shuffle=True):
    """
    Get data loader for a tabular dataset.
    """
    # Constructing PyTorch data loaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    
    return dataloader

    