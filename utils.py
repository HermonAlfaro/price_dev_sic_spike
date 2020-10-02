# imports

# external
import pandas as pd
import numpy as np
import tqdm

from sklearn.preprocessing import OrdinalEncoder

# custom
from eda import *


def get_hora_ano(horas_ano_df, dia_calendario, hora_dia):
    return horas_ano_df[(horas_ano_df["dia_calendario"] == dia_calendario) & (horas_ano_df["hora_dia"] == hora_dia)][
        "hora_ano"].values[0]


def create_time_series_features(df, columns, lags=[1, 2, 3, 4, 5], op="lag"):
    # lags: de 1, 2, ..., 24 horas

    lag_dfs = []
    for i in lags:

        barras = df["nemotecnico_se"].unique()

        tmp_dfs = []
        for sub in barras:

            if op == "lag":

                tmp = df[df["nemotecnico_se"] == sub].shift(i)[columns]

            elif op == "sma":
                tmp = df[df["nemotecnico_se"] == sub][columns].rolling(window=i).mean()

            elif op == "cma":
                tmp = df[df["nemotecnico_se"] == sub][columns].expanding(min_periods=i).mean()

            elif op == "ewm":
                tmp = df[df["nemotecnico_se"] == sub][columns].ewm(span=i, adjust=False).mean()

            elif op == "smvar":
                tmp = df[df["nemotecnico_se"] == sub][columns].rolling(window=i).var()

            elif op == "cmvar":
                tmp = df[df["nemotecnico_se"] == sub][columns].expanding(min_periods=i).var()

            elif op == "ewvar":
                tmp = df[df["nemotecnico_se"] == sub][columns].ewm(span=i, adjust=False).var()

            tmp.columns = [f"{c}_{op}_{i}" for c in columns]

            tmp_dfs.append(tmp)

        tmp_df = pd.concat(tmp_dfs)
        lag_dfs.append(tmp_df)

    lag_df = pd.concat(lag_dfs, axis=1)

    return lag_df


def split_train_vad_test(X, test_size=0.2, nb_folds=5):
    # div in train and test
    train_vad_size = (1 - test_size)

    hora_ano_unique = sorted(X["hora_ano"].unique())

    X_train_vad = X[X["hora_ano"].isin(hora_ano_unique[:round(train_vad_size * len(hora_ano_unique)) + 1])]
    # _train_vad = y[y.index.isin(X_train_vad.index)]

    X_test = X[X["hora_ano"].isin(hora_ano_unique[round(train_vad_size * len(hora_ano_unique)) + 1:])]
    # _test = y[y.index.isin(X_test.index)]

    hora_ano_unique_train_vad = sorted(X_train_vad["hora_ano"].unique())

    base_size = round((1 / (nb_folds + 1)) * len(hora_ano_unique_train_vad)) + 1

    cv_folds = []

    for i in tqdm.tqdm_notebook(range(1, nb_folds + 1)):
        X_train = X_train_vad[X_train_vad["hora_ano"].isin(hora_ano_unique_train_vad[: i * base_size])]
        # _train = y_train_vad[y_train_vad.index.isin(X_train.index)]

        X_vad = X_train_vad[X_train_vad["hora_ano"].isin(hora_ano_unique_train_vad[i * base_size: (i + 1) * base_size])]
        # _vad = y_train_vad[y_train_vad.index.isin(X_vad.index)]

        cv_folds.append((X_train.index, X_vad.index))

    return cv_folds, X_train_vad.index, X_test.index


def plot_feature_importance(feature_names, feature_importance, n=10):
    df = pd.DataFrame({"importance": feature_importance}, index=feature_names).sort_values("importance",
                                                                                           ascending=False)

    df.iloc[:n].sort_values("importance", ascending=True).plot(kind='barh', title='Feature importance',
                                                               figsize=(10, 10))

    plt.show()

    return df


def select_features_and_target(df, to_drop, target_col="cmg_cat", lag=-1, drop_target_col=False):
    # np.inf -> outliner -> delete
    df_aux = df[~df.isin([np.inf, -np.inf]).any(1)]

    # drop specified columns
    df_aux = df_aux.drop(to_drop, axis=1)

    # substation names
    barras = df_aux["nemotecnico_se"].unique()

    # init target df
    target_df = pd.DataFrame()

    # iterate to calculate target 
    for b in barras:
        tmp = df_aux[df_aux["nemotecnico_se"] == b][["hora_ano", target_col]].sort_values("hora_ano")

        tmp[f"target"] = tmp[target_col].shift(lag)

        target_df = pd.concat([target_df, tmp[["target"]]])

    # concat with target
    df_aux = pd.concat([df_aux, target_df["target"]], axis=1)

    # delete rows with nan target values
    df_aux = df_aux[df_aux["target"].notna()]

    # target as int
    df_aux["target"] = df_aux["target"].astype(int)

    # sort by hora ano
    df_aux = df_aux.sort_values("hora_ano")

    # reset index
    df_aux = df_aux.reset_index(drop=True)

    # features and target
    if drop_target_col:
        X = df_aux.drop([target_col, "target"], axis=1)
    else:
        X = df_aux.drop(["target"], axis=1)
    y = df_aux[["target"]]

    return X, y


def make_feature_engineering(df, lag=-1, stats_func_names=["lag", "sma", "cma", "ewm", "smvar", "cmvar", "ewvar"],
                             columns_to_lag=["cmg_desv", "demanda_mwh", "cap_inst_mw", "en_total_mwh", "cmg_cat"],
                             drop_target_col=False):
    ohc = OrdinalEncoder()
    ohc.fit(df[["nemotecnico_se"]])

    df[["nemotecnico_se"]] = ohc.transform(df[["nemotecnico_se"]]).astype(int)

    print(f"Creating features from {columns_to_lag}.\nUsing {stats_func_names}")

    for sfm in tqdm.tqdm_notebook(stats_func_names):
        df = pd.concat([df, create_time_series_features(df, columns_to_lag, op=sfm)], axis=1)

    miss_tab = missing_values_table(df)
    to_drop = ["fecha"]
    for feat in miss_tab.index:
        if miss_tab.loc[feat, "% of Total Values"] == 100.0:
            to_drop.append(feat)

    print(f"Dropping {len(to_drop) - 1} features because the 100 % of the column is NaN")

    X, y = select_features_and_target(df, to_drop, target_col="cmg_cat", lag=lag, drop_target_col=drop_target_col)

    return X, y
