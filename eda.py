# imports

# internal
import datetime

# external
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts and calculates null values per column

    :param df: features's DataFrame
    :return:
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Hay " + str(df.shape[1]) + " columnas.\n"
                                      "Hay " + str(mis_val_table_ren_columns.shape[0])
          + " columnas con valores nulos")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def plot_distribution_labels(data: pd.DataFrame, plot_title: str, output_col: str):
    """
    plots distribution of output labels in a pie-chart.

    :param data: DataFrame containing output_col column
    :param plot_title: title of the pie-chart
    :param output_col: name of the labels column
    :return:
    """

    # number of examples per category
    data_count = data[output_col].value_counts()

    # labels
    labels = list(data_count.index)

    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    data_count.plot.pie(
        autopct="%1.2f%%", colors=sns.color_palette("muted"), startangle=90,
        labels=labels,
        wedgeprops={"linewidth": 2, "edgecolor": "k"}, explode=[.1, 0], shadow=True)
    plt.title(plot_title)


def kde_plot_with_target(df: pd.DataFrame, col_name: str, target_col: str):
    """
    generates kernel density estimation plot (KDE) of input column with labels .

    :param df: DataFrame containing col_name and target_col columns
    :param col_name: name of the column to plot
    :param target_col: name of labels column
    :return:
    """

    plt.figure(figsize=(7, 6))

    # labels
    labels = list(df[target_col].unique())

    for lab in labels:
        # KDE plot of loans which were not repaid on time
        sns.kdeplot(df.loc[df[target_col] == lab, col_name], label=lab)

    # Labeling of plot
    plt.xlabel(col_name)
    plt.ylabel('Density')
    plt.title('Distribution of {}'.format(col_name))
    plt.legend()
    plt.show()


def plot_bar_per_category(df: pd.DataFrame, cat_col: str, target_col: str):
    """
    plots a bar plot of feature column with labels

    :param df: DataFrame containing cat_col and target_col columns
    :param cat_col: name of the column to plot
    :param target_col: name of labels column
    :return:
    """

    # categorie of categorical column
    categories = list(df[cat_col].dropna().unique())

    # labels of taregt column
    labels = list(df[target_col].unique())

    # creating temporary DataFrame to plot percentages
    labels_pct = {}

    for lab in labels:
        labels_pct[lab] =  [100 * round(len(df[(df[cat_col] == c) & (df[target_col] == lab)]) / len(df[df[cat_col] == c]), 2)
           for c in categories]

    tmp = pd.DataFrame(labels_pct).dropna()

    tmp.set_index(cat_col).plot.bar()
    plt.title(f"Distribution of  {cat_col}")
    plt.ylabel("Percentage per label")
    plt.xlabel("")
    plt.tight_layout()
    plt.legend(loc='upper center', bbox_to_anchor=(1.14, 1))
    plt.show()


def correlation_df(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    calculates correlations between the target and features

    :param data: DataFrame
    :param target_col: name of labels column to drop before calculating the correlation matrix
    :return: features's correlations to target
    """

    # Assessing correlations and sorting
    corr_target_features = data.corr()[target_col].sort_values(ascending=False).to_frame()

    # dropping target column
    corr_target_features.drop(target_col, inplace=True)

    # renaming columns
    corr_target_features.columns = ["correlation"]

    return corr_target_features

def time_plot_costo_barra(cm, codigo_barra, fecha_inicial=None, fecha_final=None):
    
    
    # filtrar codigo de barra
    tmp = cm[cm["barra"]==codigo_barra].dropna()
    
    # index by time
    tmp["datetime"] = pd.to_datetime(tmp["fecha"])
    
    if fecha_inicial:
        tmp = tmp[fecha_inicial<=tmp["datetime"]]
    if fecha_final:
        tmp = tmp[tmp["datetime"]<=fecha_final]
    
    plt.plot(tmp.set_index("datetime")[["cmg_real", "cmg_prog"]])
    plt.legend(["cmg_real", "cmg_prog"])
    plt.xlabel("fecha")
    plt.show()
    
def evolucion_diaria(ds, subestacion, variable, fechas, ax=None):
    
    fechas_dt = [datetime.date(year=int(f.split("-")[0]), 
                                   month=int(f.split("-")[1]), 
                                   day=int(f.split("-")[2])) for f in fechas]
    
    tmp = ds[(ds["nemotecnico_se"]==subestacion) & (ds["fecha"].isin(fechas_dt))]
    
    tmp = pd.pivot_table(tmp, index="hora", values=variable, columns="fecha")
    
    tmp.plot(ax=ax)
    plt.legend(fechas)
    plt.xlabel("Hora del día")
    plt.ylabel(variable)
    

    
