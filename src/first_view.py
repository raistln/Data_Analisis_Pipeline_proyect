import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np


def first_transformation(df):
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip()
    instant_df = pd.DataFrame()
    inf_df = pd.DataFrame(index=range(1))
    cols = df.columns
    for col in cols:
        pct_missing = np.mean(df[col].isnull())
        inf_df[col] = f"{round(pct_missing*100)}%"
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            instant_df[f"{col}_is_missing"] = missing
        else:
            instant_df[f"{col}_is_missing"] = 0
    display(inf_df.T)
    is_missing_cols = [col for col in instant_df.columns if "is_missing" in col]
    instant_df["num_missing"] = instant_df[is_missing_cols].sum(axis = 1)
    if len(cols) < 30:
        colors = ["#000099", "#ffff00"]
        sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colors))
    else:
        instant_df["num_missing"].value_counts().reset_index().sort_values(by="index").plot.bar(x = "index", y = "num_missing")
    plt.show()
    
    return df




def drop_columns_nans(df, max_nan=0, subset=None, keep="first", toppct = 0.95):
    a = df.shape

    df.dropna(axis=1, how="all", inplace=True)
    df = df.T.drop_duplicates().T

    df = df.dropna(axis=0, thresh=max_nan)
    df.drop_duplicates(subset, keep, inplace=True)

    num_rows = len(df.index)
    low_information_cols = []
    for col in df.columns:
        cnts = df[col].value_counts(dropna=False)
        top_pct = (cnts/num_rows).iloc[0]
        if top_pct > toppct:
            low_information_cols.append(col)
    df.drop(low_information_cols, axis=1, inplace=True)
    b = df.shape
    columns_dropped = a[1] - b[1]
    rows_dropped = a[0] - b[0]
    
    print(f"{columns_dropped} columns dropped")
    print(f"{rows_dropped} rows dropped")
    print(f"The new shape is {b[0]} rows and {b[1]} columns")
    
    return df




def all_or_part_df(df):
    print(df.columns.tolist())
    while True:
        columns = input("Write only the columns you want to work with. Without and separated by ,\n")
        new_columns = columns.replace("\'", "").split(",")
        new_columns = [i.strip() for i in new_columns]
        try:
            df = df[new_columns]
            return df
        except:
            print("Wrong columns. Try other columns")
            continue
            
            
            

def first_view(df):
    
    df = first_transformation(df)
        
    duplicates = input("Do you want to remove duplicates,empty rows/columns, and irrelevant stadistical columns? y/n:\n")
    if duplicates == "y":
        max_nan = int(input(f"How many non-nan requires in a row? Please insert a value between 0-{len(df.columns)}:\n"))
        toppct = 0.01 * float(input("What is the maximum % of nan in a column to be stadistical relevant? Insert a value between 0-100, (recommended is 95%):\n"))
        df = drop_columns_nans(df, max_nan=0, keep="first", toppct = 0.95)
                   
    part = input("Do you want to use all the df(yes) or just a part(n)? y/n:\n")
    if part == "n":
        df = all_or_part_df(df)
        
    print("\n"*3)

    return df