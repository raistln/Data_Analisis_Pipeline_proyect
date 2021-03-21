import os
import tarfile
import requests
import kaggle
import pandas as pd
import sidetable
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlas
import matplotlib
from IPython.display import display
from random import choices
plt.style.use("ggplot")
from matplotlib.pyplot import figure
import time


ROOT = os.getcwd()


def fetch_data(download_url, download_path):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(download_url, path=download_path, unzip=True)
    return kaggle.api.dataset_list_files(download_url).files
    
    
    
def import_load_data(csv_path, encoding="ISO-8859-1"):
    
    encoding = input("Default enconding is ISO-8859-1, do you prefer another one? Please type it.")
    
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    return df
    
    
    
def download_import_kaggle():
    
    download_root = input("Please enter the url where your desired csv is: ")
    download_path = os.path.join("dataset")
    
    download_url = download_root.split("/", 3)[3]
    csv_files = fetch_data(download_url, download_path)

    if len(csv_files) == 1:
        csv_path = ROOT + "/dataset/" + str(csv_files[0])
    else:
        csv_file = input(f"Please what file did you need to import from this list {csv_files}?: ")
        csv_path = ROOT + "/dataset/" + csv_file
    return import_load_data(csv_path, encoding="ISO-8859-1")




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
    #quitar columnas vacias y duplicadas
    df.dropna(axis=1, how="all", inplace=True)
    df = df.T.drop_duplicates().T
    #quitar filas por condicion Thresh y filas duplicadas
    df = df.dropna(axis=0, thresh=max_nan)
    df.drop_duplicates(subset, keep, inplace=True)
    #columnas estadisticamente irrelevantes
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
        columns = input("Write only the columns you want to work with. Without and separated by ,")
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
    
    
    duplicates = input("Do you want to remove duplicates,empty rows/columns, and irrelevant stadistical columns? y/n")
    if duplicates == "y":
        max_nan = int(input(f"Who must be the maximum nan values in a row? Please insert a value between 0-{len(df.columns)}: "))
        toppct = 0.01 * float(input("What is the maximum % of nan in a column to be stadistical relevant? Insert a value between 0-100, (default is 95%): "))
        df = drop_columns_nans(df, max_nan=0, keep="first", toppct = 0.95)
                       
    display(df.head())
    
    part = input("Do you want to use all the df(yes) or just a part(n)? y/n: ")
    if part == "n":
        df = all_or_part_df(df)
        
    print("\n"*3)
    display(df.head())

    return df



def numeric_search(df, tries = 5):
    to_num = []
    
    for col in df.columns.tolist():
        if df[col].dtype == "int64" or df[col].dtype == "float64":
            continue
        else:
            while True:
                five = choices(df[col], k=tries)
                five = [i for i in five if i is not np.nan]
                if len(five) == 5:
                    break
            try:
                five_true = list(map(float, five))
                to_num.append(col)
            except ValueError:
                continue
    for col in to_num:     
        fill = input(f"For the column \"{col}\", do you want to fill the nan with median, mean or nan? Write one of them.")
        df = col_to_num(df, col, type="float64", fill= fill)
        
    return df



def col_to_num(df, col, type="float64", fill="nan"):
    
    try:
        df[col] = df[col].str.replace(r"^[[+-]?[0-9]*\.?[0-9]+] | [^\w\s*] | [a-zA-Z]|½|\?|<|>|\"|-|\(|\)|\.|\'| |&", "", regex = True)
    except AttributeError:
        pass
    df[col] = pd.to_numeric(df[col], errors = "coerce")

    if fill == "median":
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        return df
    elif fill == "mean":
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)
        return df
    else:
        df[col].fillna(fill, inplace=True)
        return df
    
    
def categorical_search(df):
    
    insta_df = pd.DataFrame()
    df_non_numeric = df.select_dtypes(exclude = [np.number])
    non_numeric_cols = df_non_numeric.columns.values.tolist()
    prob_cat = []
    
    for col in non_numeric_cols:
        a = df.stb.freq([col], thresh= 0.9*100, other_label = "0ther")
        if len(df[col].value_counts()) < len(df.index)*0.05:
            display(a)
            print("-----------------------------------------------------------------------")
            prob_cat.append(col)
            
    return prob_cat



def transform_cat(df, col):
    pattern = []
    print(df[col].value_counts())
    print("------------------------------------------")
    
    print(f"For the column {col}")
    print(f"If the column isn´t categorical left blank the next cuestion.)
    trans_cat = input("Do you want to make a pattern? y/n: ")
    
    if trans_cat == "y":
        pattern = make_pattern(col)
        df = stand_categorical(col, pattern)
    elif trans_cat == "n":
        df = stand_categorical(col, pattern)
    else:
        continue
        
    return df


def make_pattern(col):
    pattern = []
    m_pat = input("Its a binary column? y/n")
    if m_pat == "y":
        keys = list(dict(df[col].value_counts()).keys())
        if len(keys) > 2:
            fill = input("Do you want to change the non binary values to nan(y) or to other value(type the other value)?")
            if fill == "y":
                for i in range(2, len(keys)):
                    pat = (df[col].str.contains(keys[i], case=False, regex=False, na=False), np.nan)
                    pattern.append(pat)              
            else:
                for i in range(2, len(keys)):
                    pat = (df[col].str.contains(keys[i], case=False, regex=False, na=False), fill)
    
    elif m_pat == "n":    
        while True:
            pair = input("Please write the words to change like a tuple: Example: (\"word_to_change\", \"word_change\")")
            if pair == "end" or pair == "":
                return pattern
            pairs = [i.strip() if i != "nan" else i == np.nan for i in pair.split(",")]
            if isinstance(pairs, list):
                try:
                    pat = (df[col].str.contains(pairs[0], case=False, regex=False, na=False), pairs[1])
                    pattern.append(pat)
                except: 
                    print("Try to write it again")
            print("If you finished write end or left blank.")
        
    return pattern



def stand_categorical(col, pattern, other= None):
    df[col] = df[col].astype("category")
    
    try:
        store_criteria, store_values = zip(*pattern)
        df[f"{col}_new"] = np.select(store_criteria, store_values, other)
        df[col] = df[f"{col}_new"].combine_first(df[col])
        df.drop([f"{col}_new"],axis=1, inplace=True)
        return df
    
    except:
        return df
    
    
    
def clean_datetime(df, col, new_columns):
    df[col] = df[col].str.replace(r"^[[+-]?[0-9]*\.?[0-9]+]|[^\w\s*] | ½|\?|<|>|\"|-|\(|\)|\.|\'| |&]|reported", "", regex = True)
    df[col]  = pd.to_datetime(df[col], errors= "coerce")
    df[col] = df[col].astype("datetime64[D]", errors="ignore")
    
    if new_columns:
        df["year"] = df[col].dt.year
        df["month"] = df[col].dt.month
        df["day"] = df[col].dt.day
        df.drop([col], axis = 1, inplace = True)
    return df



def stand_datetime(df):
    while True:
        col = input("Wich column do you want to change. Please enter the name of the column: ")                
        if col in df.columns.tolist():
            date_yn = input("Do you want to generate new columns with year, month and day? y/n: ")
            if date_yn == "y":
                df = clean_datetime(df, col, new_columns=True)
                return df
            else:
                df = clean_datetime(df, col, new_columns=False)
                return df
        elif col == "end":
            return df
            break
        else:
            print(f"{col} doesn´t exist. Try again or write end." )
            continue
    
    
    
    
def fill_all(df, fill, cat = "n"):
    
    df_numeric = df.select_dtypes(include = [np.number])
    df_numeric_columns = df_numeric.columns.tolist()

    df_categoric = df.select_dtypes(include = "category")
    df_categoric_columns = df_categoric.columns.tolist()

    for col in df_numeric_columns:

        if fill == "median":
            median = df[col].median()
            df[col].fillna(value=median, inplace=True)
        elif fill == "mean":
            mean = df[col].mean()
            df[col].fillna(value= mean, inplace=True)
            
    if cat == "y":
        for col in df_categoric_columns:
            mode = max(df[col].mode())
            df[col].fillna(value= mode, inplace=True)
    return df



def finish_fill(df):
    
    fill = input("With what do you want to fill, choose one of these: median, mean: ")
    cat = input("Do you want to fill with mode the categorical type columns? y/n: ")
    df = fill_all(df, fill= fill, cat=cat)
    return df




def data_wrangling(df):
    print(df.info())
    
    numeric = input("Do you want to change a column to a numerical type? y/n: ")
    if numeric == "y":
        auto = input("Do you want it to make it automatic(y) or by yourself(n)? y/n: ")
        if auto == "y":
            df = numeric_search(df=df)
        else:
            while True:
                col = input("Which column do you want to change to numerical type? Write finish if you want to continue to another task. ")
                print("Write end or left blank to finish")
                if col == "end" or col == "":
                    break
                fill = input("Do you want to fill the nan with median, mean or nan? Write one of them. ")
                df = col_to_num(df=df, col=col, type="float64", fill=fill)
    
    categorical = input("Do you want to change a column to a categorical type? y/n: ")
    if categorical == "y":
        prob_cat = categorical_search(df=df)
        cate = input(f"The columns that could be categorical are {prob_cat}, do you want to change all(y) or do it manually(n)? y/n: ")
        if cate == "y":
            for col in prob_cat:
                df = transform_cat(df=df, col=col)
        if cate == "n":
            probs = input("Insert the columns you want to change to categorical")
            prob_cat = probs.replace("\'", "").split(",")
            prob_cat = [i.strip() for i in prob_cat]
            for col in prob_cat:
                df = transform_cat(df=df, col=col)
    
    dates = input("Do you want to change a column to a datetime type? y/n: ")
    if dates == "y":
        df = stand_datetime(df=df)
        
    finish = input("To finish, do you want to fill the rest of the DataFrame nans? y/n: ")
    if finish == "y":
        df = finish_fill(df=df)

    return df




def data_analisis_pipeline():
    df = download_import_kaggle()
    df = first_view(df)
    df = data_wrangling(df)
    return df