#!/usr/bin/env python
# coding: utf-8

# In[1]:

import math
import sidetable
import urllib3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlas
import matplotlib
from IPython.display import display
from random import choices
plt.style.use("ggplot")
from matplotlib.pyplot import figure


# In[2]:



#cargar el csv path y descargar y zip con BeautifulSoup
# In[3]:


def import_load_data(csv_path, encoding="ISO-8859-1"):
    df = pd.read_csv(csv_path, encoding=encoding)
    return df


# In[4]:


def first_view(df):
    print(df.info())
    print("---------------------------------------------------------------------------")
    display(df.describe().T)
    print("---------------------------------------------------------------------------")
    display(df.head())
    print("---------------------------------------------------------------------------")
    df.hist(bins=50, figsize =(20,15))
    plt.show()


# In[5]:


def all_or_part_df(df):
    print(df.columns)
    while True:
        answer_all = input("Do you want transform all df or only somo columns? If so type df or the list of columns you want to transform without [].")
        if answer_all == "df":
            return df
        else:
# ARREGLAR LA PARTE DE LA LISTA NO VA BIEN.
            lst = answer_all.split(",")
            if lst in df.columns.tolist().split(","):
                return df[[lst]]
            else:
                print("Wrong columns. Try df or a list of columns")
                continue
            


# In[6]:


def first_transformation(df):
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip()
    instant_df = pd.DataFrame()
    cols = df.columns
    for col in cols:
        pct_missing = np.mean(df[col].isnull())
        print(f"{col} - {round(pct_missing*100)}%")
        missing = df[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:
            instant_df[f"{col}_is_missing"] = missing
        else:
            instant_df[f"{col}_is_missing"] = 0
    is_missing_cols = [col for col in instant_df.columns if "is_missing" in col]
    instant_df["num_missing"] = instant_df[is_missing_cols].sum(axis = 1)
    if len(cols) < 30:
        colors = ["#000099", "#ffff00"]
        sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colors))
    else:
        instant_df["num_missing"].value_counts().reset_index().sort_values(by="index").plot.bar(x = "index", y = "num_missing")
    return df


# In[7]:


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


# In[8]:


def numeric_search(df, tries = 5):
    for col in df.columns.tolist():
        while True:
            five = choices(df[col], k=tries)
            five = [i for i in five if i is not None]
            if len(five) < 5:
                continue
            else:
                break
        try:
            five_true = map(float, five)
            fill = input("Do you want to fill the nan with median, mean or nan? Write one of them.")
            df = col_to_num(df, col, type="float64", fill= fill)
            return df
        except ValueError:
            continue


# In[9]:


def col_to_num(df, col, type="float64", fill="nan"):
    try:
        df[col] = df[col].str.replace(r"^[[+-]?[0-9]*\.?[0-9]+] | [^\w\s*] | [a-zA-Z]|½|\?|<|>|\"|-|\(|\)|\.|\'| |&", "", regex = True)
    except AttributeError:
        pass
    df[col] = pd.to_numeric(df[col], errors = "coerce")
    #elegir entre median, mean u otro valor
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


# In[10]:


def categorical_search(df):
    insta_df = pd.DataFrame()
    df_non_numeric = df.select_dtypes(exclude = [np.number])
    non_numeric_cols = df_non_numeric.columns.values.tolist()
    prob_cat = []
    for col in non_numeric_cols:
        a = df.stb.freq([col], thresh= 0.9*100, other_label = "0ther")
        display(a)
        print("-----------------------------------------------------------------------")
        if len(df[col].value_counts()) < len(df.index)*0.05:
            prob_cat.append(col)
    return prob_cat


# In[11]:


def transform_cat(df):
    prob_cat = categorical_search(df)
    while True:
        answer_cat1 = input("Do you want to transform columns to category? yes or no?")
        if answer_cat1.lower() == "no":
            return df
        elif answer_cat1.lower() == "yes":
            while True:
                answer_cat2 = input(f"Wich columns do you want to change? Make a list without spaces and []. My suggestion is {prob_cat}")
#Arreglar para una lista más amplia
                lst = list(answer_cat2.split(","))
                if lst in df.columns.tolist():
                    for col in lst:
                        pattern = []
                        while True:
                            answer_cat3 = input("Do you want to make a pattern? yes or no?")
                            if answer_cat3 == "no":
                                df = stand_categorical()
                                return df
                            elif answer_cat3 == "yes":
                                print(df[col].values_count())
                                pattern = make_pattern()
                                df = stand_categorical()
                            else:
                                print("Please enter yes or no")
                                continue
            else:
                print("Try another list of columns")
        else:
            return df


# In[15]:


def make_pattern(col, pattern):
    while True:
        tuples = input("Please write the words to change like a tuple: Example: (\"word_to_change\", \"word_change\")")
        if isintance(tuples, tuple):
            pat = (df[col].str.contains(tuples[0], case=False, regex=False, na=False), tuples[1])
            pattern.append(pat)
        else:
            finish = input("Have you finished?? Write end or left blank.")
            if finish == "" or finish.lower() == "end":
                return pattern
            else: 
                continue


# In[16]:


def stand_categorical(col, pattern, other= None):
    df[col] = df[col].astype("category")
    store_criteria, store_values = zip(*pattern)
    df[f"{col}_new"] = np.select(store_criteria, store_values, other)
    df[col] = df[f"{col}_new"].combine_first(df[col])
    return df
    


# In[17]:


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


# In[20]:


def stand_datetime(df):
    while True:
        answer_date = input("Do you want to clean de datetime column? yes or no?")
        if answer_date.lower() == "no":
            break
        elif answer_date.lower() == "yes":
            while True:
                col = input("Wich column do you want to change. Please enter the name of the column.")
                col = col.lower()
                if col in df.columns.tolist():
                    while True:
                        answer_date3 = input("Do you want to generate new columns with year, month and day?")
                        if answer_date3 == "yes":
                            df = clean_datetime(df, col, new_columns=True)
                            return df
                        else:
                            df = clean_datetime(df, col, new_columns=False) 
                            return df
                else:
                    break                        
        else:
            print("Please enter yes or no.")
            continue


# In[21]:


def fill_all(df, fill):
    df_numeric = df.select_dtypes(include = [np.number])
    df_numeric_columns = df_numeric.columns.tolist()
    df_categoric = df.select_dtypes(include = "category")
    df_categoric_columns = df_categoric.columns.tolist()
    for col in df_numeric_columns:
        if fill == "median":
            median = df[col].median()
            df[col].fillna(median, inplace=True)
        if fill == "mean":
            mean = df[col].mean()
            df[col].fillna(mean, inplace=True)
    for col in df_categoric_columns:
        mode = df[col].mode()
        df[col].fillna(mode, inplace=True)
    return df


# In[23]:


def finish_fill(df):
    while True:
        answer_fill = input("Do you want to fill the rest of the DataFrame? yes or no?")
        if answer_fill.lower() == "no":
            return df
        elif answer_fill.lower() == "yes":
            while True:
                fill = input("With what do you want to fill, choose one of these: median, mean")
                if fill.lower() in ["median", "mean"]:
                    fill = fill.lower()
                    df = fill_all(df, fill)
                    return df
                else:
                    return df
        else:
            return df


# In[24]:


def total_preprocessing(csv_path):
    df = import_load_data(csv_path)
    first_view(df)
    df = all_or_part_df(df)
    df = first_transformation(df)
    df = drop_columns_nans(df)
    df = numeric_search(df)
    prob_cat = categorical_search(df)
    df = transform_cat(df)
    df = stand_datetime(df)
    df = finish_fill(df)
    return df

# In[ ]:




