import pandas as pd
import numpy as np
from random import choices
import sidetable
import re
from IPython.display import display




def numeric_search(df, tries = 5):
	"""This function should search in the columns where are not numerical, like object or string and transform it to a numeric type column."""
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
            except ValueError or TypeError:
                continue
    for col in to_num:     
        fill = input(f"For the column \"{col}\", do you want to fill the nan with median, mean or nan? Write one of them.\n")
        df = col_to_num(df, col, type="float64", fill= fill)
        
    return df



def col_to_num(df, col, type="float64", fill="nan"):
	"""This function clean with regex a numeric column and take out al the special simbols, and then it changes it to a numeric type column. The nan values of the column will be filled with median or mean if it is necesary."""
    
    try:
        df[col] = df[col].str.replace(r"^[[+-]?[0-9]*\.?[0-9]+] | [^\w\s*] | [a-zA-Z]|½|\?|<|>|\"|-|\(|\)|\.|\'| |&", "", regex = True)
    except AttributeError:
        pass
    print(col)    
    df[col] = pd.to_numeric(df[col], errors = "coerce")
    
    if fill == "median":
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    elif fill == "mean":
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)
    else:
        df[col].fillna(fill, inplace=True)
        
    df[col] = pd.to_numeric(df[col], errors = "coerce")
    
    return df
    
    
def categorical_search(df):
	"""This function search in the diferent columns of the DataFrame if they are categorical with a criteria that there must be at least 5% of unique values of the length of the DataFrame. It returns a list with the most probable categorical columns."""
    insta_df = pd.DataFrame()
    df_non_numeric = df.select_dtypes(exclude = [np.number])
    non_numeric_cols = df_non_numeric.columns.values.tolist()
    prob_cat = []
    
    for col in non_numeric_cols:
        a = df.stb.freq([col], thresh= 0.9*100, other_label = "0ther")
        if len(df[col].value_counts()) < len(df.index)*0.05:
            #display(a)
            prob_cat.append(col)
            
    return prob_cat



def transform_cat(df, col):
	"""This function transforms the type of the columns of the DataFrame to a categorical one. The function first ask the user if the column is a categorical column or not, if it is, the function shows a list with the unique values and ask the user if they want to make a patter to change inconsistent data in the column, if not it only change the type of the column."""
    pattern = []
    print(df[col].value_counts())
    print("------------------------------------------")
    
    print(f"For the column {col}.")
    print(f"If the column isn´t categorical left blank the next cuestion.")
    trans_cat = input("Do you want to make a pattern? y/n: ")
    
    if trans_cat == "y":
        pattern = make_pattern(col)
        df = stand_categorical(df, col, pattern)
    elif trans_cat == "n":
        df = stand_categorical(df, col, pattern)
    return df


def make_pattern(col):
	"""This function make a list of patterns with a tuple of words that are going to replace each other. """
    pattern = []
    m_pat = input("Its a binary column? y/n\n")
    if m_pat == "y":
        keys = list(dict(df[col].value_counts()).keys())
        if len(keys) > 2:
            fill = input("Do you want to change the non binary values to nan(y) or to other value(type the other value)?\n")
            if fill == "y":
                for i in range(2, len(keys)):
                    pat = (df[col].str.contains(keys[i], case=False, regex=False, na=False), np.nan)
                    pattern.append(pat)              
            else:
                for i in range(2, len(keys)):
                    pat = (df[col].str.contains(keys[i], case=False, regex=False, na=False), fill)
    
    elif m_pat == "n":    
        while True:
            pair = input("Please write the words to change like a tuple: Example: (\"word_to_change\", \"word_change\")\n")
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



def stand_categorical(df, col, pattern, other= None):
	"""This function change the type of the column to categorical and replace words with a given pattern"""
    df[col] = df[col].astype("category")
    
    try:
        store_criteria, store_values = zip(*pattern)
        df[f"{col}_new"] = np.select(store_criteria, store_values, other)
        df[col] = df[f"{col}_new"].combine_first(df[col])
        df = df.drop([f"{col}_new"],axis=1)
        return df
    
    except:
        return df
    
    
    
def clean_datetime(df, col, new_columns):
	"""This function cleans with regex the values of a given column that should be a datetime column, after that it change the type to datetime. The function gives the posibility to make 3 new columns with year, month, day separated."""
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
	"""This function is responsible to ask the user different cuestions about the transformation of the datetime column, it ask the user wich column should change to datetime type and then if the user want 3 new columns with year, month and day separately."""
    while True:
        col = input("Wich column do you want to change. Please enter the name of the column:\n")                
        if col in df.columns.tolist():
            date_yn = input("Do you want to generate new columns with year, month and day? y/n:\n")
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
	"""This function will fill numeric columns with mean or median and categorical columns with mode if the user want it."""
    
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
	"""This function ask the cuestion if you want to fill the numeric column with mean or median and if you want to fill categorical columns with mode."""
    
    fill = input("With what do you want to fill, choose one of these: median, mean: ")
    cat = input("Do you want to fill with mode the categorical type columns? y/n: ")
    df = fill_all(df, fill, cat)
    return df




def data_wrangling(df):
	"""This funtion summarizes all the procces of wrangling the data. First it will search for numeric columns and transform it, then with categorical columns, then with datetime columns and at least it will fill de nan values of the DataFrame with a choosen value."""
    print(df.info())
    
    numeric = input("Do you want to change a column to a numerical type? y/n:\n")
    if numeric == "y":
        auto = input("Do you want it to make it automatic(y) or by yourself(n)? y/n: \n")
        if auto == "y":
            df = numeric_search(df)
        else:
            while True:
                col = input("Which column do you want to change to numerical type? Write end if you want to continue to another task.\n")
                print("Write end or left blank to finish")
                if col == "end" or col == "":
                    break
                fill = input("Do you want to fill the nan with median, mean or nan? Write one of them.\n")
                df = col_to_num(df, col, fill=fill, type="float64")
    
    categorical = input("Do you want to change a column to a categorical type? y/n: \n")
    if categorical == "y":
        prob_cat = categorical_search(df)
        cate = input(f"The columns that could be categorical are {prob_cat}, do you want to change all(y) or do it manually(n)? y/n:\n")
        if cate == "y":
            for col in prob_cat:
                df = transform_cat(df, col)
        if cate == "n":
            probs = input("Insert the columns you want to change to categorical:\n")
            prob_cat = probs.replace("\'", "").split(",")
            prob_cat = [i.strip() for i in prob_cat]
            for col in prob_cat:
                df = transform_cat(df, col)
    
    dates = input("Do you want to change a column to a datetime type? y/n:\n")
    if dates == "y":
        df = stand_datetime(df)
        
    finish = input("To finish, do you want to fill the rest of the DataFrame nans? y/n:\n")
    if finish == "y":
        df = finish_fill(df)
    df.to_csv("../OUTPUT/clean_csv.csv")
    return df
