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
plt.style.use("ggplot")
from matplotlib.pyplot import figure




def clean_functions_info():
    print("The clean functions are:")
    print("def load_imports()")
    print("List of all the modules to import")
    print("1.-", "import_load_data(csv_path, encoding=\"ISO-8859-1\")")
    print("Load data.")
    print("1.-","first_view(df)")
    print("Overview of the data.")
    print("2.-","first_transformation(df, drop_rate = None)")
    print("Standarize the Dataframe and first study about missing values.")
    print("3.-","drop_columns_nans(df, max_nan=0, subset=None, keep=\"first\")")
    print("Drop duplicated, and missing values in columns and rows. Clean also non statistical columns")
    print("4.-","col_to_num(col, type=\"float64\", fill=\"nan\")")
    print("Transform columns to numeric and fill na")
    print("5.-","categorical_search(df):")
    print("Overview to see wich column could be a categorical column")
    print("6.-","stand_categorical(col, pattern, other= None)")
    print("Transform a column to categorical and clean it fill it with other or mode" "Pattern must see like below")
    print("""pattern = [(df[\"sex\"].str.contains(\"m\", case=False, regex=False, na=False), \"m\"),
           (df[\"sex\"].str.contains("f", case=False, regex=False, na=False), "f"),
           (df["sex"].str.contains("n", case=False, regex=False, na=False), "n")]""")   
    print("7.-","clean_datetime(col, new_columns=False)")
    print("Transform column to datetime and add new columns if needed")
    print("8.-","fill_all(df, fill=\"median\")")
    print("Fill missing numeric columns with median or mean")


    
def import_load_data(csv_path, encoding="ISO-8859-1"):
    df = pd.read_csv(csv_path, encoding=encoding)
    return df
    
def first_view(df):
    print(df.info())
    print("---------------------------------------------------------------------------")
    display(df.describe().T)
    print("---------------------------------------------------------------------------")
    display(df.head())
    print("---------------------------------------------------------------------------")
    df.hist(bins=50, figsize =(20,15))
    plt.show()
    
def first_transformation(df, drop_rate = None):
    #strip y lower
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

def drop_columns_nans(df, max_nan=0, subset=None, keep="first"):
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
        if top_pct > 0.95:
            low_information_cols.append(col)
    df.drop(low_information_cols, axis=1, inplace=True)
    b = df.shape
    columns_dropped = a[1] - b[1]
    rows_dropped = a[0] - b[0]
    print(f"{columns_dropped} columns dropped")
    print(f"{rows_dropped} rows dropped")
    print(f"The new shape is {b[0]} rows and {b[1]} columns")
    return df
 
def col_to_num(col, type="float64", fill="nan"):
    try:
        df[col] = df[col].str.replace(r"^[[+-]?[0-9]*\.?[0-9]+] | [^\w\s*] | [a-zA-Z]|½|\?|<|>|\"|-|\(|\)|\.|\'| |&", "", regex = True)
    except AttributeError:
        pass
    df[col] = pd.to_numeric(df[col], errors = "coerce")
    #elegir entre median, mean u otro valor
    if fill == "median":
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    elif fill == "mean":
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)
    else:
        df[col].fillna(fill, inplace=True)
    return df
    
def categorical_search(df):
    insta_df = pd.DataFrame()
    df_non_numeric = df.select_dtypes(exclude = [np.number])
    non_numeric_cols = df_non_numeric.columns.values.tolist()
    for col in non_numeric_cols:
        a = df.stb.freq([col], thresh= 0.9*100, other_label = "0ther")
        display(a)
        print("-----------------------------------------------------------------------")
       
def stand_categorical(col, pattern, other= None):
    df[col] = df[col].astype("category")
    store_criteria, store_values = zip(*pattern)
    df[f"{col}_new"] = np.select(store_criteria, store_values, other)
    df[col] = df[f"{col}_new"].combine_first(df[col])
    return df
    
def clean_datetime(col, new_columns=False):
    df[col] = df[col].str.replace(r"^[[+-]?[0-9]*\.?[0-9]+]|[^\w\s*] | ½|\?|<|>|\"|-|\(|\)|\.|\'| |&]|reported", "", regex = True)
    df[col]  = pd.to_datetime(df[col], errors= "coerce")
    df[col] = df[col].astype("datetime64[D]", errors="raise")
    if new_columns:
        df["year"] = df[col].dt.year
        df["month"] = df[col].dt.month
        df["day"] = df[col].dt.day
        df.drop([col], axis = 1, inplace = True)
    return df

def fill_all(df, fill="median"):
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

def load_imports():
    print("""print(You also need to import all these modules:
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
            plt.style.use("ggplot")
            from matplotlib.pyplot import figure
            %matplotlib inline
            matplotlib.rcParams["figure.figsize"] = (12,8)""")