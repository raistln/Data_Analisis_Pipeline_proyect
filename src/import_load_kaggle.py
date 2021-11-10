import kaggle
import pandas as pd




def fetch_data(download_url, download_path):
    """This function is a automatization step to use the kaggle api and then detect the files of the choosen site."""
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(download_url, path=download_path, unzip=True)
    return kaggle.api.dataset_list_files(download_url).files
    
    
    
def import_load_data(csv_path):
    """The purpose of this function is to detect the encoding of the csv file and load it in to a DataFrame"""
    with open(csv_path, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(100000))
        encoding = result.get("encoding")
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except:
        print("Bad encoding, please correct the function")
    return df

    
    
    
def download_import_kaggle():
    """This function loads a csv from a Kaggle site and return a DataFrame. It can load several csv files in one DataFrame."""
    download_root = input("Please enter the url where your desired csv is: \n")
    download_path = "../INPUT"
    
    download_url = download_root.split("/", 3)[3]
    csv_files = fetch_data(download_url, download_path)
        
    if len(csv_files) == 1:
        csv_path = "../INPUT/" + str(csv_files[0])

    else:
        csv_files = input(f"Please what file/s did you need to import from this list {csv_files}?(I you select more than one csv, separate the files with comas please): \n")
        csv_files = list(map(str.strip, csv_files.split(",")))
        if len(csv_files) == 1:
            csv_path = "../INPUT/" + csv_files[0]
            return import_load_data(csv_path)
        else:
            csv_paths = [f"../INPUT/{file}" for file in csv_files]
            df_dic = {path.split("/")[-1]: import_load_data(path) for path in csv_paths}
            df =  pd.concat(df_dic.values())
            return df