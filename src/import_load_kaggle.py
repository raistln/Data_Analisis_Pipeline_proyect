import kaggle
import os
import pandas as pd

ROOT = os.getcwd()


def fetch_data(download_url, download_path):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(download_url, path=download_path, unzip=True)
    return kaggle.api.dataset_list_files(download_url).files
    
    
    
def import_load_data(csv_path, encoding="ISO-8859-1"):
    
    encoding = input("Default enconding is ISO-8859-1, do you prefer another one? Please type it.\n")
    
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    return df
    
    
    
def download_import_kaggle():
    
    download_root = input("Please enter the url where your desired csv is: \n")
    download_path = "../INPUT"
    
    download_url = download_root.split("/", 3)[3]
    csv_files = fetch_data(download_url, download_path)

    if len(csv_files) == 1:
        csv_path = ROOT + "../INPUT/" + str(csv_files[0])
    else:
        csv_file = input(f"Please what file did you need to import from this list {csv_files}?: \n")
        csv_path = ROOT + "../INPUT/" + csv_file
    return import_load_data(csv_path, encoding="ISO-8859-1")