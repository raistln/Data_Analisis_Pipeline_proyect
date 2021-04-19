from import_load_kaggle import download_import_kaggle
from first_view import first_view
from data_wrangling import data_wrangling
from pdf_report import pdf_report

def main():
    df = download_import_kaggle()
    df = first_view(df)
    df = data_wrangling(df)
    pdf_report(df)
    
if __name__ == "__main__":
    main()