# Data_Analisis_PIpeline_proyect

The proyect is totally done by myself, I know it´s can be better. But time to the time. I will continue with it when i can.

The intention of this proyect is to obtain a more or less cleaned pandas dataframe to work on it and practice plots and machine learning skills. The project was born from the need to have dataframes in a short period of time.

That is a very simple data analisis pipeline[data_analisis_pipeline()]. Made only with more or less simple python code. It has 3 parts. 
First [download_import_kaggle()] It takes a csv file from [Kaggle]("https://www.kaggle.com/") and import it to a new folder called [datasets]. 
Second [first_view(df)] it gives a first view of the dataset and ask the user if she/he wants to clean duplicates/empty rows and columns, standarize columns and rows (putting all to lower and strip trailing whitespaces). It also ask the user if he/she want to use all de dataframe or just some columns.
Third [data_wrangling(df)], where it transforms, asking the user, the dataframe. First it search for numeric columns that are non numeric types and transform it if the user wants, then it makes the same with categorical types columns and at least it ask the user if she/he wants to fill the [nan] in the numerical types columns with [media] or [mean], and the categorical columns with [mode].

This is a good begining to have a fast solution to clean the dataframes to train other skills like diferent plots or machine learning skills.
