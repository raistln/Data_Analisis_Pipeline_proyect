import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sidetable
import os
from fpdf import FPDF
import dataframe_image as dfi

title = 'Data Analisis Pipeline Report'

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Calcular ancho del texto (title) y establecer posición
        w = self.get_string_width(title) + 6
        self.set_x((210 - w) / 2)
        # Colores del marco, fondo y texto
        self.set_draw_color(0, 80, 180)
        self.set_fill_color(143, 220, 154)
        self.set_text_color(1, 1, 1)
        # Grosor del marco (1 mm)
        self.set_line_width(1)
        # Titulo
        self.cell(w, 9, title, 1, 1, 'C', 1)
        # Salto de línea
        self.ln(10)

    def footer(self):
        # Posición a 1.5 cm desde abajo
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Color de texto en gris
        self.set_text_color(128)
        # Numero de pagina
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', '', 14)
        # Color de fondo
        self.set_fill_color(200, 220, 255)
        # Titulo
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Salto de línea
        self.ln(4)
        
    def print_chapter(self, num, title):
        self.add_page()
        self.chapter_title(num, title)
        #self.chapter_body(name)
        
    def print_columns(self, num, title):
        self.set_font('Arial', '', 14)
        self.set_fill_color(150, 220, 255)
        self.cell(0, 6, 'Column %d : %s' % (num, title), 0, 1, 'L', 1)
        # Salto de línea
        self.ln(4)
        
    def chapter_body(self, df):
        self.set_font('Arial', '', 14)
        self.df = df
        num = 1
        for col in self.df.columns:
            if df[col].dtype == "float64":
                self.add_page()
                self.set_font('Times', '', 12)
                self.ln()
                self.print_columns(num, col)
                stats, table = column_analisis(col)
                top = self.y -10 
                offset = self.x + 50
                self.cell(40, 10,  f"mean   = {round(stats[1],2)}",0,1,"L",1)
                self.cell(40, 10,  f"median = {round(stats[5],2)}",0,1,"L",1)
                self.cell(40, 10,  f"std    = {round(stats[2],2)}",0,1,"L",1)
                self.cell(40, 10,  f"min    = {round(stats[3],2)}",0,1,"L",1)
                self.cell(40, 10,  f"Q1     = {round(stats[4],2)}",0,1,"L",1)
                self.cell(40, 10,  f"Q3     = {round(stats[6],2)}",0,1,"L",1)
                self.cell(40, 10,  f"max    = {round(stats[7],2)}",0,1,"L",1)
                self.cell(40,10)
                self.image(f"{col}.png", offset, top)
                self.y = top
                self.x = offset

            elif df[col].dtype.name == "category":
                self.add_page()
                self.set_font('Times', '', 12)
                self.ln()
                self.print_columns(num, col)
                stats, table = column_analisis(col)
                dfi.export(table,"table.png")
                self.image(f"table.png")
                self.image(f"{col}.png")
            num += 1
            
    def first_chapter_body(self, df):
        self.df = df
        df_info = info_df(self.df)
        first_transformation_report(df)
        dfi.export(df_info,"df_info.png")
        self.image("df_info.png", w = 180, h = 260)
        self.image("nulls.png", x = 100)
    
    def last_chapter(self, df):
        self.add_page("L")
        self.set_font('Times', '', 12)
        self.ln()
        self.df = df
        dfi.export(self.df.tail(), "tail_df.png")
        dfi.export(self.df.head(), "head_df.png")
        self.image("head_df.png", w = 260, h = 180)
        self.image("tail_df.png", w = 260, h = 180)


def info_df(df): 
    name = {"Name":[]}
    non_null = {"Non_null":[]}
    dtypes = {"Dtype":[]}
    number = {"Nº":[]}
    null_per = {"%_of_null": []}
    num = 1
    for col in df.columns:
        number["Nº"].append(num)
        name["Name"].append(col)
        non_null["Non_null"].append(df[col].notnull().sum())
        null_per["%_of_null"].append(round(100*df[col].isnull().sum()/df.shape[0],1))
        dtypes["Dtype"].append(df[col].dtype)
        num += 1
    info_df = pd.DataFrame(dict(**number,**name,**non_null,**null_per,**dtypes)).set_index("Nº")
    return info_df

def first_transformation_report(df):
    instant_df = pd.DataFrame()
    cols = df.columns
    for col in cols:
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
    plt.savefig("nulls.png")

    
def plot_num(col):
    Q1 = df[col].describe()[4]
    Q3 = df[col].describe()[6]
    figure , axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(df[col], bins=100)
    axes[1].hist(df[col][(df[col]>Q1) & (df[col]<Q3)], bins=100)
    plt.savefig(f"{col}.png")
    
def plot_cat(col):
    dic = dict(df[col].value_counts())
    bars = plt.barh(list(dic.keys()), dic.values(), color = "tab:green")
    bars[0].set_color("r")
    plt.gca().invert_yaxis()
    plt.savefig(f"{col}.png")

    
def column_analisis(col):
    a = df[col].describe()
    b = df.stb.freq([col])
    if df[col].dtype == "float64":
        plot_num(col)
    elif df[col].dtype.name == "category":
        plot_cat(col)
    return a, b

def pdf_report(df):   
    pdf = PDF()
    pdf.set_title(title = 'Data Analisis Pipeline Report')
    pdf.set_author('Samuel Martín')
    pdf.print_chapter(1, 'Global Overview')
    pdf.first_chapter_body(df)
    pdf.print_chapter(2, 'Columns Analisis')
    pdf.chapter_body(df)
    pdf.print_chapter(3, "Head and Tail")
    pdf.last_chapter(df)
    pdf.output('../OUTPUT/Dataframe_clean_analisis.pdf', 'F')
    list(map(lambda x: os.remove(x), list(filter(lambda x: x.endswith(".png"), os.listdir()))))