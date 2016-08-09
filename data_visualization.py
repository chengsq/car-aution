#encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def Stripplot(x_,y_,data_):
    sns.set(style="whitegrid", color_codes=True)
    sns.stripplot(x = x_, y=y_, data = data_, jitter=True);

def PairPlot(df):
    sns.set()
    sns_plot = sns.pairplot(df)
    sns_plot.savefig("pairplot.png")

def BarPlot(x_,y_,df):
    sns.set()
    ax = sns.barplot(y=y_,x=x_, data=df)

def CountPlot(y_s,hue_s,df):
    sns.set()
    sns.countplot(y=y_s, hue=hue_s, data=df, palette="Greens_d");

def DistPlot(df):
    sns.set()
    sns.distplot(df,bins = 10,kde = False)

def FactorPlot(index,df):
    sns_plot = sns.factorplot(index, col="deck", col_wrap=4,
      data=df,
      kind="count", size=2.5, aspect=.8)

def JoinPlot(x_,y_,df):
    sns_plot = sns.jointplot(x=x_, y= y_, data=df,color="g")
    sns_plot.savefig("joinplot.png")

def Regplot(x_,y_,df):
    plt.figure(x_+'-'+y_)
    plt.title(x_+'-'+y_)
    sns_plot = sns.regplot(x=x_, y= y_, data=df,color="g",x_jitter=.1,order=1)

if __name__ == '__main__':
    file_name = "data/tianjin_jingjia_csv.csv"

    df = pd.read_csv( file_name)
    print df.head()
    #BarPlot('','',df)
    #CountPlot('IDATACOMPLETIONSCORES','target',df)
    #DistPlot(df['IDATACOMPLETIONSCORES'])
    #FactorPlot('IDATACOMPLETIONSCORES',)
    #JoinPlot('IDATACOMPLETIONSCORES','target',df)
    #DistPlot(df['mean'])
    #JoinPlot('first','mean',df)
    Regplot('month','second',df)
    #PairPlot(df)
    plt.show()
