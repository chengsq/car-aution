#encoding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cross_validation
import numpy as np
from sklearn import datasets,linear_model
import data_visualization
import xgboost as xgb
import sys


def ReadData(file_name):
    df = pd.read_csv(file_name, index_col=0)
    x = df[['first','second','month']]
    ymin = df['min']
    ymean = df['mean']
    return df

def FeatureProcess(df):
    cur_mean = list(df['mean'])
    #cur_mean.insert(0,0)
    cur_mean.pop()
    cur_mean.insert(0,10000)
    df['l_mean'] = cur_mean
    cur_mean.insert(0,10000)
    cur_mean.pop()
    df['ll_mean'] = cur_mean

    df['r1'] = (df['mean'] - df['first'])/df['first']
    df['r2'] = (df['mean'] - df['second'])/df['second']
    df['r3'] = (df['second'] - df['first'])/df['first']
    df['r4'] = (df['mean'] - df['l_mean'])/df['l_mean']

    df['l_mean_norm'] = df['l_mean']/ df['l_mean']
    df['ll_mean_norm'] = df['ll_mean']/ df['l_mean']
    df['first_norm'] = df['first']/ df['l_mean']
    df['second_norm'] = df['second']/ df['l_mean']
    df['mean_norm'] = df['mean']/ df['l_mean']


    #df['d1'] = df['mean'] - df['l_mean']

    return df

def DatasetSplit(x,y,size):
    index = int(len(y)*(1-size))
    print index
    train_x = x.iloc[2:index,:]
    test_x = x.iloc[index:,:]
    train_y = y.iloc[2:index]
    test_y = y.iloc[index:]

    return train_x,test_x,train_y,test_y


def EvaluateDrift(gy,ty):
    result = [0] * len(gy)
    gy = list(gy)
    for i in range(len(gy)):
        result[i] = (gy[i] - ty[i])/gy[i]
        #print gy[i]

    return result

def ScatterPlot(x,y,yp):
    plt.scatter(x, y, color='r',linewidth=2,label='y-ground')
    plt.scatter(x,yp, color='blue',linewidth=2,label='y-predict')
    plt.xticks(())
    plt.yticks(())
    plt.legend()

def ShowPlot(x,y,yp):
    plt.scatter(x, y, color='r',linewidth=2,label='y-ground')
    plt.plot(x,yp, color='blue',linewidth=2,label='y-predict')
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()
    plt.legend()


def XgboostRegression():

    reg = xgb.XGBRegressor(max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                silent=True,
                objective='reg:linear',
                nthread=-1,
                gamma=0,
                min_child_weight=1,
                max_delta_step=0,
                subsample=1,
                colsample_bytree=1,
                colsample_bylevel=1,
                reg_alpha=0, reg_lambda=1,
                scale_pos_weight=1,
                base_score=0.5,
                seed=0,
                missing=None)





def ModelProcess(x,y,n = 1):
    # Split the data into training/testing sets
    #diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = \
    #          cross_validation.train_test_split(x, y,random_state =0 ,test_size=0.1)

    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = DatasetSplit(x,y,0.1)

    print diabetes_X_train.shape,diabetes_X_test.shape,diabetes_y_train.shape, \
        diabetes_y_test.shape

        # Create linear regression object
    #regr = linear_model.LinearRegression()
    regr = linear_model.Ridge(alpha = 1.0, fit_intercept=False)
    #
    '''
    regr = xgb.XGBRegressor(max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                silent=True,
                objective='reg:linear',
                nthread=-1,
                gamma=0,
                min_child_weight=1,
                max_delta_step=0,
                subsample=1,
                colsample_bytree=1,
                colsample_bylevel=1,
                reg_alpha=0, reg_lambda=1,
                scale_pos_weight=1,
                base_score=0.5,
                seed=0,
                missing=None)
    '''

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)



    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean square error
    error =  np.mean(np.abs(regr.predict(diabetes_X_test) - diabetes_y_test))
    print("Residual sum of squares: %.2f"
      % error)

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    '''
    drift = EvaluateDrift(y,regr.predict(x))
    drift = map(abs,drift)
    print('score dift ', drift)
    print "max  min    mean   median"
    print max(drift),min(drift),np.mean(drift),np.median(drift)
    '''

    # Plot outputs
    #plt.scatter(diabetes_X_test[], diabetes_y_test,  color='black')

    plot_x = range(len(x))
    plot_yp = regr.predict(x)


    print list(diabetes_y_test)
    print (list(regr.predict(diabetes_X_test)))

    ShowPlot(plot_x,y,plot_yp)
    plt.show()

    return error



def GridResearch(x,y):
    length = [10] #8,10,11,12,13,14,15,16,18,20]
    sample_num = len(y)

    total = []
    ll = []
    for l in length:
        p = []
        for index in range(2,sample_num - l):
            x_sample = x.iloc[index:index + l,:]
            y_sample = y[index:index + l]
            v = ModelProcess(x_sample,y_sample)
            p.append(v)
        print p[-1]
        ll.append(p[-1])
        total.append(np.mean(p))

    print total,ll



def PolyFit(x,y):
    x = np.asarray(x)
    y = np.asarray(y)

    print x.shape,y.shape
    z = np.polyfit(x, y,2)
    f = np.poly1d(z)


    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)

    print f(30)

    plt.plot(x,y,'o', x_new, y_new)
    plt.xlim([x[0]-1, x[-1] + 1 ])
    plt.show()

# main start
if __name__ == '__main__':
    file_name = "data/tianjin_jingjia_csv.csv"
    df = pd.read_csv(file_name)
    df = FeatureProcess(df)
    #x = df[['l_mean','first','second']]

    data_visualization.Regplot('first','second',df)
    data_visualization.Regplot('l_mean','first',df)
    data_visualization.Regplot('l_mean','mean',df)
    data_visualization.Regplot('first','mean',df)
    plt.show()
    sys.exit()
    #x = x.iloc[-6:,:]
    #y = y[-6:]
    #plt.title('(mean-l_mean)/l_mean')
    #data_visualization.BarPlot('month','r4',df)
    #plt.figure()
    #data_visualization.DistPlot(df['r4'])
    #plt.show()
    #print "model processing"
    #ModelProcess(x,y)
