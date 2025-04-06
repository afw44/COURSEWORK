from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import sys
from data import load_data
#df = load_data(from_month=1, to_month=2)[['epoch', 'res_bool', 'area_int','Wind Offshore', 'Wind Onshore', 'ExpectedFossilRequired']]

def update_prep(coldict):

    df = load_data(from_month=1, to_month=2)
    print(coldict[:-1])
    print('check1')
    df['ExpectedFossilRequired'] = df['TotalLoadValue'] - df['Solar'] - df['Wind Onshore'] - df['Wind Offshore']
    print('check1')
    df['res_bool'] = df['ResolutionCode'].apply(lambda x: np.argwhere(x == df['ResolutionCode'].unique())[0][0])
    print('check1')
    df['area_int'] = df['AreaCode'].apply(lambda x: np.argwhere(x == df['AreaCode'].unique())[0][0])
    print('check1')
    df['minute'] = df['DateTime'].apply(lambda x: x.minute)
    df['dayofweek'] = df['DateTime'].apply(lambda x: x.dayofweek)
    df['weekofyear'] = df['DateTime'].apply(lambda x: x.week)

    X = normalize(df[coldict[:-1]].to_numpy(), axis=0, norm='max')
    y = df[coldict[-1]]

    np.savetxt("stored_dfs/X.csv", X, delimiter=",")
    np.savetxt(f"stored_dfs/y.csv", y, delimiter=",")

    return X, y

def model(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    regr1 = MLPRegressor(random_state=3,
                         max_iter=100,
                         tol=0.1,
                         learning_rate='adaptive',
                         solver='adam',
                         hidden_layer_sizes=(20, 50, 100, 100, 50, 20))

    regr1.fit(X_train, y_train)

    regr2 = LinearRegression()
    regr2.fit(X_train, y_train)

    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (15,15))

    axs[0].scatter(y_test,
                   regr1.predict(X_test),
                   s=1)
    axs[0].plot(np.linspace(0,400,10),
                np.linspace(0,400,10),
                color='red',
                linestyle='--',
                linewidth=3)
    axs[0].set_xlim(0,150)
    axs[0].set_ylim(0,150)


    axs[1].scatter(y_test,
                   regr2.predict(X_test),
                   s=1)
    axs[1].plot(np.linspace(0, 400, 10),
                np.linspace(0, 400, 10),
                color='red',
                linestyle='--',
                linewidth=3)
    axs[1].set_xlim(0,150)
    axs[1].set_ylim(0,150)

    plt.show()

def main():

    update = False

    coldict = ['TotalLoadValue',
               'ExpectedFossilRequired',
               'res_bool',
               'area_int',
               'dayofweek',
               'weekofyear',
               'Price[Currency/MWh]']

    if update:
        X,y = update_prep(coldict)
    else:
        X = np.genfromtxt('stored_dfs/X.csv', delimiter=',')
        y = np.genfromtxt('stored_dfs/y.csv', delimiter=',')


    model(X,y)

    return

if __name__ == '__main__':
    main()


