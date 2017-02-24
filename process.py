import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    print(data)

    X = data[:, :-1]  # Every thing but the Last column
    Y = data[:, -1]   # Last column only
    print(X)
    print(Y)

    # 0 based index.  Set Column 2 and 3
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    print(X[:, 1])
    print(X[:, 2])

    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    Z = np.zeros((N, 4))
    Z[np.arange(N), X[: , D-1].astype(np.int32)] = 1
    # X2[:, -4:] = Z
    assert(np.abs(X2[:, -4:] - Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    print('X - ', X)
    print('Y - ', Y)

    # Filter X and Y rows based on whether Y is 0 or 1 i.e. anything in Y that is 2 or greater is excluded.
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    print('X2 - ', X2)
    print('Y2 - ', Y2)
    return X2, Y2

get_binary_data()
