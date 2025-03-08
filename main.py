from LinearReggression import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    lr = LinearRegression()

    dataFrame = pd.read_csv(filepath_or_buffer = "placement.csv")
    X = dataFrame.iloc[:, 0].values
    y = dataFrame.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2)

    lr.fit(X_train, y_train)

    print(lr.predict(X = 6.22))



