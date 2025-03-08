class LinearRegression:
    def __init__(self):
        self.m = 0
        self.b = 0

    def fit(self, X_train, y_train):
        number = 0
        denominator = 0


        for i in range(X_train.shape[0]):
            denominator += X_train[i] - X_train.mean() * (X_train[i] - X_train.mean())
            number += (X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean())

        self.m = number / denominator
        self.b = y_train.mean() - self.m * X_train.mean()


    def predict(self, X):
        return self.m * X + self.b