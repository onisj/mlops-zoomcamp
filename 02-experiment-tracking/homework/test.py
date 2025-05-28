import pickle
with open("./output/val.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)
print(X_train.shape, y_train.shape)