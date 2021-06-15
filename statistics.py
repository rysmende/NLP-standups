import json
import joblib as jl

X = jl.load('X_train')
    
print("Current data: ", len(X), "\nTotal data: 27690")