import json
from IPython.display import clear_output
import joblib as jl

with open("standups.json") as json_file:
    data = json.load(json_file)

X = jl.load('X_train')
Y = jl.load('Y_train')
to_save = False

for i in range(len(X), len(data)):
    clear_output(wait=True)
    node = data[i]
    t = int(input(node['fields']['text'] + "\n"))
    if (t < 0 or t > 3): 
        if (t == -1):
            to_save = True
        break

    X.append(node['fields']['text'])
    Y.append(t)

if to_save:
    jl.dump(X, 'X_train')
    jl.dump(Y, 'Y_train')
