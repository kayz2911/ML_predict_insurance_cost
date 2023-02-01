import pandas as pd
import pickle

# Loading Data
data = pd.read_csv("finaldata.csv")


def value_predictor(df):
    loaded_model = pickle.load(open("model3DT.pkl", "rb"))
    result = loaded_model.predict(df)
    return result[0]


df = pd.DataFrame(columns=data.iloc[:, :6].columns)
new_vl = []
for x in data.iloc[:, :6].columns:
    value = input(f"Please enter {x} :")
    new_vl.append(value)
df.loc[len(df.index)] = new_vl

result = value_predictor(df)
print("Predict result :", result)
