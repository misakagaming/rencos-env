import pandas as pd

data = []

with open("train.txt.src") as source, open("train.txt.tgt") as target: 
    for x, y in zip(source, target):
        x = x.strip()
        y = y.strip()
        data.append([x,y])

df = pd.DataFrame(data, columns=['code', 'summary'])

print(df)

df.to_csv("train.csv", index=False)