import pandas as pd

data = []

with open("valid.txt.src") as source, open("valid.txt.tgt") as target: 
    for x, y in zip(source, target):
        x = x.strip()
        y = y.strip()
        data.append([x,y])

df = pd.DataFrame(data, columns=['code', 'summary'])

print(df)

df.to_csv("valid.csv", index=False)