import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df1 = df.code
df1 = df1.replace(r'\n',' ', regex=True)
df1 = df1.replace(r'\t',' ', regex=True)
df1 = df1.replace(r'\s', ' ', regex = True)
df1.to_csv('train.src.csv', index=False, header=False)
df2 = df.summary
df2.to_csv('train.tgt.csv', index=False, header=False)
    
df = pd.read_csv('test.csv')
df1 = df.code
df1 = df1.replace(r'\n',' ', regex=True)
df1 = df1.replace(r'\t',' ', regex=True)
df1 = df1.replace(r'\s', ' ', regex = True)
df1.to_csv('test.src.csv', index=False, header=False)
df2 = df.summary
df2.to_csv('test.tgt.csv', index=False, header=False)

    
df = pd.read_csv('valid.csv')
df1 = df.code
df1 = df1.replace(r'\n',' ', regex=True)
df1 = df1.replace(r'\t',' ', regex=True)
df1 = df1.replace(r'\s', ' ', regex = True)
df1.to_csv('valid.src.csv', index=False, header=False)
df2 = df.summary
df2.to_csv('valid.tgt.csv', index=False, header=False)

