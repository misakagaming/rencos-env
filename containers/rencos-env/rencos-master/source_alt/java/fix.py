import numpy as np
import pandas as pd


files = ["test/test.spl.src", "valid/valid.spl.src", "train/train.spl.src"]
for file in files:
    with open(file, encoding="utf-8") as f:
        data = f.readlines()
    for i in range(len(data)):
        if data[i][0] == '"':
            data[i] = str(data[i][1:-2] + "\n")
    with open(file, 'w', encoding="utf-8") as file:
        file.writelines( data )