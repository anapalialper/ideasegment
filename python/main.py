import os
import pandas as pd

dir_path = '../app/public/assets/'

file_list = os.listdir(dir_path)
print(file_list)

excl_merged = pd.DataFrame()

for file in file_list:
    print(file + ": " + str(pd.read_excel(dir_path + file).shape[0]))
    excl_merged = excl_merged.append(pd.read_excel(dir_path + file), ignore_index=True)
print("done")
df = excl_merged

print(df[0])
