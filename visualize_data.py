import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

file_errors_location = 'Data4Dipesh/Ash/Ash_Back.xlsx'
df = pd.read_excel(file_errors_location)
df2 = pd.read_excel(file_errors_location, sheet_name='225-20')
# print(df[24:27])
print(df2[0:4])
print(df2.shape)

# plt.plot(df[])