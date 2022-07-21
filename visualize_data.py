import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

file_errors_location = 'Data4Dipesh/Ash/Ash_Front.xlsx'
# df = pd.read_excel(file_errors_location)
df = pd.read_excel(file_errors_location, sheet_name='325-40')
# print(df[24:27])
# print(df2.head())
print(df2.columns[4], df2.columns[7])
after = df[df2.columns[7]]
before = df[df2.columns[4]]

k = (after-before)/before

# print(df2[0:4])
# print(df2.shape)

# plot two plots in one figure
first_few = 72
# pdb.set_trace()
mean_L = np.mean([df[df2.columns[4]][:24], df[df2.columns[4]][24:48], df[df2.columns[4]][48:72]], axis=0)
L_diff = df[df2.columns[4]] - df[df2.columns[7]]

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(df[df2.columns[4]][:first_few])
ax[0].plot(mean_L)
ax[1].plot(L_diff[:first_few])
ax[2].plot(k[:first_few])
plt.savefig('plots/Ash_Back.png')
# plt.show()


# plt.plot(df[df2.columns[4]][:24])
# plt.plot(L_diff[:24])