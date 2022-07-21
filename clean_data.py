import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

file_errors_location = 'Data4Dipesh/Ash/Ash_Front.xlsx'
df_init = pd.read_excel(file_errors_location)
temps = [225, 250, 275, 300, 325]
times = [10, 20, 30, 40, 50]

clean_data = pd.DataFrame(columns=df_init.columns)

for temp in temps:
    for time in times:
        df = pd.read_excel(file_errors_location, sheet_name=f'{temp}-{time}')
        # pdb.set_trace()

        # df['Point_No']
        # df = df[df['Point_No']>4 and df['Point_No']<= 20]
        df = df[df['Point_No']<= 20]
        df = df[df['Point_No']>4]
        clean_data = pd.concat([clean_data, df], axis=0)
        
        # print(df2.columns[4], df2.columns[7])
        # after = df[df.columns[7]]
        # before = df[df.columns[4]]
        # k = (after-before)/before
            

clean_data.to_csv('Data4Dipesh/Ash/Ash_Front_Clean.csv')

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