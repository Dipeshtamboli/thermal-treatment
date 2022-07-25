import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

exp_name = "Ash"
file_errors_location = f'Data4Dipesh/{exp_name}/{exp_name}_Front.xlsx'
df_init = pd.read_excel(file_errors_location)
temps = [225, 250, 275, 300, 325]
times = [10, 20, 30, 40, 50]

mean_for_anova = pd.DataFrame(columns=['temp', 'time', 'color_diff_mean'])

for temp in temps:
    for time in times:
        file_name = f'{temp}-{time}'
        print(file_name)
        df = pd.read_excel(file_errors_location, sheet_name=file_name)

        # df = df[df['Point_No']>4 and df['Point_No']<= 20]
        df = df[df['Point_No']<= 20]
        df = df[df['Point_No']>4]
        
        L_after = df[df.columns[7]]
        L_before = df[df.columns[4]]
        a_after = df[df.columns[8]]
        a_before = df[df.columns[5]]
        b_after = df[df.columns[9]]
        b_before = df[df.columns[6]]

        color_diff = np.sqrt((L_after-L_before)**2 + (a_after-a_before)**2 + (b_after-b_before)**2)
        mean = np.mean(color_diff)
        std = np.std(color_diff)

        # print(f"before shape: {df.shape}")
        df = df[color_diff < mean+2*std]
        df = df[color_diff > mean-2*std]
        color_diff = color_diff[color_diff < mean+2*std]
        color_diff = color_diff[color_diff > mean-2*std]
        # print(f"after shape: {df.shape}")
        mean = np.mean(color_diff)
        # df2 = pd.DataFrame([df[df.columns[1]], df[df.columns[2]]]).transpose()
        # df2['color_diff'] = color_diff
        df2 = pd.DataFrame([[temp,time,mean]], columns=['temp', 'time', 'color_diff_mean'])
        # df2 = df
        mean_for_anova = pd.concat([mean_for_anova, df2], axis=0)


# 3d plot in matplotlib with temp and time as x and y and color_diff as z
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mean_for_anova['temp'], mean_for_anova['time'], mean_for_anova['color_diff_mean'])
ax.set_xlabel('temp')
ax.set_ylabel('time')
ax.set_zlabel('color_diff_mean')
plt.savefig(f"temp_vs_time_vs_color_diff.png")

f_one_result = f_oneway(mean_for_anova['temp'], mean_for_anova['time'], mean_for_anova['color_diff_mean'])

tukey = pairwise_tukeyhsd(endog=mean_for_anova['color_diff_mean'], groups=mean_for_anova['temp'], alpha=0.05)

pdb.set_trace()