import pdb
import pandas as pd
import numpy as np

exp_name = "YellowPoplar"
file_errors_location = f'Data4Dipesh/{exp_name}/{exp_name}_Front.xlsx'
df_init = pd.read_excel(file_errors_location)
temps = [225, 250, 275, 300, 325]
times = [10, 20, 30, 40, 50]

columns=['temp','time','operation','L_before', 'a_before', 'b_before', 'L_after', 'a_after', 'b_after','delta_L','delta_a','delta_b','color_diff_E']
file_mean_var = pd.DataFrame(columns=columns)

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
        # df.keys()
        delta_L = (L_after-L_before)
        delta_a = (a_after-a_before)
        delta_b = (b_after-b_before)

        color_diff = np.sqrt((L_after-L_before)**2 + (a_after-a_before)**2 + (b_after-b_before)**2)
        color_diff_mean = np.mean(color_diff)
        color_diff_std = np.std(color_diff)
        # pdb.set_trace()
        file_mean_var.loc[len(file_mean_var.index)] = [temp, time, "mean", L_before.mean(), a_before.mean(), b_before.mean(), L_after.mean(), a_after.mean(), b_after.mean(),delta_L.mean(), delta_a.mean(), delta_b.mean(), color_diff_mean] 
        file_mean_var.loc[len(file_mean_var.index)] = [temp, time, "std", L_before.std(), a_before.std(), b_before.std(), L_after.std(), a_after.std(), b_after.std(),delta_L.std(), delta_a.std(), delta_b.std(), color_diff_std] 

file_mean_var.to_csv(f'Data4Dipesh/{exp_name}/{exp_name}_Front_mean_var.csv')