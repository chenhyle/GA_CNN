import pandas as pd
import os

# generated u_6, u_12,u_18,and v_6, v_12,v_18
folder_path ="E:\\py_practice\\typhoon\\era5_data_02\\"
paths = ['u', 'v']
new_path = None

for pre in [1, 2, 3]:
    for path in paths:
        path = os.path.join(folder_path,path)
        files = os.listdir(path)
        print(files)
        files.sort()
        for file in files:
            analysis = pd.read_csv(path+'/'+file, header=None)
            tid = analysis[4].unique()
            for tid in analysis[4].unique():
                indexes = analysis[analysis[4] == tid].index
                for i in indexes:
                    if i + pre <= indexes[-1]:
                        analysis.at[i, 0] = analysis.at[i + pre, 0]
                        analysis.at[i, 1] = analysis.at[i + pre, 1]
                        analysis.at[i, 2] = analysis.at[i + pre, 2]
                        analysis.at[i, 3] = analysis.at[i + pre, 3]
                    else:
                        analysis.drop(index=i, inplace=True)
            analysis.reset_index(drop=True, inplace=True)
            analysis[[0, 1]] = analysis[[0, 1]].astype(str)
            new_path = path+'_'+str(pre*6)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            analysis.to_csv(new_path + '/' + file, header=False, index=False)
            print(file + ' is done!')
