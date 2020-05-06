import pandas as pd
df=pd.read_csv('Sample4 - Iris.csv')
print(df)
if 'Target'  in df.columns:
    print('hllo')
else:
    pass



