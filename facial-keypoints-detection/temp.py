import csv
import pandas as pd
import os



label_df = pd.read_csv(os.path.join(f'dataset', 'train', 'trainLabels.csv'))
print(label_df['level'])