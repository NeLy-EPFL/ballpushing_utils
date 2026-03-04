import pandas as pd

df = pd.read_feather('/mnt/upramdya_data/MD/F1_Tracks/Datasets/260217_19_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather')

print('All columns:')
for i, c in enumerate(df.columns):
    print(f'{i:2d}. {c}')

print('\n\nPosition-related columns:')
print([c for c in df.columns if any(x in c.lower() for x in ['x_', 'y_', '_x', '_y', 'centre', 'center'])])
