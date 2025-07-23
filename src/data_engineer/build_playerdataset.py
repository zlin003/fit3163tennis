import os
import pandas as pd


data_folder = os.path.join('../..', 'data', 'atp1_2000-2024')


all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]


df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)


print(f"✅ Combined Data Shape: {df.shape}")
print(f"✅ Columns: {df.columns.tolist()}")


output_path = os.path.join('../..', 'src', 'output','player_dataset.csv')
df.to_csv(output_path, index=False)

print(f"✅ Saved merged player dataset to {output_path}")
