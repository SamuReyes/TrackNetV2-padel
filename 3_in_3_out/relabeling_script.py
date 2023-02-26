import pandas as pd

base_root = "D:/archivos_locales/TFG/datasets/tracknetV2_padel/test/4/"

label_df = pd.read_csv(base_root + "Label.csv")

# Create a copy of the dataframe
new_label_df = label_df

# Rename columns
new_label_df.rename(columns={'file name': 'Frame', 'visibility': 'Visibility',
                    'x-coordinate': 'X', 'y-coordinate': 'Y', 'status': 'Status'}, inplace=True)

# Delete .jpg extension from file name column
new_label_df['Frame'] = new_label_df['Frame'].astype(str).str[:-4]

# Change states (0-> no ball, 1->frying, 2->hit, 3->bouncing at ground, 4-> bouncing at wall)
new_label_df.loc[new_label_df["Status"] == 2, "Status"] = 3
new_label_df.loc[new_label_df["Status"] == 1, "Status"] = 2
new_label_df.loc[new_label_df["Status"] == 0, "Status"] = 1
new_label_df["Status"].fillna(0, inplace=True)

# Create new state (bouncing at the wall)
new_label_df.loc[new_label_df["Visibility"] == 2, "Status"] = 4

# Recover original visibility
new_label_df.loc[new_label_df["Visibility"] == 2, "Visibility"] = 1

# For vsibility = 0, take 0 as X and Y
new_label_df["X"].fillna(0, inplace=True)
new_label_df["Y"].fillna(0, inplace=True)

# Create new parameter (Occluded)
new_label_df["Occluded"] = 0

# Move occluded status to the new parameter
new_label_df.loc[new_label_df["Visibility"] == 3, "Occluded"] = 1
new_label_df.loc[new_label_df["Visibility"] == 3, "Visibility"] = 0 # Set 0 if we dont want to predict where is a occluded ball and change X and Y values to 0
#new_label_df.loc[new_label_df["Visibility"] == 0, "X"] = 0
#new_label_df.loc[new_label_df["Visibility"] == 0, "Y"] = 0

new_label_df[new_label_df["Status"]==4]

# Save the file with all the columns (future purposes)
new_label_df.to_csv(base_root + "FullLabel2.csv", index = False)

# Save the file
new_label_df = new_label_df.drop("Status", axis=1)
new_label_df = new_label_df.drop("Occluded", axis=1)
new_label_df.to_csv(base_root + "TNLabel2.csv", index = False)
