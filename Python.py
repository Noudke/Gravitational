import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import vaex
import time

# "file" is train data from D:\Datasets\g2net-detecting-continuous-gravitational-waves (1)
# "file2" is test data from D:\Datasets\g2net-detecting-continuous-gravitational-waves (1)

train_folder = 'D:\\Datasets\\g2net-detecting-continuous-gravitational-waves (1)\\train'
test_folder = 'D:\\Datasets\\g2net-detecting-continuous-gravitational-waves (1)\\test'

# List of names of all files in the folder
train_files = os.listdir(train_folder)
test_files = os.listdir(test_folder)

train_labels = 'D:\\Datasets\\g2net-detecting-continuous-gravitational-waves (1)\\train_labels.csv'

# Create a dictionary with train_files values as keys and train_labels values as values
train_labels_dict = {}
with open(train_labels, 'r') as f:
    for line in f:
        (key, val) = line.split(',')
        train_labels_dict[key] = val

# Remove "\n" from the end of the values
for key in train_labels_dict:
    train_labels_dict[key] = train_labels_dict[key].rstrip()

# Get contents of first train file
with h5.File(train_folder + '\\' + train_files[0], 'r') as f:
    for file_key in f.keys():
        group = f[file_key]
        print(group)
        try:
            for group_key in group.keys():
                group2 = group[group_key]
                print(f"---->{group2}")
                for group_key2 in group2.keys():
                    print(f"--------->{group2[group_key2]}")
        except AttributeError:
            pass


def traverse_hdf5(hdf5_file, file_name):
    data = []
    # Open the HDF5 file
    with h5.File(hdf5_file, 'r') as f:
        # Recursively traverse through the groups in the HDF5 file
        def traverse(name, path):
            # Get the object at the current path
            obj = f[name]
            # If the object is a group, traverse through it
            if isinstance(obj, h5.Group):
                for key in obj.keys():
                    traverse(f"{name}/{key}", f"{path}/{key}")
            # If the object is a dataset, store the data and the dataset name in the list
            elif isinstance(obj, h5.Dataset):
                data.append((obj[()], path, file_name))

        traverse("/", "/")
    # Create a Pandas dataframe from the data
    df = pd.DataFrame(data, columns=["data", "name", "file"])

    # If the end of the column of a name is "SFTs", replace the value within the data column with the average of the values
    for i in range(len(df)):
        if df['name'][i][-4:] == 'SFTs':
            df['data'][i] = np.mean(df['data'][i])

    # If the end of the column of a name is "frequency_Hz", replace the value within the data column with the average of
    # the values
    for i in range(len(df)):
        if df['name'][i][-12:] == 'frequency_Hz':
            df['data'][i] = np.mean(df['data'][i])

    # Remove the columns with names that end with "timestamps_GPS"
    df = df[~df['name'].str.endswith('timestamps_GPS')]

    # Transpose the dataframe, with eventually columns of "Filename", "H1-SFTs", "L1-SFTs", "Frequency_Hz"
    df = df.pivot(index='file', columns='name', values='data').reset_index()

    # Rename the "Filename" column to "id"
    df.rename(columns={'file': 'Id'}, inplace=True)

    # Filename is the name of the file without the extension
    Filename = file_name[:-5]

    # Rename the "H1-SFTs" column to "H1"
    df.rename(columns={'//' + Filename + '/H1/SFTs': 'Avg H1-SFTs'}, inplace=True)
    df.rename(columns={'//' + Filename + '/L1/SFTs': 'Avg L1-SFTs'}, inplace=True)
    df.rename(columns={'//' + Filename + '/frequency_Hz': 'Avg Frequency_Hz'}, inplace=True)
    print(Filename)

    return df


complete_df = pd.DataFrame(columns=["Id", "Avg H1-SFTs", "Avg L1-SFTs", "Avg Frequency_Hz"])

# Execute the traverse_hdf5 function on all files in the train folder and replace complete_df with complete_df + df
for file in train_files:
    df = traverse_hdf5(train_folder + '\\' + file, file)
    complete_df = complete_df.append(df)

print(complete_df.head())


# Function to convert scientific notation to float
def convert_to_float(x):
    return float(x)


# Remove ".hdf5" from the end of the Id column for each row
complete_df['Id'] = complete_df['Id'].str[:-5]

# Convert the values in the "Avg H1-SFTs" column to float
complete_df['Avg H1-SFTs'] = complete_df['Avg H1-SFTs'].apply(convert_to_float)

# Convert the values in the "Avg L1-SFTs" column to float
complete_df['Avg L1-SFTs'] = complete_df['Avg L1-SFTs'].apply(convert_to_float)


labels_df = pd.read_csv(train_labels)

labels_df = labels_df.rename(columns={"id": "Id"})

complete_df = pd.merge(complete_df, labels_df, on="Id")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Separate the independent variables (predictor variables) from the dependent variable (response variable)
X = complete_df.drop(columns=["Id", "target"])
y = complete_df["target"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create an instance of the LogisticRegression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

complete_df_test = pd.DataFrame(columns=["Id", "Avg H1-SFTs", "Avg L1-SFTs", "Avg Frequency_Hz"])

# Execute the traverse_hdf5 function on all files in the train folder and replace complete_df with complete_df + df
for file in test_files:
    df = traverse_hdf5(test_folder + '\\' + file, file)
    complete_df_test = complete_df_test.append(df)


# Function to convert scientific notation to float
def convert_to_float(x):
    return float(x)


# Remove ".hdf5" from the end of the Id column for each row
complete_df_test['Id'] = complete_df_test['Id'].str[:-5]

# Convert the values in the "Avg H1-SFTs" column to float
complete_df_test['Avg H1-SFTs'] = complete_df_test['Avg H1-SFTs'].apply(convert_to_float)

# Convert the values in the "Avg L1-SFTs" column to float
complete_df_test['Avg L1-SFTs'] = complete_df_test['Avg L1-SFTs'].apply(convert_to_float)

print(complete_df_test.head())


X_test = complete_df_test.drop(columns=["Id"])

# Make predictions on the test data
y_pred_test = model.predict(X_test)

# Create csv file with first column being test_files without the .hdf5 extension and second column being the predictions
submission = pd.DataFrame({'Id': test_files, 'target': y_pred_test})
submission['Id'] = submission['Id'].str[:-5]
submission.to_csv('submission.csv', columns=["Id", "target"], index=False)

