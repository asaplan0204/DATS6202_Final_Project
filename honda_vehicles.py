#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# loading the data
#%%
df = pd.read_csv("vehicles.csv")

# filtering data to just honda makes
# %%
df_honda = df[df["make"] == "Honda"]

# %%
print(df_honda.head())


# splitting the data into train, test and val sets
#%%
random_seed = 42

# %%
df_train, df_test = train_test_split(
    df_honda,
    train_size=0.8,
    random_state=random_seed
)
# %%
df_train, df_val = train_test_split(
    df_train,
    train_size=0.75,
    random_state=random_seed
)
# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# uncommon features
#%%
target = "comb08"

# %%
def common_var_checker(df_train, df_val, df_test, target):
    common_vars = np.intersect1d(
        np.intersect1d(df_train.columns, df_val.columns),
        df_test.columns
    )

    return pd.DataFrame(common_vars, columns=["common var"])


# %%
df_common_var = common_var_checker(
    df_train, df_val, df_test, target
)

print(df_common_var)


# handling identifiers, date time features, missing data
# %%
columns_to_keep = ["year", "displ", "cylinders", "drive", "trany", "VClass", "fuelType1", "model", "comb08"]
# %%
df_train = df_train[columns_to_keep].copy()
df_val = df_val[columns_to_keep].copy()
df_test = df_test[columns_to_keep].copy()

# %%
print("Train missing values:\n", df_train.isnull().sum())
print("Validation missing values:\n", df_val.isnull().sum())
print("Test missing values:\n", df_test.isnull().sum())

# %%
df_train = df_train.dropna().reset_index(drop=True)
df_val = df_val.dropna().reset_index(drop=True)
df_test = df_test.dropna().reset_index(drop=True)

# %%
print("Train missing values:\n", df_train.isnull().sum())
print("Validation missing values:\n", df_val.isnull().sum())
print("Test missing values:\n", df_test.isnull().sum())
# %%
print(df_train.shape[0])
print(df_val.shape[0])
print(df_test.shape[0])

# encoding categorical features
# %%
numeric_features = ["year", "displ", "cylinders"]

categorical_features = [
    "drive", "trany", "VClass", "fuelType1", "model"
]

#%%

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Fit only on training data
X_train_cat = encoder.fit_transform(df_train[categorical_features])

# Transform validation and test data
X_val_cat = encoder.transform(df_val[categorical_features])
X_test_cat = encoder.transform(df_test[categorical_features])


# splitting the features and target
# %%
y_train = df_train[target]
y_val = df_val[target]
y_test = df_test[target]

# Separate numerical features
numeric_features = ["year", "displ", "cylinders"]

X_train_num = df_train[numeric_features].to_numpy()
X_val_num = df_val[numeric_features].to_numpy()
X_test_num = df_test[numeric_features].to_numpy()

# Combine numerical and encoded categorical features
X_train = np.hstack([X_train_num, X_train_cat])
X_val = np.hstack([X_val_num, X_val_cat])
X_test = np.hstack([X_test_num, X_test_cat])
#%%
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# scaling the data using standardscaler
# %%
# Initialize feature scaler
x_scaler = StandardScaler()

# Fit on training numerical features only
X_train[:, :3] = x_scaler.fit_transform(X_train[:, :3])

# Transform validation and test numerical features
X_val[:, :3] = x_scaler.transform(X_val[:, :3])
X_test[:, :3] = x_scaler.transform(X_test[:, :3])

#%%
# Initialize target scaler
y_scaler = StandardScaler()

# Fit on training target only
y_train = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

# Transform validation and test targets
y_val = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
).ravel()

y_test = y_scaler.transform(
    y_test.to_numpy().reshape(-1, 1)
).ravel()

#%%
print("Training target:")
print("Mean:", np.mean(y_train))
print("Standard deviation:", np.std(y_train))

print("\nValidation target:")
print("Mean:", np.mean(y_val))
print("Standard deviation:", np.std(y_val))

print("\nTest target:")
print("Mean:", np.mean(y_test))
print("Standard deviation:", np.std(y_test))

#%%
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# %%
