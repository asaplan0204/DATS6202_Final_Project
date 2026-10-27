#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

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
print("Train datatypes:\n", df_train.dtypes)
print("\nValidation datatypes:\n", df_val.dtypes)
print("\nTest datatypes:\n", df_test.dtypes)
# %%
print(df_train.head())
# %%
categorical_features = ["drive", "trany", "VClass", "fuelType1", "model"]

#one hot encoding
# %%
df_train = pd.get_dummies(df_train, columns=categorical_features, dtype=int)
df_val = pd.get_dummies(df_val, columns=categorical_features, dtype=int)
df_test = pd.get_dummies(df_test, columns=categorical_features, dtype=int)

# %%
df_val = df_val.reindex(columns=df_train.columns, fill_value=0)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
# %%
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)


# splitting the features and target
# %%
# Training split
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

# Validation split
X_val = df_val.drop(columns=[target])
y_val = df_val[target]

# Test split
X_test = df_test.drop(columns=[target])
y_test = df_test[target]
# %%
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# scaling the data using standardscaler
# %%
ss = StandardScaler()
# %%
# Standardize the training data
X_train = ss.fit_transform(X_train)

# Standardize the validation data
X_val = ss.transform(X_val)

# Standardize the test data
X_test = ss.transform(X_test)
# %%
# Standardize the training target
y_train = ss.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).reshape(-1)

# Standardize the validation target
y_val = ss.transform(
    y_val.to_numpy().reshape(-1, 1)
).reshape(-1)

# Standardize the test target
y_test = ss.transform(
    y_test.to_numpy().reshape(-1, 1)
).reshape(-1)
# %%
print("Training target:")
print("Mean:", np.mean(y_train))
print("Standard deviation:", np.std(y_train))

print("\nValidation target:")
print("Mean:", np.mean(y_val))
print("Standard deviation:", np.std(y_val))

print("\nTest target:")
print("Mean:", np.mean(y_test))
print("Standard deviation:", np.std(y_test))
# %%
