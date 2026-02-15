import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\pranesh.S.S\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\D7FC61322C3B74B6CD41C9C3FA0FB0905AA7C652\transfers\2026-06\azure_dataset_missing_values.csv")
print(df)


print(df[['usage_units', 'cost_usd']].corr())

df['timestamp'] = pd.to_datetime(df['timestamp']) #converts the string data into datetime objects
df = df.sort_values(by='timestamp') # sorting is mandatory for the time-series

#  TIME FEATURES (MANDATORY FOR FORECASTING) 
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['quarter'] = df['timestamp'].dt.quarter

print(df)

df['region'] = df['region'].str.lower().str.replace(" ", "-")# str.lower- removes the case diff
#.replaces- standaridizes the formatting
print(df)

numeric_cols = [
    'usage_units','provisioned_capacity','cost_usd',
    'availability_pct','economic_index','market_demand_index'
]

# Boxplot (shows outliers clearly)
plt.figure(figsize=(14, 6))
df[numeric_cols].boxplot()
plt.title("Before Preprocessing - Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histograms (distribution & skewness)
df[numeric_cols].hist(bins=30, figsize=(14, 10))
plt.suptitle("Before Preprocessing - Distribution", fontsize=16)
plt.tight_layout()
plt.show()

df.info() #gives information about the dataset

print(df.duplicated())# shows dupliccated values by checking each rows

print(df.duplicated().sum())# shows the total no of duplicated values

df = df.drop_duplicates()   # drops the duplicate values

print(df.columns)# shows the column name in a list 

print(df["usage_units"].skew())# shows if the specific column in the dataset is skewed or not " if 0 normal if not skewed distribution

columns = ['usage_units','provisioned_capacity', 'cost_usd', 'availability_pct',
           'economic_index', 'market_demand_index']
# store only the colums with numerical values to check if it is skewed or not


print(df.isnull().sum())# shows how many null values in each column

# important note : " if a specific column's null values is greater than 50% we should drop the column because it will cause errors .
# we cannot fill it without an machine learning model "

# Fills usage first 
df["usage_units"] = df["usage_units"].fillna(df["usage_units"].median())

# Calculate rate only using valid rows 
valid_rows = df[df['cost_usd'].notnull() & df['usage_units'].notnull()]
rate = (valid_rows['cost_usd'] / valid_rows['usage_units']).median()
print("Estimated pricing rate:", rate)

# Now fill cost using calculated rate
df['cost_usd'] = df['cost_usd'].fillna(df['usage_units'] * rate)


df["provisioned_capacity"] = df["provisioned_capacity"].fillna(df["provisioned_capacity"].median())


df["availability_pct"] = df["availability_pct"].fillna(df["availability_pct"].median())

df["economic_index"] = df["economic_index"].fillna(df["economic_index"].median())

df["market_demand_index"] = df["market_demand_index"].fillna(df["market_demand_index"].median())       

print(df.isnull().sum())

columns = [
    'usage_units','provisioned_capacity','cost_usd',
    'availability_pct','economic_index','market_demand_index'
]

for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] >= upper_bound) | (df[col]<= lower_bound)]
    print(len(outliers))


for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] >= upper_bound) | (df[col]<= lower_bound)]
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)


for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
    print(len(outliers))


#  prints the rows before lag  
print("Rows before lag:", len(df))



#  LAG FEATURES (CRITICAL FOR TIME SERIES) 

df['lag_1'] = df['usage_units'].shift(1)
df['lag_7'] = df['usage_units'].shift(7)
df['rolling_mean_7'] = df['usage_units'].rolling(window=7).mean()

# Drop rows created by lag

df = df.dropna(subset=['lag_1','lag_7','rolling_mean_7'])

# prints the rows after lag 

print("Rows after lag:", len(df))


# BUSINESS LOGIC VALIDATION

print("\n--- BUSINESS RULE VALIDATION ---")

# 1. Negative usage check
print("Negative usage values:", (df['usage_units'] < 0).sum())

# 2. Availability > 100%
print("Availability above 100%:", (df['availability_pct'] > 100).sum())

# 3. Usage exceeding capacity
print("Usage > Provisioned Capacity:",
      (df['usage_units'] > df['provisioned_capacity']).sum())

# Inspect a few rows
print("\nSample Over-Capacity Rows:")
print(df[df['usage_units'] > df['provisioned_capacity']][
    ['timestamp','region','service_type','usage_units','provisioned_capacity']
].head())

# Create over-capacity flag feature
df['over_capacity_flag'] = (df['usage_units'] > df['provisioned_capacity']).astype(int)

# 4. Economic index sanity check
print("Economic index outside realistic range:",
      ((df['economic_index'] < 80) | (df['economic_index'] > 120)).sum())



#  ENCODING CATEGORICAL VARIABLES 

df = pd.get_dummies(df, columns=['region', 'service_type'], drop_first=True)


#  Core Continuous Variables 
core_vars = [
    'usage_units',
    'provisioned_capacity',
    'cost_usd',
    'availability_pct',
    'economic_index',
    'market_demand_index',
    'lag_1',
    'lag_7',
    'rolling_mean_7'
]

plt.figure(figsize=(14,6))
df[core_vars].boxplot()
plt.title("After Preprocessing - Core Variables Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram
df[core_vars].hist(bins=30, figsize=(16,12))
plt.suptitle("After Preprocessing - Core Variables Distribution", fontsize=16)
plt.tight_layout()
plt.show()

#  TIME FEATURE DISTRIBUTION 
import seaborn as sns


plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.countplot(x='month', data=df)

plt.subplot(2,2,2)
sns.countplot(x='day_of_week', data=df)

plt.subplot(2,2,3)
sns.countplot(x='quarter', data=df)

plt.subplot(2,2,4)
sns.countplot(x='year', data=df)

plt.tight_layout()
plt.show()

#  FINAL SHAPE OF THE DATASET 
print("\nFinal dataset shape:", df.shape)





   

