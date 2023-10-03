import pandas as pd

df = pd.read_csv("E:/1auto.csv")

#3
print(df.head(20))

#5
df.info()

#6
print(df.columns)
df = df.rename(columns={'SellingPrice': 'celling_price', 'kmdriven':
'km_driven', 'seller_Type': 'seller_type'})

#7
print(df.isna())
print(df.isna().sum())
df = df.dropna(subset=['celling_price', 'km_driven'])
print(df.isna().sum())
print(df.isna())

#8
print(df.duplicated().sum())
df = df.drop_duplicates().reset_index()
print(df.duplicated().sum())
df.info()

print(df["year"].unique(), "\n")
df['year'] = df['year'].replace("'2007'", "2007")
print(df["year"].unique())

#9
df['year'] = pd.to_datetime(df['year'], format='%Y')
df.info()
print(df["year"].unique())
print(df["fuel"].unique())

#10
data_pivot = df.pivot_table(index=['year'], columns='transmission', values='celling_price', aggfunc='sum')
print(data_pivot)
