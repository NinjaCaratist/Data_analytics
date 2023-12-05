import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("6games.csv")

#2
print(df.head(20))

df.info()

print(df.columns)



print(df.isna())
print(df.isna().sum())
df = df.dropna(subset=['Name'])
df['Genre'] = df['Genre'].fillna('none')
df['Year_of_Release'] = df['Year_of_Release'].fillna(1950)
df['Rating'] = df['Rating'].fillna('none')
tmp_df = df
df['Critic_Score'] = df['Critic_Score'].fillna(0)
df['User_Score'] = df['User_Score'].fillna(0)

print(df.isna().sum())

print("Количесвто дубликатов: " + str(df.duplicated().sum()))
df.info()

print(df['Name'].nunique(), "\n")
df = df.drop_duplicates(subset=['Name', 'Platform', 'Year_of_Release']).reset_index()
df.info()
print( "\n")
print(df['Year_of_Release'].unique())
print(df['Platform'].unique())
print(df['Genre'].unique())
print(df['Rating'].unique())

df['Year_of_Release'] = pd.to_datetime(df['Year_of_Release'], format='%Y')
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
df['User_Score'] = df['User_Score'].fillna(0)
print(df['Year_of_Release'].unique())
print(df['User_Score'].unique())
df.info()

data_pivot = df.pivot_table(index=['Genre'], columns='Platform', values='NA_sales', aggfunc='sum')
print(data_pivot)

#3

print(df.describe(include='all',datetime_is_numeric=True))
df.hist()
plt.show()

df.plot(x='Year_of_Release', y='NA_sales', kind='scatter')
plt.show()

pd.plotting.scatter_matrix(df)
plt.show()

#4,5
tmp_df = tmp_df.dropna(subset=['Critic_Score'])
tmp_df['Year_of_Release'] = pd.to_datetime(tmp_df['Year_of_Release'], format='%Y')
tmp_df['User_Score'] = pd.to_numeric(tmp_df['User_Score'], errors='coerce')
tmp_df = tmp_df.dropna(subset=['User_Score'])

corr = tmp_df.corr(method = 'pearson')
print(corr)
sb.heatmap(corr, cmap="Blues", annot=True)
plt.show()

cov = tmp_df.cov()
print(cov)
sb.heatmap(cov, cmap="Blues", annot=True)
plt.show()

