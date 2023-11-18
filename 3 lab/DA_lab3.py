import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("4heart2.csv")

#2
print(df.head(20))

df.info()

print(df.columns)


print(df.isna().sum())

print("Количесвто дубликатов: " + str(df.duplicated().sum()))

print("Uniqe age")
print(df['age'].unique())
print("Uniqe anaemia")
print(df['anaemia'].unique())
print("Uniqe creatinine_phosphokinase")
print(df['creatinine_phosphokinase'].unique())
print("Uniqe diabetes")
print(df['diabetes'].unique())
print("Uniqe ejection_fraction")
print(df['ejection_fraction'].unique())
print("Uniqe high_blood_pressure")
print(df['high_blood_pressure'].unique())
print("Uniqe ejection_fraction")
print(df['ejection_fraction'].unique())
print("Uniqe platelets")
print(df['platelets'].unique())
print("Uniqe serum_creatinine")
print(df['serum_creatinine'].unique())
print("Uniqe serum_sodium")
print(df['serum_sodium'].unique())
print("Uniqe sex")
print(df['sex'].unique())
print("Uniqe smoking")
print(df['smoking'].unique())
print("Uniqe time")
print(df['time'].unique())
print("Uniqe DEATH_EVENT")
print(df['DEATH_EVENT'].unique())

df = df.astype({"age":'int'})
df.info()
print(df['age'])

#3
scaler = StandardScaler() # создаём объект класса scaler
scaler.fit(df) # обучаем стандартизатор
X_sc = scaler.transform(df) # преобразуем набор данных

n = 299
plt.plot([1]*n, df['age'], 'bo', label='age')
plt.plot([2]*n, df['anaemia'], 'co', label='anaemia')
plt.plot([3]*n, df['creatinine_phosphokinase'], 'mo', label='creatinine_phosphokinase')
plt.plot([4]*n, df['diabetes'], 'co', label='diabetes')
plt.plot([5]*n, df['ejection_fraction'], 'go', label='ejection_fraction')
plt.plot([6]*n, df['high_blood_pressure'], 'co', label='high_blood_pressure')
plt.plot([7]*n, df['platelets'], 'ro', label='platelets')
plt.plot([8]*n, df['serum_creatinine'], 'yo', label='serum_creatinine')
plt.plot([9]*n, df['serum_sodium'], 'ko', label='serum_sodium')
plt.plot([10]*n, df['sex'], 'co', label='sex')
plt.plot([11]*n, df['smoking'], 'wo', label='smoking')
plt.plot([12]*n, df['time'], 'bo', label='time')
plt.plot([13]*n, df['DEATH_EVENT'], 'co', label='DEATH_EVENT')
plt.legend(loc=0)
plt.show()

#обязательная стандартизация данных перед работойс алгоритмами
linked = linkage(X_sc, method = 'ward')
plt.figure(figsize=(15, 10))
dendrogram(linked, orientation='top')
plt.title('Hierarchial clustering for heart ')
plt.show() 

# обязательная стандартизация данных перед работой с алгоритмами
km = KMeans(n_clusters = 3, random_state=0) # задаём число кластеров, равное 5, и фиксируем значение random_state для воспроизводимости результата
labels = km.fit_predict(X_sc) # применяем алгоритм к данным и формируем вектор кластеров
print(silhouette_score(X_sc, labels))

inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_sc)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 8), inertia, marker='s')
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$')
plt.show()
inertia = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X_sc)
    inertia.append(silhouette_score(X_sc, labels))

plt.plot(range(2, 8), inertia, marker='s')
plt.xlabel('$k$')
plt.ylabel('$Silhouette$')
plt.show()

