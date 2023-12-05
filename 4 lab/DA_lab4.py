import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("3cancer.csv")

#2
def inform(df):
    print(df.head(20))
    df.info()
    print(df.columns)
    print(df.isna().sum())

print("Количесвто дубликатов: " + str(df.duplicated().sum()))
df = df.drop_duplicates().reset_index()
print("Количесвто дубликатов: " + str(df.duplicated().sum()))
def uniqe_check(df):
    print("Uniqe id")
    print(df['id'].unique())
    print("Uniqe clump_thickness")
    print(df['clump_thickness'].unique())
    print("Uniqe size_uniformity")
    print(df['size_uniformity'].unique())
    print("Uniqe shape_uniformity")
    print(df['shape_uniformity'].unique())
    print("Uniqe marginal_adhesion")
    print(df['marginal_adhesion'].unique())
    print("Uniqe epithelial_size")
    print(df['epithelial_size'].unique())
    print("Uniqe bare_nucleoli")
    print(df['bare_nucleoli'].unique())
    print("Uniqe bland_chromatin")
    print(df['bland_chromatin'].unique())
    print("Uniqe normal_nucleoli")
    print(df['normal_nucleoli'].unique())
    print("Uniqe mitoses")
    print(df['mitoses'].unique())
    print("Uniqe class ")
    print(df['class'].unique())

df['bare_nucleoli'] = df['bare_nucleoli'].replace('?', np.NaN)
df = df.dropna(subset=['bare_nucleoli']).reset_index(drop=True)
# print("Uniqe bare_nucleoli")
# print(df['bare_nucleoli'].unique())

df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'])
df.info()

# sb.kdeplot(df['clump_thickness'],vertical=True,color='blue',shade=False, label='clump_thickness')
# sb.kdeplot(df['size_uniformity'],vertical=True,color='red',shade=False)
# sb.kdeplot(df['shape_uniformity'],vertical=True,color='green',shade=False)
# sb.kdeplot(df['marginal_adhesion'],vertical=True,color='yellow',shade=False)
# sb.kdeplot(df['epithelial_size'],vertical=True,color='cyan',shade=False)
# sb.kdeplot(df['bare_nucleoli'],vertical=True,color='orange',shade=False)
# sb.kdeplot(df['bland_chromatin'],vertical=True,color='violet',shade=False)
# sb.kdeplot(df['normal_nucleoli'],vertical=True,color='pink',shade=False)
# sb.kdeplot(df['mitoses'],vertical=True,color='black',shade=False)
# sb.kdeplot(df['class'],vertical=True,color='olive',shade=False)
# plt.show()
def correlation (df):
    corr = df.corr(method = 'pearson')
    sb.heatmap(corr, cmap="Blues", annot=True)
    plt.show()

df = df.drop('id', axis = 1)
# df = df.drop('bare_nucleoli', axis = 1)
# df = df.drop('shape_uniformity', axis = 1)

def fit_train(df):
    train, test = train_test_split(df, test_size=0.3)

    train.info()
    test.info()
    model = DecisionTreeClassifier()

    train_y = train["class"]
    print(train_y)
    train_X = train.drop('class', axis = 1)

    train_X.info()

    test_y = test["class"]
    test_X = test.drop('class', axis = 1)
    test.info()

    model.fit(train_X,train_y)
    predictions = model.predict(test_X)
    print("predictions:")
    print(predictions)
    
    rf_probs = model.predict_proba(test_X)[::, 1]
    
    return test_y, predictions, rf_probs

def ROC_curve (test_y,roc_auc,rf_probs):
    print (test_y)
    test_y = test_y.replace(to_replace = 2, value = 0)
    test_y = test_y.replace(to_replace = 4, value = 1)
    print (test_y)
    fpr, tpr, _ = metrics.roc_curve (test_y, rf_probs)

    #create ROC curve
    plt.plot (fpr,tpr,label=" AUC= "+str(roc_auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show() 

def model_check (test_y,predictions,rf_probs):
    # Матрица ошибок
    cm = confusion_matrix(test_y,predictions)
    tn, fp, fn, tp = cm.ravel() # "выпрямляем" матрицу,чтобы вытащить нужные значения
    print(tn, fp, fn, tp)

    #Доля правильных ответов
    acc = accuracy_score(test_y,predictions)
    print("Accuracy score = " + str(acc))

    #Точность (precision) и полнота (recall)
    precision = precision_score (test_y,predictions,pos_label=2)
    recall = recall_score (test_y,predictions,pos_label=2)
    print("precision_score = " + str(precision))
    print("recall_score = " + str(recall))

    #F1-мера
    f1= f1_score(test_y,predictions, pos_label=2)
    print("F1 score = " + str(f1))

    #ROC кривая
    roc_auc = roc_auc_score(test_y,rf_probs)
    print("ROC = " + str(roc_auc))
    ROC_curve (test_y,roc_auc,rf_probs)

def forest(df):
    train, test = train_test_split(df, test_size=0.3)
    train_y = train["class"]
    train_X = train.drop('class', axis = 1)
    test_y = test["class"]
    test_X = test.drop('class', axis = 1)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    # Вероятности для каждого класса
    rf_probs = clf.predict_proba(test_X)[::, 1]
    return test_y, y_pred, rf_probs

correlation (df)
test_y, y_pred, rf_probs = fit_train(df)
model_check (test_y,y_pred,rf_probs)






















# #3
# scaler = StandardScaler() # создаём объект класса scaler
# scaler.fit(df) # обучаем стандартизатор
# X_sc = scaler.transform(df) # преобразуем набор данных

# n = 675
# plt.plot([1]*n, df['age'], 'bo', label='age')
# plt.plot([2]*n, df['anaemia'], 'co', label='anaemia')
# plt.plot([3]*n, df['creatinine_phosphokinase'], 'mo', label='creatinine_phosphokinase')
# plt.plot([4]*n, df['diabetes'], 'co', label='diabetes')
# plt.plot([5]*n, df['ejection_fraction'], 'go', label='ejection_fraction')
# plt.plot([6]*n, df['high_blood_pressure'], 'co', label='high_blood_pressure')
# plt.plot([7]*n, df['platelets'], 'ro', label='platelets')
# plt.plot([8]*n, df['serum_creatinine'], 'yo', label='serum_creatinine')
# plt.plot([9]*n, df['serum_sodium'], 'ko', label='serum_sodium')
# plt.plot([10]*n, df['sex'], 'co', label='sex')
# plt.plot([11]*n, df['smoking'], 'wo', label='smoking')
# plt.plot([12]*n, df['time'], 'bo', label='time')
# plt.plot([13]*n, df['DEATH_EVENT'], 'co', label='DEATH_EVENT')
# plt.legend(loc=0)
# plt.show()

# #обязательная стандартизация данных перед работойс алгоритмами
# linked = linkage(X_sc, method = 'ward')
# plt.figure(figsize=(15, 10))
# dendrogram(linked, orientation='top')
# plt.title('Hierarchial clustering for heart ')
# plt.show() 

# # обязательная стандартизация данных перед работой с алгоритмами
# km = KMeans(n_clusters = 3, random_state=0) # задаём число кластеров, равное 5, и фиксируем значение random_state для воспроизводимости результата
# labels = km.fit_predict(X_sc) # применяем алгоритм к данным и формируем вектор кластеров
# print(silhouette_score(X_sc, labels))

# inertia = []
# for k in range(1, 8):
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(X_sc)
#     inertia.append(np.sqrt(kmeans.inertia_))

# plt.plot(range(1, 8), inertia, marker='s')
# plt.xlabel('$k$')
# plt.ylabel('$J(C_k)$')
# plt.show()
# inertia = []
# for k in range(2, 8):
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     labels = kmeans.fit_predict(X_sc)
#     inertia.append(silhouette_score(X_sc, labels))

# plt.plot(range(2, 8), inertia, marker='s')
# plt.xlabel('$k$')
# plt.ylabel('$Silhouette$')
# plt.show()

