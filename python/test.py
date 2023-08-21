dir_path = '/home/alper/Downloads/yenidata/'
file_list = os.listdir(dir_path)

def box_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
    series = series[mask]
    return series


excl_merged = pd.DataFrame()

for file in file_list:
    print(file + ": " + str(pd.read_excel(dir_path + file).shape[0]))
    excl_merged = excl_merged.append(pd.read_excel(dir_path + file), ignore_index=True)
print("done")
df = excl_merged

df = df[['SEKTÖR', 'KAYNAK', 'E-TICARET TECRÜBESI', 'DEMO SKORU', 'AKSIYON ADEDI', 'PBX ADEDI',
         'FORM TIPI', 'ŞEHIR', 'FIRSAT ADEDI', 'SATIŞ ADEDI', 'İPTAL NEDENI', 'OLUMSUZLUK NEDENI', 'DEMO OLUŞTURULMA SAATI', 'SATIŞ TEMSILCISI']]

#eksik datalar
df = df.fillna(method="ffill")
df = df.fillna(method="bfill")

#4 ve üstü sastışlar segmente
df.loc[df['SATIŞ ADEDI'] >= 4, 'SATIŞ ADEDI'] = 4

#drop na
df = df.dropna()


#grafilkler
for col in df.columns:
    plt.title(col, fontsize=30)
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(20.7,15.27)})
    plt.xticks(rotation=90)
    ax = sns.countplot(x=col, data=df)

    plt.figure()


#cross validation ?denedik mi?

X = df.drop(['SATIŞ ADEDI'], axis=1)
y = df['SATIŞ ADEDI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.shape, X_test.shape


filteredColumns = X_train.dtypes[X_train.dtypes == np.object]
listOfColumnNames = list(filteredColumns.index)
print(listOfColumnNames)

encoder = ce.OrdinalEncoder(cols=listOfColumnNames)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


plt.boxplot(df['FIRSAT ADEDI'])
plt.title("Aksyion adedi Aykırı Değerleri", fontsize=30)
plt.yticks(fontsize=33)
plt.xticks(fontsize=22)

fig = plt.figure(figsize =(20, 15))
plt.show()


#outlier hesapları
out = df['AKSIYON ADEDI']

# 1st quartile
q1 = np.quantile(out, 0.25)

# 3rd quartile
q3 = np.quantile(out, 0.75)
med = np.median(out)

# iqr region
iqr = q3-q1

upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print(iqr, upper_bound, lower_bound)


#tahminler
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#feature score
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores)


#rfe
# evaluate RFE for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pipeline = Pipeline(steps=[('s',rfe),('m',model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


feature_scores_rfe = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores_rfe)

#confsion matrix
cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

