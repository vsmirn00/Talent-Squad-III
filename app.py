#!/usr/bin/env python
# coding: utf-8

# In[546]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import neighbors,linear_model
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.naive_bayes import ComplementNB
from imblearn.over_sampling import SMOTE
from sklearn import svm

import warnings
warnings.filterwarnings('ignore')
plt.style.use("ggplot")


# In[320]:


train = pd.read_csv("train.csv")
train.tail()


# In[321]:


test = pd.read_csv("test.csv")
test.head()


# In[322]:


train = train.rename(columns={"Unnamed: 0": "id"})
test = test.rename(columns={"Unnamed: 0": "id"})


# In[323]:


for i in train.columns[1:]:
    print(train[i].unique())


# In[324]:


n_duplicates = train.drop(labels=["id"], axis=1).duplicated().sum()
print(f"We seem to have {n_duplicates} duplicates in our database.")


# In[325]:


# Handling duplicates based on id
columns_to_consider = train.drop(labels=["id"], axis=1).columns
train = train.drop_duplicates(subset=columns_to_consider)
train.shape


# In[326]:


# For each numerical feature compute number of unique entries
unique_values = train.select_dtypes(include="number").nunique().sort_values()
unique_values.plot.bar(logy=True, figsize=(15, 4), title="Unique values per feature");


# In[327]:


pd.DataFrame({"Null values": np.sum(train.isnull()), "Null percentages":              (np.sum(train.isnull())/len(train)) * 100})


# In[328]:


train.describe().T


# In[329]:


train.info()


# In[330]:


test.describe().T


# In[331]:


test.info()


# In[332]:


train_copy = train.copy()


# In[333]:


train_copy["gender"] = train_copy["gender"].replace("female", 0)
train_copy["gender"] = train_copy["gender"].replace("male", 1)
train_copy["lunch"] = train_copy["lunch"].replace("free/reduced", 0)
train_copy["lunch"] = train_copy["lunch"].replace("standard", 1)
train_copy["test preparation course"] = train_copy["test preparation course"].replace("none", 0)
train_copy["test preparation course"] = train_copy["test preparation course"].replace("completed", 1)
train_copy.head()


# ##### Exploratory Data Analysis

# We have "gender", "lunch" and "test preparation course" as binary variables. 

# In[334]:


def var_boxplot(var):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    train.boxplot(column=["math score"], by=var, ax=ax1)
    train.boxplot(column=["reading score"], by=var, ax=ax2)
    train.boxplot(column=["writing score"], by=var, ax=ax3)


# In[335]:


var_boxplot("gender")


# In[336]:


var_boxplot("lunch")


# In[337]:


var_boxplot("test preparation course")


# In[338]:


train["lunch"].unique()


# In[339]:


def correlation_mat(x):
    methods = ["pearson", "spearman", "kendall"]
    palette = ["magma", "viridis", "cubehelix"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    for i in range(3):
        sns.heatmap(x.corr(method=methods[i]), ax=axes[i], vmax=.8, square=True, annot=True, linewidths=.5, cmap=palette[i])
        axes[i].set_title(methods[i] + " correlation")


# In[340]:


correlation_mat(train_copy)


# In[341]:


train_corr = train_copy.corr(method="pearson")
labels = np.where(np.abs(train_corr)>0.75, "S",
                  np.where(np.abs(train_corr)>0.5, "M",
                           np.where(np.abs(train_corr)>0.25, "W", "")))
plt.figure(figsize=(15, 15))
sns.heatmap(train_corr, mask=np.eye(len(train_corr)), square=True,
            center=0, annot=labels, fmt='', linewidths=.5,
            cmap="vlag", cbar_kws={"shrink": 0.8});


# In[342]:


lower_triangle_mask = np.tril(np.ones(train_corr.shape), k=-1).astype("bool")
df_corr_stacked = train_corr.where(lower_triangle_mask).stack().sort_values()
display(df_corr_stacked)


# In[343]:


def var_boxplot_2(var):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("parental level of education")
    for i in range(2):
        for j in range(3):
            sns.boxplot(ax=axes[i, j], data=train, x="parental level of education", y=var)


# In[344]:


var_boxplot_2("math score")


# In[345]:


var_boxplot_2("reading score")


# In[346]:


var_boxplot_2("writing score")


# In[347]:


train.head()


# In[348]:


c = 0
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
for i in train.columns[-3:]:
    sns.distplot(train[i], ax=ax[c])
    c += 1


# In[349]:


c = 0
fig, ax = plt.subplots(3, 1, figsize=(12, 6))
for i in train.columns[-3:]:
    sns.boxplot(train[i], ax=ax[c])
    c += 1


# In[350]:


binary_var = ["gender", "lunch", "test preparation course"]


# In[351]:


def binary_comparison(df=train):
    c = 0
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))
    for i in binary_var:
        sns.countplot(df[i], ax=ax[c])
        c += 1


# binary_comparison()

# In[352]:


train["parental level of education"].value_counts().plot.bar()


# In[353]:


train["mean_score"] = round(train[["math score", "reading score", "writing score"]].mean(axis=1), 2)
train.head()


# In[354]:


print(f"Skewness of ", train["mean_score"].skew())
print(f"Kurtosis of ", train["mean_score"].kurt())
print(f"Mean of ", train["mean_score"].mean())
print(f"Median of ", train["mean_score"].median())


# In[355]:


pd.DataFrame(train.groupby(["parental level of education"])["test preparation course"].value_counts())


# In[356]:


round(train.groupby(["gender","parental level of education"])[["math score", "reading score", "writing score", "mean_score" ]].mean(), 2)


# In[357]:


pd.DataFrame(train.groupby("gender").size())


# In[358]:


round(train.groupby("gender")["math score"].agg(
    mean="mean",
    median="median",
    standard_deviation="std"
), 2)


# In[359]:


# Do not use groupby with categorical variables, use pd.cut instead

groups = pd.cut(train["math score"], bins=3)
train.groupby(groups)["math score"].agg(count="count")


# In[360]:


train_arr = train_copy.values
train_arr = np.asarray(train_arr)

# Finding normalised array of X_Train
x_std=StandardScaler().fit_transform(train_arr)
sns.distplot(x_std)


# In[361]:


pca = PCA().fit(x_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0, 7, 1)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")


# 5 Components explain almost 100% of the variance.

# In[362]:


pca = PCA(n_components=5)
train_arr=pca.fit_transform(x_std)

sns.set(style="darkgrid")
f, ax = plt.subplots(figsize=(8, 8))

ax = sns.kdeplot(train_arr[:,0], train_arr[:,1], cmap="Greens",
                shade=True, shade_lowest=False)
ax = sns.kdeplot(train_arr[:,1], train_arr[:,2], cmap="Reds",
                shade=True, shade_lowest=False)
ax = sns.kdeplot(train_arr[:,2], train_arr[:,3], cmap="Blues",
                shade=True, shade_lowest=False)
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
green = sns.color_palette("Greens")[-2]
ax.text(0.5, 0.5, "2nd and 3rd projection", size=12, color=blue)
ax.text(-4, 0.0, "1st and 3rd Projection", size=12, color=red)
ax.text(2, 0, "1st and 2nd Projection", size=12, color=green)
plt.xlim(-6,5)
plt.ylim(-2,2)


# ##### Algorithm

# In[363]:


train = train.drop("id", axis=1)
test = test.drop("id", axis=1)


# In[364]:


train_reg = train.copy()
train_reg.head()


# In[365]:


train = train.drop("mean_score", axis=1)


# In[366]:


train.head()


# ##### OHE

# In[367]:


encoder = OneHotEncoder()


# In[368]:


encoder_df = pd.DataFrame(encoder.fit_transform(train[["gender", "lunch", "test preparation course"]]).toarray())
encoder_df.columns = ["female", "male", "free/reduced", "standard", "completed", "none"]
train_encoded = train.join(encoder_df)
train_encoded.drop(["gender", "lunch", "test preparation course"], axis=1, inplace=True)
train_encoded.head()


# In[369]:


encoder_df = pd.DataFrame(encoder.fit_transform(test[["gender", "lunch", "test preparation course"]]).toarray())
encoder_df.columns = ["female", "male", "free/reduced", "standard", "completed", "none"]
test_encoded = test.join(encoder_df)
test_encoded.drop(["gender", "lunch", "test preparation course"], axis=1, inplace=True)
test_encoded.head()


# ##### VIF

# <!-- ##### Vif formula for OHE
# def calculate_vif(data=train_encoded):
#     vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
#     x_var_names = data.columns
#     for i in range(0, x_var_names.shape[0]):
#         y = data[x_var_names[i]]
#         x = data[x_var_names.drop([x_var_names[i]])]
#         r_squared = sm.OLS(y,x).fit().rsquared
#         vif = round(1/(1-r_squared),2)
#         vif_df.loc[i] = [x_var_names[i], vif]
#     return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)
# 
# # X=df.drop(['Salary'],axis=1)
# calculate_vif() -->

# In[370]:


pd.DataFrame({"Null values": np.sum(train_encoded.isnull()), "Null percentages":              (np.sum(train_encoded.isnull())/len(train)) * 100})


# <!-- imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(train_encoded)
# train_encoded = imputer.transform(train_encoded) -->

# In[371]:


# Don't drop just take what's not null
train_encoded = train_encoded[train_encoded["female"].notna()]


# In[372]:


train_encoded.head()


# In[373]:


train_encoded_2 = train_encoded[["parental level of education", "math score", "reading score",
                               "writing score", "female", "free/reduced", "completed"]]


# y_3 = train["parental level of education"]
# x_3 = train.drop("parental level of education", axis=1)

# In[374]:


train_encoded_2.head()


# In[416]:


y = train_encoded_2["parental level of education"]
x = train_encoded_2.drop("parental level of education", axis=1)


# y = train_encoded["parental level of education"]
# x = train_encoded.drop(["parental level of education"], axis=1)

# In[376]:


# Count the disbalance
train.groupby("parental level of education")["parental level of education"].agg("count").values


# In[377]:


oversample = SMOTE()
x, y = oversample.fit_resample(x, y)


# In[378]:


y


# In[379]:


pd.DataFrame(y)["parental level of education"].value_counts().plot.bar()


# In[380]:


x["female"] = x["female"].agg(lambda x: int(x))
x["free/reduced"] = x["free/reduced"].agg(lambda x: int(x))
x["completed"] = x["completed"].agg(lambda x: int(x))


# In[381]:


x.head()


# In[382]:


pd.DataFrame(y).groupby("parental level of education")["parental level of education"].agg("count").values


# In[384]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)


# In[385]:


x_train.head()


# In[389]:


pd.DataFrame(y_val)["parental level of education"].value_counts().plot.bar()


# ### SVM

# In[390]:


clf = SVC(decision_function_shape="ovo", kernel = 'poly', degree= 3, class_weight="balanced")
clf.fit(x_train, y_train)
y_pred=clf.predict(x_val)
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_val, y_pred)))


# In[391]:


clf = SVC(kernel = 'rbf', class_weight="balanced").fit(x_train, y_train)
clf_pred = clf.predict(x_val)
clf_pred


# In[392]:


clf_accuracy = accuracy_score(y_val, clf_pred)
clf_f1 = f1_score(y_val, clf_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (clf_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (clf_f1*100))


# oversample = SMOTE()
# x, y = oversample.fit_resample(x, y)

# In[393]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)


# In[394]:


rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1, class_weight="balanced").fit(x_train_scaled, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, class_weight="balanced").fit(x_train_scaled, y_train)


# In[395]:


poly_pred = poly.predict(x_val_scaled)
rbf_pred = rbf.predict(x_val_scaled)


# In[396]:


poly_accuracy = accuracy_score(y_val, poly_pred)
poly_f1 = f1_score(y_val, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))


# In[397]:


params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000],
               'class_weight': ['balanced']},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
                    'class_weight': ['balanced']}]


# In[398]:


pd.DataFrame({"Null values": np.sum(x_train.isnull()), "Null percentages":              (np.sum(x_train.isnull())/len(train)) * 100})


# In[399]:


get_ipython().run_cell_magic('time', '', 'svm_model = GridSearchCV(SVC(), params_grid, cv=5)\nsvm_model.fit(x_train_scaled, y_train)')


# In[400]:


# View the accuracy score
print('Best score for training data:', svm_model.best_score_,"\n") 

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n") 
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")


# In[ ]:


final_model = svm_model.best_estimator_
y_pred = final_model.predict(x_val_scaled)
# y_pred = y_pred.reshape(1, -1)
y_pred_label = list(encoder.inverse_transform(y_pred))


# In[ ]:


print(confusion_matrix(y_val, clf_pred))
print("\n")
print(classification_report(y_val, clf_pred))

print("Training set score for SVM: %f" % final_model.score(x_train, y_train))
print("Testing  set score for SVM: %f" % final_model.score(x_val, y_val))

svm_model.score


# In[ ]:


rmse_ridge = np.round(np.sqrt(np.square(np.subtract(y_val, y_pred)).mean()), 3)
print(f"The final error for the ridge_model is: {rmse_ridge}")


# In[ ]:


train_error=[rmse_ridge]
col={"Train Error": train_error}
models=['Ridge Regression']
df_rmse=pd.DataFrame(data=col,index=models)
df_rmse


# In[ ]:


df_rmse.plot(kind='bar', logy=True, title= "RMSE per model")


# ##### XGB

# In[ ]:


model = XGBClassifier()


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


y_pred = model.predict(x_val)


# In[ ]:


print(confusion_matrix(y_val, y_pred))
print("\n")
print(classification_report(y_val, y_pred))

print("Training set score for xgboost: %f" % model.score(x_train, y_train))
print("Testing  set score for xgboost: %f" % model.score(x_val, y_val))


# In[ ]:


nb = ComplementNB()
nb.fit(x_train, y_train)


# In[273]:


pred_nb = nb.predict(x_val)


# In[274]:


print(confusion_matrix(y_val, pred_nb))
print("\n")
print(classification_report(y_val, pred_nb))


# In[275]:


rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_val)


# In[276]:


print(confusion_matrix(y_val, rfc_pred))
print("\n")
print(classification_report(y_val, rfc_pred))


# In[277]:


features = train[["math score", "reading score", "writing score"]]
y_ = train["parental level of education"]


# ##### NN

# In[278]:


x_train.head()


# In[300]:


epochs = 100
batch_size = 16


# In[301]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='tanh', input_dim=6))
model.add(tf.keras.layers.Dense(256, activation='tanh'))
model.add(tf.keras.layers.Dense(512, activation='tanh'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[302]:


model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          steps_per_epoch=x_train.shape[0]//batch_size,)


# In[283]:


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_val)


# In[284]:


print(confusion_matrix(y_val, knn_pred))
print("\n")
print(classification_report(y_val, knn_pred))


# In[285]:


train.shape


# In[286]:


y = train_encoded["parental level of education"]
x = train_encoded.drop(["parental level of education"], axis=1)


# In[321]:


kfold = model_selection.KFold(n_splits=10) 
# create the sub models
estimators = [] 
model1 = LogisticRegression() 
estimators.append(('logistic', model1)) 
model2 = DecisionTreeClassifier() 
estimators.append(('cart', model2)) 
model3 = SVC()
estimators.append(('svm', model3)) 
# create the ensemble model 
ensemble=VotingClassifier(estimators) 
results = model_selection.cross_val_score(ensemble, x, y, cv=kfold)
print(results.mean())


# #####  Stratified Kfold

# In[556]:


from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
# from imblearn.pipeline import make_pipeline


# In[557]:


train_encoded_2["female"] = train_encoded_2["female"].agg(lambda x: int(x))
train_encoded_2["free/reduced"] = train_encoded_2["free/reduced"].agg(lambda x: int(x))
train_encoded_2["completed"] = train_encoded_2["completed"].agg(lambda x: int(x))


# In[562]:


y = train_encoded_2["parental level of education"]
x = train_encoded_2.drop("parental level of education", axis=1)
y = LabelEncoder().fit_transform(y)


# <!-- y = LabelEncoder().fit_transform(y)
# # transform the dataset
# strategy = {0:100, 1:100, 2:200, 3:200, 4:200, 5:200}
# oversample = SMOTE(sampling_strategy="all")
# x, y = oversample.fit_resample(x, y) -->

# In[573]:


from sklearn.utils import class_weight
classes_weights = list(class_weight.compute_class_weight('balanced',
                                             classes=np.unique(y),
                                             y=y))

weights = np.ones(y.shape[0], dtype = 'float')
for i, val in enumerate(y):
    weights[i] = classes_weights[val-1]


# In[574]:


steps = [('scaler', RobustScaler()), ('over', SMOTE()), 
         ('model', XGBClassifier(sample_weight=weights))]
pipeline = Pipeline(steps=steps)
# pipeline = make_pipeline[('over', SMOTE()), ('model', LogisticRegression())]


# In[560]:


train_encoded.head()


# In[570]:


pd.DataFrame(y).value_counts().plot.bar()


# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# In[571]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
scores = cross_val_score(pipeline, x, y, scoring="f1_macro", cv=cv)
print("Model F1-Score",  " mean=", scores.mean() , "stddev=", scores.std())
print(scores)


# In[452]:


train.head()

