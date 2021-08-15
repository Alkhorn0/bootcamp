import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 읽어오기
df = pd.read_csv('./train_mdf.csv')

# 데이터 전처리
# 목표 변수 편집 (SalePrice)
bin_dividers = np.array([0, 130000, 300000, 755000])
bin_names = [0, 1, 2]
df['SalePrice_bin'] = pd.cut(x=df['SalePrice'], bins=bin_dividers, labels=bin_names)

# 모델 학습시 활용 dataset 재구축
df = df[['OverallQual','GrLivArea','GarageCars','ExterQual','BsmtQual','KitchenQual','YearBuilt','GarageArea','FullBath','GarageYrBlt','GarageFinish','TotalBsmtSF','SalePrice_bin']]

# label encoding
df['ExterQual'].replace({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}, inplace=True)
df['BsmtQual'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}, inplace=True)
df['KitchenQual'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}, inplace=True)
df['GarageFinish'].replace({'Fin':3, 'RFn':2, 'Unf':1}, inplace=True)

# 결측값이 있는 경우 결측값 처리
df['BsmtQual'].fillna(0, inplace=True)
df['GarageYrBlt'].fillna(0, inplace=True)
df['GarageFinish'].fillna(0, inplace=True)

# 변수 설정
X = df.loc[:, df.columns != 'SalePrice_bin'].values
y = df['SalePrice_bin']

# X를 정규화
from sklearn import metrics, preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data와 test data로 구분(7:3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Random Forest 모델 사용
from sklearn.ensemble import RandomForestClassifier

# max_depth 결정
best_accuracy = 0
best_d = 0
accuracy_list = []
for d in range(1, 5):
    rfc = RandomForestClassifier(max_depth=d, random_state=0)
    rfc.fit(X_train, y_train)
    y_hat = rfc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_hat)
    accuracy_list.append(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_d = d      # best_d = 4 하지만, 검증 결과 d = 3으로 사용

# 모형 학습
rfc_3 = RandomForestClassifier(max_depth=3, random_state=0)
rfc_3.fit(X_train, y_train)

y_hat_rfc_3 = rfc_3.predict(X_test)

# accuracy 값 출력
print(metrics.accuracy_score(y_test, y_hat_rfc_3))

# Confusion Matrix 활용 모형 평가
rfc3_matrix = metrics.confusion_matrix(y_test, y_hat_rfc_3)
print(rfc3_matrix)

# 모형 성능 평가 - 평가 지표 계산
rfc3_report = metrics.classification_report(y_test, y_hat_rfc_3)
print(rfc3_report)

# 검증 : learning curve
import scikitplot as skplt
skplt.estimators.plot_learning_curve(rfc_3, X_train, y_train)
plt.show()