# 선형 회귀 (Linear Regression) 구현

이 저장소는 Python과 NumPy를 사용하여 선형 회귀(Linear Regression)를 처음부터 구현한 코드입니다. 본 구현에서는 단순 선형 회귀와 다중 선형 회귀를 **경사 하강법(Gradient Descent)** 을 이용하여 학습합니다.

## 주요 기능
- **단순 선형 회귀(Simple Linear Regression)** 지원 (하나의 특성)
- **다중 선형 회귀(Multiple Linear Regression)** 지원 (여러 개의 특성)
- **경사 하강법(Gradient Descent)** 을 이용한 최적화
- **평균 제곱 오차(MSE, Mean Squared Error)** 를 비용 함수로 사용
- **Matplotlib을 이용한 데이터 시각화**
- **Scikit-Learn의 Linear Regression과 비교하여 성능 검증**

## 구현 코드
```
poly_reg_df = pd.read_csv("C:\\Users\\aqtg6\\Downloads\\hw2_poly_reg.csv")

import pandas as pd

poly_reg_df.columns = ["X", "y"]

X = poly_reg_df[["X"]].values  # (N, 1) 형태 유지
y = poly_reg_df["y"].values    # (N, ) 형태

model = LinearRegression()
model.fit(X, y)

print("직접 구현한 코드로 학습된 가중치:", model.theta_)

from sklearn.linear_model import LinearRegression

sklearn_model = LinearRegression()
sklearn_model.fit(X, y)
np.hstack((sklearn_model.intercept_, sklearn_model.coef_))
```
결과
직접 구현한 코드로 학습된 가중치: [1.11694149 0.78017273]
sklearn을 활용한 가중치 : array([1.28666847, 1.5981036 ])

## 결과 비교
내가 직접 만든 LinearRegression과 sklearn의 LinearRegression에서 bias는 비슷한데 가중치는 꽤 차이가 났습니다.

## 원인 분석
1. 학습률 (alpha) 차이
나의 선형 회귀는 경사 하강법을 사용하지만, sklearn은 정규 방정식(Closed-form solution)을 사용해서 최적 가중치를 한 번에 계산합니다.

2. 반복 횟수 부족
나의 모델은 max_iter가 정해져 있어서 충분히 수렴하지 못 했을 가능성이 있습니다.
sklearn 모델은 정규 방정식을 사용하기 때문에 반복 없이 최적 해를 찾습니다.

## Scikit-Learn과의 비교
커스텀 모델의 성능이 sklearn의 LinearRegression과 비교하여 유사한 결과를 도출하는지 검증하였습니다.

추가적인 실습으로 선형 회귀에 대한 더욱 깊은 학습을 했습니다.
