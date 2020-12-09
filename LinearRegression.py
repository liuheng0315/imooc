import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('‪E:\\java\\贪心学院\\贪心学院初级\\资料\\机器学习训练营\\机器学习第一次更新\\24-lesson\\data\\Advertising.csv'.strip('\u202a'))
print(data.head())
print(data.info())
print("-------------")
print(data.columns)

x = data['TV'].values.reshape((-1, 1))
y = data['sales'].values.reshape((-1, 1))

reg = LinearRegression()
reg.fit(x, y)

print('a = {:.5}'.format(reg.coef_[0][0]))
print('b={:.5}'.format(reg.intercept_[0]))
print("线性模型为: Y = {:.5}X + {:.5} ".format(reg.coef_[0][0], reg.intercept_[0]))

predictions = reg.predict(x)

plt.figure(figsize=(16, 8))
plt.scatter(data['TV'], data['sales'], c='black')
plt.plot(data['TV'], predictions, c='blue', linewidth=2)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.show()

predictions = reg.predict([[100]])
print(predictions)
