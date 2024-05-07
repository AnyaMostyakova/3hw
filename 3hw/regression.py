import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
bike_data = pd.read_csv('bikes_rent.csv')

# Пункт 2: Простая линейная регрессия
feature = bike_data[['weathersit']].values
target = bike_data['cnt'].values

linear_model = LinearRegression()
linear_model.fit(feature, target)

plt.scatter(feature, target, color='blue')
plt.plot(feature, linear_model.predict(feature), color='red')
plt.title('Прогноз спроса на основе благоприятности погоды')
plt.xlabel('Благоприятность погоды')
plt.ylabel('Спрос')
plt.show()

# Пункт 3: Предсказание значения cnt
new_weather_condition = np.array([[3]])  # Замените 3 на ваше значение
predicted_demand = linear_model.predict(new_weather_condition)
print(f'Предсказанное количество аренд: {predicted_demand[0]}')

# Пункт 4: Уменьшение размерности и построение 2D графика
dimensionality_reducer = PCA(n_components=2)
reduced_data = dimensionality_reducer.fit_transform(bike_data.drop('cnt', axis=1))

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target)
plt.title('2D график предсказания cnt')
plt.show()

# Пункт 5: Регуляризация Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(bike_data.drop('cnt', axis=1), target)

# Определение признака, который оказывает наибольшее влияние на cnt
feature_coefficients = pd.Series(lasso_model.coef_, index=bike_data.drop('cnt', axis=1).columns)
print(f'Признак, оказывающий наибольшее влияние на cnt: {feature_coefficients.idxmax()}')
