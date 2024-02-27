import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
st.title('Распределение признака')
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data_streamlit.csv').iloc[:, 1:]

feature_dist = st.selectbox("Выберите признак", df.columns, key="selectbox_dist")

plt.figure(figsize=(8, 6))
sns.histplot(df[feature_dist], kde=True)
plt.title(f"Распределение признака {feature_dist}")
plt.xlabel(feature_dist)
plt.ylabel("Частота")
st.pyplot()
st.write(df[feature_dist].describe())



st.title('Матрица корреляций')

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Матрица корреляций")
st.pyplot()
st.write(corr_matrix)



st.title('График зависимости целевой переменной и признака (усатый ящик)')

feature_dep = st.selectbox("Выберите признак", df.columns, key="selectbox_dep")

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['TARGET'], y=df[feature_dep])
plt.title(f"Зависимость между {feature_dep} и целевой переменной")
plt.ylabel(feature_dep)
plt.xlabel("Целевая переменная")
st.pyplot()

correlation = df[[feature_dep, 'TARGET']].corr().iloc[0, 1]
st.write(f"Корреляция между {feature_dep} и целевой переменной: {correlation}")



st.title('Дескриптивные статистики')

features = ['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER',	'CHILD_TOTAL',
            'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_CLOSED',	'LOAN_NUM_TOTAL']
selected_columns_stat = st.multiselect("Выберите столбцы", df.columns, default=features)

statistics_df = df[selected_columns_stat].describe().transpose()

st.write(statistics_df)



st.title('Моделирование отклика на маркетинговую компанию (модель случайного леса)')

features = ['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER',	'CHILD_TOTAL',
            'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_CLOSED',	'LOAN_NUM_TOTAL']
feature_columns = st.multiselect("Выберите признаки", df.columns, default=features)
target_variable = st.selectbox("Выберите целевую переменную", df.columns, index=2)

X = df[feature_columns]
y = df[target_variable]

model = RandomForestClassifier()
model.fit(X, y)

input_data = {}
for feature in feature_columns:
    input_data[feature] = st.number_input(f"Введите значение для {feature}", value=0.0)

if st.button("Получить предсказание"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write("Предсказание:", prediction)
