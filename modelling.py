import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Моделирование отклика на маркетинговую компанию (модель случайного леса)')
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data_streamlit.csv').iloc[:, 1:]

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
