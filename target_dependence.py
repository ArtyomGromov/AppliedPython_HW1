import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
st.title('График зависимости целевой переменной и признака (усатый ящик)')
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data_streamlit.csv').iloc[:, 1:]

feature = st.selectbox("Выберите признак", df.columns)

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['TARGET'], y=df[feature])
plt.title(f"Зависимость между {feature} и целевой переменной")
plt.ylabel(feature)
plt.xlabel("Целевая переменная")
st.pyplot()

correlation = df[[feature, 'TARGET']].corr().iloc[0, 1]
st.write(f"Корреляция между {feature} и целевой переменной: {correlation}")