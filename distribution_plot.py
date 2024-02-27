import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
st.title('Распределение признака')
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data_streamlit.csv').iloc[:, 1:]

feature = st.selectbox("Выберите признак", df.columns)

plt.figure(figsize=(8, 6))
sns.histplot(df[feature], kde=True)
plt.title(f"Распределение признака {feature}")
plt.xlabel(feature)
plt.ylabel("Частота")
st.pyplot()
st.write(df[feature].describe())