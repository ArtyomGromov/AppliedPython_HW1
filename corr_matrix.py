import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title('Матрица корреляций')
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data_streamlit.csv').iloc[:, 1:]

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Матрица корреляций")
st.pyplot()
st.write(corr_matrix)