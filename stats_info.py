import streamlit as st
import pandas as pd
st.title('Дескриптивные статистики')
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data_streamlit.csv').iloc[:, 1:]

features = ['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER',	'CHILD_TOTAL',
            'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_CLOSED',	'LOAN_NUM_TOTAL']
selected_columns = st.multiselect("Выберите столбцы", df.columns, default=features)

statistics_df = df[selected_columns].describe().transpose()

st.write(statistics_df)