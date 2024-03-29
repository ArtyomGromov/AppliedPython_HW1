# -*- coding: utf-8 -*-
"""Прикладной питон дз-1 streamlit Громов Артем.ipynb"

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DYV01mgS_0rdAlNSeeUhJB2iO8N_XGNF

# Практическая работа

# Задача

Один из способов повысить эффективность взаимодействия банка с клиентами — отправлять предложение о новой услуге не всем клиентам, а только некоторым, которые выбираются по принципу наибольшей склонности к отклику на это предложение.

Задача заключается в том, чтобы предложить алгоритм, который будет выдавать склонность клиента к положительному или отрицательному отклику на предложение банка. Предполагается, что, получив такие оценки для некоторого множества клиентов, банк обратится с предложением только к тем, от кого ожидается положительный отклик.

Для решения этой задачи загрузите файлы из базы в Postgres (или используйте `*.csv` как есть).
Эта БД хранит информацию о клиентах банка и их персональные данные, такие как пол, количество детей и другие.

Описание таблиц с данными представлено ниже.

**D_work**

Описание статусов относительно работы:
- ID — идентификатор социального статуса клиента относительно работы;
- COMMENT — расшифровка статуса.


**D_pens**

Описание статусов относительно пенсии:
- ID — идентификатор социального статуса;
- COMMENT — расшифровка статуса.


**D_clients**

Описание данных клиентов:
- ID — идентификатор записи;
- AGE	— возраст клиента;
- GENDER — пол клиента (1 — мужчина, 0 — женщина);
- EDUCATION — образование;
- MARITAL_STATUS — семейное положение;
- CHILD_TOTAL	— количество детей клиента;
- DEPENDANTS — количество иждивенцев клиента;
- SOCSTATUS_WORK_FL	— социальный статус клиента относительно работы (1 — работает, 0 — не работает);
- SOCSTATUS_PENS_FL	— социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер);
- REG_ADDRESS_PROVINCE — область регистрации клиента;
- FACT_ADDRESS_PROVINCE — область фактического пребывания клиента;
- POSTAL_ADDRESS_PROVINCE — почтовый адрес области;
- FL_PRESENCE_FL — наличие в собственности квартиры (1 — есть, 0 — нет);
- OWN_AUTO — количество автомобилей в собственности.


**D_agreement/target**

Таблица с зафиксированными откликами клиентов на предложения банка:
- AGREEMENT_RK — уникальный идентификатор объекта в выборке;
- ID_CLIENT — идентификатор клиента;
- TARGET — целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было).
    
    
**D_job**

Описание информации о работе клиентов:
- GEN_INDUSTRY — отрасль работы клиента;
- GEN_TITLE — должность;
- JOB_DIR — направление деятельности внутри компании;
- WORK_TIME — время работы на текущем месте (в месяцах);
- ID_CLIENT — идентификатор клиента.


**D_salary**

Описание информации о заработной плате клиентов:
- ID_CLIENT — идентификатор клиента;
- FAMILY_INCOME — семейный доход (несколько категорий);
- PERSONAL_INCOME — личный доход клиента (в рублях).


**D_last_credit**

Информация о последнем займе клиента:
- ID_CLIENT — идентификатор клиента;
- CREDIT — сумма последнего кредита клиента (в рублях);
- TERM — срок кредита;
- FST_PAYMENT — первоначальный взнос (в рублях).


**D_loan**

Информация о кредитной истории клиента:
- ID_CLIENT — идентификатор клиента;
- ID_LOAN — идентификатор кредита.

**D_close_loan**

Информация о статусах кредита (ссуд):
- ID_LOAN — идентификатор кредита;
- CLOSED_FL — текущий статус кредита (1 — закрыт, 0 — не закрыт).

Ниже представлен минимальный список колонок, которые должны находиться в итоговом датасете после склейки и агрегации данных. По своему усмотрению вы можете добавить дополнительные к этим колонки.

- AGREEMENT_RK — уникальный идентификатор объекта в выборке;
    - TARGET — целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было);
    - AGE — возраст клиента;
    - SOCSTATUS_WORK_FL — социальный статус клиента относительно работы (1 — работает, 0 — не работает);
    - SOCSTATUS_PENS_FL — социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер);
    - GENDER — пол клиента (1 — мужчина, 0 — женщина);
    - CHILD_TOTAL — количество детей клиента;
    - DEPENDANTS — количество иждивенцев клиента;
    - PERSONAL_INCOME — личный доход клиента (в рублях);
    - LOAN_NUM_TOTAL — количество ссуд клиента;
    - LOAN_NUM_CLOSED — количество погашенных ссуд клиента.

Будьте внимательны при сборке датасета: это реальные банковские данные, в которых могут наблюдаться дубли, некорректно заполненные значения или значения, противоречащие друг другу. Для получения качественной модели необходимо предварительно очистить датасет от такой информации.

## Задание 1

Соберите всю информацию о клиентах в одну таблицу, где одна строчка соответствует полной информации об одном клиенте.
"""

import pandas as pd

!apt-get install -y git

!git clone https://github.com/ksusonic/bank-ml-mailing.git

# Commented out IPython magic to ensure Python compatibility.
# %cd bank-ml-mailing

df_clients = pd.read_csv('data/D_clients.csv')
df_close_loan = pd.read_csv('data/D_close_loan.csv')
df_job = pd.read_csv('data/D_job.csv')
df_last_credit = pd.read_csv('data/D_last_credit.csv')
df_loan = pd.read_csv('data/D_loan.csv')
df_pens = pd.read_csv('data/D_pens.csv')
df_salary = pd.read_csv('data/D_salary.csv')
df_target = pd.read_csv('data/D_target.csv')
df_work = pd.read_csv('data/D_work.csv')

df_loan_subset = df_close_loan.merge(df_loan, on='ID_LOAN', how='left')
df_loan_subset = df_loan_subset.groupby(['ID_CLIENT', 'CLOSED_FL']).size().unstack(fill_value=0).reset_index()
df_loan_subset['LOAN_NUM_TOTAL'] = df_loan_subset.iloc[:, 1] + df_loan_subset.iloc[:, 2]
df_loan_subset['LOAN_NUM_CLOSED'] = df_loan_subset.iloc[:, 2]
df_loan_subset = df_loan_subset[['ID_CLIENT', 'LOAN_NUM_CLOSED', 'LOAN_NUM_TOTAL']]

df_loan_subset['LOAN_NUM_TOTAL'] = df_loan_subset.iloc[:, 1] + df_loan_subset.iloc[:, 2]
df_loan_subset['LOAN_NUM_CLOSED'] = df_loan_subset.iloc[:, 2]

#merge
df_target_subset = df_target[['ID_CLIENT', 'AGREEMENT_RK', 'TARGET']]
df_clients_subset = df_clients[['ID', 'AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS']]
df_clients_subset['ID_CLIENT'] = df_clients_subset['ID']
df_salary_subset = df_salary[['ID_CLIENT', 'PERSONAL_INCOME']]

df_loan_subset = df_close_loan.merge(df_loan, on='ID_LOAN', how='left')
df_loan_subset = df_loan_subset.groupby(['ID_CLIENT', 'CLOSED_FL']).size().unstack(fill_value=0).reset_index()
df_loan_subset['LOAN_NUM_TOTAL'] = df_loan_subset.iloc[:, 1] + df_loan_subset.iloc[:, 2]
df_loan_subset['LOAN_NUM_CLOSED'] = df_loan_subset.iloc[:, 2]
df_loan_subset = df_loan_subset[['ID_CLIENT', 'LOAN_NUM_CLOSED', 'LOAN_NUM_TOTAL']]


df = df_target_subset.merge(df_clients_subset, on='ID_CLIENT', how='left')
df = df.merge(df_salary_subset, on='ID_CLIENT', how='left')
df = df.merge(df_loan_subset, on='ID_CLIENT', how='left')

df = df.drop_duplicates()
df = df.drop(columns = 'ID')

df.head(2)

df.describe()

df.info()

df.to_csv('data_streamlit.csv')

"""## Задание 2

При помощи инструмента Streamlit проведите разведочный анализ данных. В него может входить:

* построение графиков распределений признаков
* построение матрицы корреляций
* построение графиков зависимостей целевой переменной и признаков
* вычисление числовых характеристик распределения числовых столбцов (среднее, min, max, медиана и так далее)
* любые другие ваши идеи приветствуются!

[Пример Streamlit-приложения](https://rateyourflight.streamlit.app) с разведочным анализом, прогнозом модели и оценкой ее результатов.
"""
