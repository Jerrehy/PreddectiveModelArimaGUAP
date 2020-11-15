import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import warnings

warnings.filterwarnings(action='ignore')
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

pytrend = TrendReq(hl='en-US', tz=360)


def arima(keyword):
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d')

    series = read_csv('{0}.csv'.format(keyword),
                      header=0, parse_dates=[0],
                      index_col=0, squeeze=True, date_parser=parser)
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    b = plt.figure()
    plt.plot(test)
    plt.plot(predictions, color='red')

    plt.xlabel("Number of observations")
    plt.ylabel("Requests, %")

    st.write(b)


def graph(data, keyword):
    a = plt.figure()

    plt.plot(data)

    plt.xlabel("Date, year")
    plt.ylabel("Requests, %")

    st.write(a)


def load_data(keywords, profile):
    pytrend.build_payload(
        kw_list=keywords,
        cat=0,
        timeframe='today 5-y',
        geo='',
        gprop='', )
    data = pytrend.interest_over_time()
    data = data.drop(labels=['isPartial'], axis='columns')
    plt.clf()
    data.to_csv('{0}.csv'.format(profile), encoding='utf_8_sig')
    return data


st.title('Preddiction Model.')

st.write('Выбор специальности')

list_specialties = ['System analyst', 'Data Scientist', 'System administrator',
                    'Frontend developer', 'Backend developer', 'System architect'
    , 'Tester QA', 'Game developer', 'Security Researcher'
    , 'UI UX designer']

prof = st.selectbox("model_select", list_specialties)

data = load_data([prof], prof)

st.write(data)

'''График процентного использования в поиске выбранной специальности'''
graph(data, prof)

'''Модель прогнозирования на основе  интегрированная модель авторегрессии — скользящего среднего '''
'''ARIMA'''
arima(prof)

'''Создается линейный график, показывающий ожидаемые значения (синим цветом) 
    по сравнению с прогнозами скользящего прогноза (красным).'''
