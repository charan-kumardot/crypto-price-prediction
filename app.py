from flask import Flask, render_template
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import requests
import re
from transformers import pipeline
import requests
from decouple import config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import json
import plotly
import plotly.graph_objs as go
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import price
import os
import json
import plotly.io as pio
from datetime import datetime
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import re
from transformers import pipeline
import csv


app = Flask(__name__)  # creating the Flask class object
count = 0


def price():
    crypto_currency = 'BTC'
    against_currency = 'USD'
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()
    data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)
    # prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 60
    future_day = 30
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Create Neural Network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    # Testing the model
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
    print(test_data)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    print("len",len(model_inputs))
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        print(x)
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)

    prediction_prices = scaler.inverse_transform(prediction_prices)

    prices = [item for sublist in prediction_prices for item in sublist]

    print(prices)
    df = pd.DataFrame({'prices': prices})

    df.to_csv('result.csv', index=False)
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(prediction_prices, color='green', label='Predicted Prices')
    plt.title(f'{crypto_currency} price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    return prices

def create_plot():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    print(current_time)
    if current_time=="10:13" or current_time=="10:14":
        print("true")
        if os.path.exists("result.csv"):
           os.remove("result.csv")
        price()

    df2 = pd.read_csv("result.csv")
    fig = px.line(df2, y='prices',title='price prediction',template="plotly_dark")
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(width=1100, height=450)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON



def create_output_array(summaries, scores, urls):
    output = []
    for ticker in ['BTC']:
        for counter in range(len(summaries[ticker])):
            output_this = [
                            ticker,
                            summaries[ticker][counter],
                            scores[ticker][counter]['label'],
                            scores[ticker][counter]['score'],
                            urls[ticker][counter]
                          ]
            output.append(output_this)
    return output



count = 0

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sentiment')
def sentiment():
    results = []
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    print(current_time)
    if current_time == "01:17" or current_time == "01:18":
        if os.path.exists("ethsummaries.csv"):
           os.remove("ethsummaries.csv")
        # sentiment

        # 2. Setup Model
        model_name = "human-centered-summarization/financial-summarization-pegasus"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        # 3. Setup Pipeline
        monitored_tickers = ['BTC']

        # 4.1. Search for Stock News using Google and Yahoo Finance
        print('Searching for stock news for', monitored_tickers)

        def search_for_stock_news_links(ticker):
            search_url = 'https://www.google.com/search?q=yahoo+finance+{}&tbm=nws'.format(ticker)
            r = requests.get(search_url)
            soup = BeautifulSoup(r.text, 'html.parser')
            atags = soup.find_all('a')
            hrefs = [link['href'] for link in atags]
            return hrefs

        raw_urls = {ticker: search_for_stock_news_links(ticker) for ticker in monitored_tickers}

        # 4.2. Strip out unwanted URLs
        print('Cleaning URLs.')
        exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

        def strip_unwanted_urls(urls, exclude_list):
            val = []
            for url in urls:
                if 'https://' in url and not any(exc in url for exc in exclude_list):
                    res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                    val.append(res)
            return list(set(val))

        cleaned_urls = {ticker: strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}

        # 4.3. Search and Scrape Cleaned URLs
        print('Scraping news links.')

        def scrape_and_process(URLs):
            ARTICLES = []
            for url in URLs:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                results = soup.find_all('p')
                text = [res.text for res in results]
                words = ' '.join(text).split(' ')[:350]
                ARTICLE = ' '.join(words)
                ARTICLES.append(ARTICLE)
            return ARTICLES

        articles = {ticker: scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}

        # 4.4. Summarise all Articles
        print('Summarizing articles.')

        def summarize(articles):
            summaries = []
            for article in articles:
                input_ids = tokenizer.encode(article, return_tensors="pt")
                output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
                summary = tokenizer.decode(output[0], skip_special_tokens=True)
                summaries.append(summary)
            return summaries

        summaries = {ticker: summarize(articles[ticker]) for ticker in monitored_tickers}

        # 5. Adding Sentiment Analysis
        print('Calculating sentiment.')
        sentiment = pipeline("sentiment-analysis")
        scores = {ticker: sentiment(summaries[ticker]) for ticker in monitored_tickers}

        # # 6. Exporting Results
        print('Exporting results')
        final_output = create_output_array(summaries, scores, cleaned_urls)
        with open('ethsummaries.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,encoding='cp1252',engine = 'python')
            csv_writer.writerows(final_output)

    with open("ethsummaries.csv") as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:
            results.append(row)
    return render_template('sentiment.html', final_output=results, count=count, length=len(results))




@app.route('/news')
def news():
    news_data = requests.get('https://min-api.cryptocompare.com/data/v2/news/?lang=EN').json()
    news_articles = news_data['Data']
    return render_template('news.html', news_articles=news_articles)


@app.route('/priceprediction')
def price_prediction():
    plot = create_plot()
    price_data = requests.get('https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false').json()
    return render_template('price.html', plot=plot,price_data=price_data)


if __name__ == '__main__':
    app.run(debug=False)
