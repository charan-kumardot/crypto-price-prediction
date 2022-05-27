import requests
from decouple import config

def get_latest_news():
    news_data = requests.get('https://newsdata.io/api/1/news?apikey=pub_69828a26307fd0cfc298005cbdeb237d8676&q=cryptocurrency&language=en').json()
    return news_data['results']



print(get_latest_news())


