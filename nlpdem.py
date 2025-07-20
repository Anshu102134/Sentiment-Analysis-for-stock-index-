


from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt


nltk.download('vader_lexicon')


newsapi = NewsApiClient(api_key='enter your  key ')  # <-- Replace with your NewsAPI key


articles = newsapi.get_everything(q='TATA MOTORS', language='en', sort_by='publishedAt', page_size=100)


headlines = [article['title'] for article in articles['articles']]
dates = [article['publishedAt'][:10] for article in articles['articles']]


analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(title)['compound'] for title in headlines]


def categorize(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

sentiment_labels = [categorize(score) for score in sentiment_scores]


df = pd.DataFrame({
    'headline': headlines,
    'date': pd.to_datetime(dates),
    'sentiment': sentiment_labels
})


sentiment_counts = df['sentiment'].value_counts()
sentiment_counts = sentiment_counts.reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)  # Keep order


plt.figure(figsize=(8, 8))
colors = ['#66b3ff', '#99ff99', '#ff9999']
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=140
)
plt.title('📰 News Sentiment for Reliance Industries')
plt.axis('equal')
plt.tight_layout()
plt.show()
