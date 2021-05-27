from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import re
import csv
from time import sleep
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

vader = SentimentIntensityAnalyzer()
classifier = TextClassifier.load('en-sentiment')


def get_sentiment(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    flair_score = sentence.labels[0].score
    
    if sentence.labels[0].value=='NEGATIVE':
        flair_score *= -1
    
    nltk_score = vader.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity
    avg = np.nanmean([flair_score, nltk_score, textblob_score])
    return dict(Flair=flair_score, nltk=nltk_score, textblob=textblob_score, avg=avg)

def score_articles(articles):
    flair_scores = []
    textblob_scores = []
    nltk_scores = []
    avg_scores = []
    weights = []
    for article in articles:
        headline = article[0]
        d = get_sentiment(headline)
        flair_scores.append(d['Flair'])
        textblob_scores.append(d['textblob'])
        nltk_scores.append(d['nltk'])
        avg_scores.append(d['avg'])
        weight = get_weight_in_days(article[2])
        weights.append(weight)

    # Adjust weights such that more recent ones are assigned higher weight linearly
    weights = (1-weights) + np.min(1-weights)
    return [flair_scores, textblob_scores, nltk_scores, avg_scores], weights

def get_weight_in_days(article):
    times = ['year', 'month', 'week', 'day', 'hour', 'minute', 'second']

    period = '22 weeks ago'
    sep = period.split(' ')
    if sep[1] == times[0] or sep[1] == times[0]+'s':
        weight = float(sep[0]) * 365
    elif sep[1] == times[1] or sep[1] == times[1]+'s':
        weight = float(sep[0]) * 30
    elif sep[1] == times[2] or sep[1] == times[2]+'s':
        weight = float(sep[0]) * 7
    elif sep[1] == times[3] or sep[1] == times[3]+'s':
        weight = float(sep[0])
    elif sep[1] == times[4] or sep[1] == times[4]+'s':
        weight = float(sep[0]) / 24
    elif sep[1] == times[5] or sep[1] == times[5]+'s':
        weight = float(sep[0]) / (24*60)
    elif sep[1] == times[6] or sep[1] == times[6]+'s':
        weight = float(sep[0]) / (24*60*60)
    else:
        print('unknown time frame')
        weight = sep[0]
    return weight


def plot_sentiment(keyword, scores, weights):
    flair_scores, nltk_scores, textblob_scores, avg_scores = scores
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(keyword)

    plt.subplot(141)
    sns.violinplot(data=flair_scores)
    plt.title('Flair')
    plt.ylim(-1, 1)
    xlim = plt.xlim()
    w_avg = np.average(flair_scores, weights=weights)
    plt.plot(xlim, [w_avg, w_avg], color='green')
    avg = np.mean(flair_scores)
    plt.plot(xlim, [avg, avg], color='orange')


    plt.subplot(142)
    sns.violinplot(data=nltk_scores)
    plt.title('nltk')
    plt.ylim(-1, 1)
    xlim = plt.xlim()
    w_avg = np.average(nltk_scores, weights=weights)
    plt.plot(xlim, [w_avg, w_avg], color='green')
    avg = np.mean(flair_scores)
    plt.plot(xlim, [avg, avg], color='orange')


    plt.subplot(143)
    sns.violinplot(data=textblob_scores)
    plt.title('textblob')
    plt.ylim(-1, 1)
    xlim = plt.xlim()
    w_avg = np.average(textblob_scores, weights=weights)
    plt.plot(xlim, [w_avg, w_avg], color='green')
    avg = np.mean(flair_scores)
    plt.plot(xlim, [avg, avg], color='orange')

    plt.subplot(144)
    sns.violinplot(data=avg_scores)
    plt.title('Avg.')
    plt.ylim(-1, 1)
    xlim = plt.xlim()
    w_avg = np.average(avg_scores, weights=weights)
    plt.plot(xlim, [w_avg, w_avg], color='green')
    avg = np.mean(flair_scores)
    plt.plot(xlim, [avg, avg], color='orange')
    print(f'Green: Weighted Arithmetic Average\nOrange: Arithmetic Average')

def get_article(card):
    """Extract article information from the raw html"""
    headline = card.find('h4', 's-title').text
    source = card.find("span", 's-source').text
    posted = card.find('span', 's-time').text.replace('·', '').strip()
    description = card.find('p', 's-desc').text.strip()
    raw_link = card.find('a').get('href')
    unquoted_link = requests.utils.unquote(raw_link)
    pattern = re.compile(r'RU=(.+)\/RK')
    clean_link = re.search(pattern, unquoted_link).group(1)
    
    article = (headline, source, posted, description, clean_link)
    return article

def get_the_news(search, maxpage=100):
    """Run the main program"""
    
    headers = {
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'referer': 'https://www.google.com',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
    }

    template = 'https://news.search.yahoo.com/search?p={}'
    url = template.format(search)
    articles = []
    links = set()
    cnt = 0
    while cnt < maxpage:
        print(f'Searching page {cnt}')
        cnt += 1
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        cards = soup.find_all('div', 'NewsArticle')
        
        # extract articles from page
        for card in cards:
            article = get_article(card)
            link = article[-1]
            if not link in links:
                links.add(link)
                articles.append(article)        
                
        # find the next page
        try:
            url = soup.find('a', 'next').get('href')
            sleeptime = get_stochastic_sleeptime()
            sleep(sleeptime)
        except AttributeError:
            break
        
            
    # save article data
    # with open('results.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Headline', 'Source', 'Posted', 'Description', 'Link'])
    #     writer.writerows(articles)
    print(f'Found {len(articles)} articles.')
    return articles
    
def get_stochastic_sleeptime():
    ''' Returns a number between approx. -0.8 and 1.2 for jittered sleep time between requests'''
    # sleeptime = ((np.random.rand(1)+1)*0.2)[0] * np.random.choice([-1, 1])
    sleeptime = 1 + (np.random.rand()-0.5) * 0.5
    return sleeptime