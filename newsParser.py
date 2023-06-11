from newspaper import Article
from newspaper import Config

import nltk

nltk.download('punkt')

# A new article from TOI
url = "http://timesofindia.indiatimes.com/world/china/chinese-expert-warns-of-troops-entering-kashmir/articleshow/59516912.cms"

# user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

# config = Config()
# config.browser_user_agent = user_agent
# config.request_timeout = 10


# For different language newspaper refer above table
toi_article = Article(url, language="en")  # en for English

# To download the article
toi_article.download()

# To parse the article
toi_article.parse()

# To perform natural language processing ie..nlp
toi_article.nlp()

# # To extract title
# print("Article's Title:")
# print(toi_article.title)
# print("n")
#
# # To extract text
# print("Article's Text:")
# print(toi_article.text)
# print("n")
#
# # To extract summary
# print("Article's Summary:")
# print(toi_article.summary)
# print("n")
#
# # To extract keywords
# print("Article's Keywords:")
# print(toi_article.keywords)

def getNewsTitleText(url):
    toi_article = Article(url, language="en") # en for English

    #To download the article
    toi_article.download()

    #To parse the article
    toi_article.parse()

    #To perform natural language processing ie..nlp
    toi_article.nlp()

    #To extract title
    title = toi_article.title
    # print("Article's Title:")
    # print(title)
    # print("n")

    #To extract text
    text = toi_article.text
    # print("Article's Text:")
    # print(toi_article.text)

    return title, text

