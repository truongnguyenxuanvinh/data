try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain

def install(package):
	pipmain(['install', package])

if __name__ == '__main__':
    install('vaderSentiment')
    install('pandas')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from transformers import pipeline

df = pd.read_csv('data.csv', encoding="ISO-8859-1")

age_classifier = pipeline("text-classification", model="ayaenna/Age_Prediction", return_all_scores=True)
age_list = []
df['age'] = ""

for i in range(len(df)):

  print(i)

  age = age_classifier(df.iloc[i][3][:512])
  age_list.append(age)

series_age = pd.Series(age_list)
df['age'] = series_age.values

df.to_csv('data_classified.csv')