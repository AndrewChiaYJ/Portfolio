# Portfolio

Hello! I am Andrew, and I am aspiring to be Data Scientist, after working as a Data Analyst in a semiconductor firm. This is a summary of various projects I have worked on. More specific technical details are in their respective githubs (links included). If you are interested to know more, feel free to [get in touch](mailto:chiayj95@hotmail.com), or check out my [Resume](../master/Resume_Chia%20Yih%20Jeng.pdf) or [LinkedIn](https://www.linkedin.com/in/andrewchiayj/). Thank you! 


### Contents

1. [Data Science Projects](#Data-Science-Projects)

## Data Science Projects

1. Fake News Classification (General Assembly Course)

*Sep 2021 | https://github.com/AndrewChiaYJ/Fake-News-Classification*

<img src="./Visualisations/Fake_News_Classifications.png">

In this project, I have looked into fake news, in particular in US. Fake news, as a word, became a buzzword in 2016 US Presidential Election. It's definition is that false or misleading information presented as news. The purpose of fake news is to damage the reputation of a person or entity, or making money through advertising revenue. Fake News is dangerous to the economy if not dealt with properly and swiftly. According to a report published by  [CHEC](https://s3.amazonaws.com/media.mediapost.com/uploads/EconomicCostOfFakeNews.pdf)  in the year 2019, it was estimated that the global economic cost of fake news to be $78 billion per year which includes direct and indirect cost incurred in the following areas: stock markets, media, reputation management, election campaigns, financial information and healthcare.

This project uses NLP techniques to clean (remove punctuations, remove stop words, tokenize), lemmatize, and vectorize the words (count vectorization, TF-IDF Vectorization) in news articles published near the 2016 US Presidential Election time period. After that, the vectorized data are being processed by multiple machine learning classification models. This is to train the models, and so as to be able to predict if a certain news article is either real or fake.

The final production model (XGBoost with Count Vectorization data) is able to classify the fake news with a high accuracy of 97%, and with a high F1 score (balance between precision and recall) of 98%. 

With this high accuracy model, a simple flask website is built, as shown in the image above. Users are able to use this website to fact check on the news articles that they read, and to be more aware that the issue of fake news is prevalent. 

Further expansions to include more news articles from various areas (healthcare, sports, entertainment), or from different countries (Singapore, Malaysia, Indonesia, UK, etc.) are possible.