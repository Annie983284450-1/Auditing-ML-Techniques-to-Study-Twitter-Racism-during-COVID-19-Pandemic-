@Summary of Anni Zhou's contributions
## Initializations 

+ Removing URLs from tweets
+ Converting all tweets to lowercase
+ Removing punctuations
+ Removing stopwords

__Example:__

0    #DidYouKnow: @WFP is 100% voluntarily funded: ...

1    To treat COVID-19, administration expected to ...

2    To treat COVID-19, administration expected to ...

3    @MichiganDOT will not accept cash transactions...

4    @MichiganDOT will not accept cash transactions...

__Target:__

0    didyouknow wfp 100 voluntarily funded every ai...

1    treat administration expected relax physician ...

2    treat administration expected relax physician ...

3    michigandot accept cash transactions eastbound...

4    michigandot accept cash transactions eastbound...



## Initial Analysis

We have plotted tweet frequency worldwide in a specifc time period (from Apr 2nd to Apr 5th, 2020). And we can see some peaks of the "coronavirus" topic in April on twitter. And we have also shown the words with high frequency in the corpus. 

## Wordcloud Visulization
 
We visualized the word cloud based on the whole dataset in a specifc time period (from Apr 2nd to Apr 5th, 2020), the positive tweets, negative tweets and neutral tweets, respectively. 

## Sentiment Analysis

We have done some simple sentiment analysis. From the result, we can see that most of the tweets in the test dataset are positive tweets. 

##  Term Frequency–inverse Document Frequency (TF-IDF)

TF-IDF is a numerical statistic that is intended to reflect the importance of a word to a document in a collection of corpus. TF-IDF is proportional to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word. In this project, the performance of TF-IDF is explored. For simplicity, the top-k frequent words in the corpus are treated as the features.  


```python

```
