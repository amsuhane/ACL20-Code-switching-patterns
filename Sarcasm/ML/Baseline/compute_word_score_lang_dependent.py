def ret_Score_Freq(train_tweets,labels):
    """
    To populate the score and freq
    """
    Score = {}
    Freq={}
    lang_agnostic_vocab = set()

    for tweet,label in zip(train_tweets,labels):
        for j in tweet:
            try:
                Freq[j]+=1
                Score[j][label]+=1
            except:
                Freq[j]=1
                Score[j][label]=[0,0]
                Score[j][label]+=1
    return Score,Freq

def create_Features(train_tweets,train_labels,test_tweets):
    """
    Create lang-dependent word features
    """
    Score,Freq=ret_Score_Freq(train_tweets,train_labels)

    train_word_dependent, test_word_dependent = [],[]

    for tweet_train,tweet_test in zip(train_tweets,test_tweets):
        features_train= []
        for word in tweet_train:
            if Freq[word]<5:
                features_train.append(0)
            else:
                score=max([Score[word][0]/Freq[word],Score[word][1]/Freq[word]])
                if score<0.6:
                    features_train.append(0)
                else:
                    features_train.append(score)
        features_test= []
        for word in tweet_test:
            if Freq[word]<5:
                features_test.append(0)
            else:
                score=max([Score[word][0]/Freq[word],Score[word][1]/Freq[word]])
                if score<0.6:
                    features_test.append(0)
                else:
                    features_test.append(score)
        train_word_dependent.append(features_train)
        test_word_dependent.appemd(features_test)
    return train_word_dependent,test_word_dependent