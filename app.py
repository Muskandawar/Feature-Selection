# 28 seconds
def classification(file_name):
    import time
    startTime = time.time()
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    from boruta import BorutaPy
    from sklearn.model_selection import train_test_split
    df=pd.read_csv(file_name)
    df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna()
    df = df.astype(float)
    y = df['Target'].values
    X=df.drop(['Target'],axis=1)
    col=X.columns.tolist()
    col = ",".join(col)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(X,y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3,random_state=33,stratify=y)


    #################################### BORUTA ####################################################


    rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)
    boruta_selector = BorutaPy(rfc, n_estimators='auto')
    boruta_selector.fit(X_train.values, y_train) 
    rank=boruta_selector.ranking_.tolist()

    writefp=open("Ranks.csv",'w')

    s = [str(i) for i in rank] 
    res = (",".join(s))
    writefp.write('Classifiers'+','+ col + '\n')
    writefp.write('Boruta Feature Selection'+','+res + '\n')
    # writefp.write("\n\n\n")
    writefp.close() 

    ############################# RECURSIVE FEATURE ELIMINATION ###########################################


    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model,1)
    fit = rfe.fit(X_train, y_train)
    Rank_rfe = fit.ranking_.tolist() 


    writefp=open("Ranks.csv",'a')

    s = [str(i) for i in Rank_rfe] 
    res = (",".join(s)) 
    writefp.write('Recursive Feature Elimination'+','+res+'\n')
    # writefp.write("\n\n\n")
    writefp.close() 


    ###################################### SELECT K BEST #####################################################


    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, mutual_info_classif,f_classif

    num_features = len(X_train.columns)

    test = SelectKBest(score_func=f_classif, k=2)
    test.fit(X_train, y_train)
    scores = []
    for i in range(num_features): 
        scores.append(test.scores_[i])
            
    Ranks = sorted(scores, reverse = True)

    writefp=open("Ranks.csv",'a')

    s = [str(i) for i in Ranks] 
    res = (",".join(s)) 
    writefp.write('Select K Best,'+res+'\n')

    writefp.close() 


    # ################################## RANDOM FOREST CLASSIFIER #######################################


    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(X_train, y_train)


    writefp=open("Ranks.csv",'a')

    s = [str(i) for i in clf.feature_importances_] 
    res = (",".join(s)) 
    writefp.write('Random Forest Classifier,'+res+'\n')
    writefp.close() 


    # ############################## EXTRA TREES CLASSIFIER #######################################


    #METHOD 5
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    writefp=open("Ranks.csv",'a')

    s = [str(i) for i in importances] 
    res = (",".join(s)) 
    writefp.write('Extra Trees Classifier,'+res+'\n')
    writefp.close() 
    writefp.close() 

    # ############################ CORRELATION ########################################################

    corr = []
    for i in X.columns.tolist():
        corr.append(df['Target'].corr(df[i]))


    writefp=open("Ranks.csv",'a')
    s = [str(i) for i in corr] 
    res = (",".join(s))

    writefp.write('Correlation With Target,'+res+'\n')

    writefp.close()

    # ##################################################################################################

    endTime = time.time()
    final_time=endTime - startTime
    def convert(seconds): 
        seconds = seconds % (24 * 3600) 
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        
        return "%d:%02d:%02d" % (hour, minutes, seconds) 
      
    n = final_time
    print(convert(n))

classification('Sample12 - Default of credit card clients (2).csv')
