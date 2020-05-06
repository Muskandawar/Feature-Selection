# 193 seconds
def regression(file_name):
    import time
    startTime = time.time()

    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
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
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3,random_state=33)


    ################################################# SELECT K BEST ##############################################################################

    #Selected
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, mutual_info_regression,f_regression

    num_features = len(X_train.columns)

    test = SelectKBest(score_func=f_regression, k=2)
    test.fit(X_train, y_train)
    scores = []
    for i in range(num_features): 
        scores.append(test.scores_[i])
            
    Ranks = sorted(scores, reverse = True)

    writefp=open("Ranks_reg.csv",'w')

    s = [str(i) for i in Ranks] 
    res = (",".join(s)) 
    writefp.write('Classifiers,'+col+'\n')
    writefp.write('Select K Best,'+res+'\n')
    writefp.close() 

    ##################################################### EXTRA TREES REGRESSOR ###########################################################################


    #Selected
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesRegressor

    # Build a forest and compute the feature importances
    forest = ExtraTreesRegressor(n_estimators=250,
                                random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    writefp=open("Ranks_reg.csv",'a')

    s = [str(i) for i in importances] 
    res = (",".join(s)) 
    writefp.write('Extra Trees Regressor,'+res+'\n')
    writefp.close() 


    ############################################# RANDOM FOREST REGRESSOR ####################################################################3

    # Takes 186 seconds to complete

    #Selected

    clf = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)

    clf.fit(X_train, y_train)


    writefp=open("Ranks_reg.csv",'a')

    s = [str(i) for i in clf.feature_importances_] 
    res = (",".join(s)) 
    writefp.write('Random Forest Regressor,'+res+'\n')
    writefp.close()  


    ######################################### RIDGE ########################################

    # fast
    #Selected
    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=7)
    ridge.fit(X_train, y_train)

    writefp=open("Ranks_reg.csv",'a')

    s = [str(i) for i in np.abs(ridge.coef_)] 
    res = (",".join(s)) 
    writefp.write('Ridge Regressor,'+res + '\n')
    writefp.close()

    ###################################### LINEAR REGRESSION ###################################

    #Fast
    #Selected
    from sklearn.linear_model import LinearRegression 
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)

    writefp=open("Ranks_reg.csv",'a')

    s = [str(i) for i in np.abs(lr.coef_)] 
    res = (",".join(s)) 
    writefp.write('Linear Regression ,'+res+'\n')
    writefp.close()


    ################################ F_REGRESSOR #################################################


    from sklearn.feature_selection import RFE, f_regression
    f, pval  = f_regression(X_train, y_train, center=True)

    writefp=open("Ranks_reg.csv",'a')

    s = [str(i) for i in f] 
    res = (",".join(s))
    s = [str(i) for i in pval] 
    res1 = (",".join(s))  
    writefp.write('F_regressor,'+res+'\n')
    writefp.close()


    ################################# LASSO ######################################################


    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.05,max_iter=5000)
    lasso.fit(X_train, y_train)

    writefp=open("Ranks_reg.csv",'a')
    s = [str(i) for i in np.abs(lasso.coef_)] 
    res = (",".join(s))
    writefp.write('Lasso ,'+res+'\n')
    writefp.close()

    ############################# RANDOMIZED LASSO ################################################


    from sklearn.linear_model import RandomizedLasso
    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X_train, y_train)

    writefp=open("Ranks_reg.csv",'a')
    s = [str(i) for i in np.abs(rlasso.scores_)] 
    res = (",".join(s))
    writefp.write('Randomized Lasso,'+res+'\n')
    writefp.close()


    ############################ CORRELATION ########################################################

    corr = []
    for i in X.columns.tolist():
        corr.append(df['Target'].corr(df[i]))


    writefp=open("Ranks_reg.csv",'a')
    s = [str(i) for i in corr] 
    res = (",".join(s))
    writefp.write('Correlation With Target,'+res+'\n')
    writefp.close()

    #################
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
regression('uniqe_output_data.csv')



