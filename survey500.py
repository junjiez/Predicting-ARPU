### This code was written in late 2016 to analyse the responses of 500+ participants to 75+ questions in a semi-randomised control trial to look at the effects of pico-solar ownership. There is a previous piece of code which deals with data cleaning (Cleaner.py) and another that analyses the longitudinal effects of solar ownership  (treatment.py). As well as survey data we have data from telecom partners on Malawian Mpesa equivalent and airtime usage etc etc.


#Import useful modules
import pandas as pd
import statsmodels.formula.api as smf
import sys
from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.decomposition import PCA
import seaborn as sns
import sklearn.linear_model  as lm
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
#from mpl_toolkits.mplot3d import Axes3D
#import clustering
from patsy import dmatrices, dmatrix, demo_data
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import normalize
import statsmodels.stats as sms
from sklearn.cross_validation import train_test_split
import random
import pylab as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import sklearn
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import LinearSVC as SVC
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
import sklearn.feature_selection as fs
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans
from sklearn import cluster
import math
from sklearn.feature_selection import chi2
from scipy import stats
import matplotlib.image as mpimg
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import colors
import six

#set some deafualts for plotting
mpl.rcParams['pdf.fonttype'] = 42
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.0})
old_settings = np.seterr(all='ignore')

print('The scikit-learn version is {}.'.format(sklearn.__version__))



def get_spaced_colours(n): #Random colour generator
    col=[]
    colours_ = list(six.iteritems(colors.cnames))
    for i in range(n):
        col.append(colours_[i][0])
    col=np.asarray(col)
    return col
       
def pd_boot1(data,query1,q2): #Crude bootstrap error estimator
        x1=[]
        for i in range(10):
            samp=data.sample(frac=0.67,replace=True)
            sampq1=samp[q2]
            key=sampq1.value_counts().keys()[0]
            cnts=samp[samp[q2]=='yes']["Age"].count()
            x1.append(samp[samp[query1]=='yes'][q2].value_counts().divide(cnts[0])*100)
        return np.std(x1) 
 
#Score Kbest algos by looking at Random Forest OOB score vs number of features included. Sanity check for SelectKBest algo. 
def Kbest_scoring(datas,y_target): 
    Kscores=[]
    for i in range(50):
        i=i+1
        Kbest=SelectKBest(chi2, k=len(datas.columns)/i)
        Kb_df=Kbest.fit_transform(datas, y_target)
        RFktest=RandomForestClassifier(n_estimators=500, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42) 
        RFktest.fit(Kb_df,y_target)
        Kscores.append(RFktest.oob_score_)
        print i, RFktest.oob_score_
    plt.figure("Kbest feature selection on Smart phones")
    ax=plt.subplot(111)
    ax.plot(range(len(Kscores)),Kscores)
    ax.set_ylim([0,1.0])
        

#Simple feature to re-group categorical features (i.e. to un-dummy them) and plot resulting graph.
def graph_feature_imp(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    #ensemble= Name of emsemble to be graphed
    #featurenames= list of names to display
    #autoscale
    #sumarrised_columns= list of colum prefixes to summarise on for dummy vars
    if autoscale:
        x_scale=model.feature_importances_.max()+headroom
    else:
        x_scale=1
        
    feature_dict=dict(zip(feature_names, model.feature_importances_))

    if summarized_columns:
        for col_name in summarized_columns:
            sum_val=sum(x for i, x in feature_dict.iteritems() if col_name in i)
            key_to_remov=[i for i in feature_dict.keys() if col_name in i]
            for i in key_to_remov:
                feature_dict.pop(i)
            feature_dict[col_name]=sum_val
    results = pd.Series(feature_dict.values(),index=feature_dict.keys())
    results.sort_values(inplace=True)
    return feature_dict

    
#Feature to select whether problem is a categorisation or regression problem, and to give an ordered list of feature importance for further manipulation
def RF_type(X,Y): #feed X and target data and get model, scoring and sorted features
    class_score=-0.1
    reg_score=-0.1
    typ=[]
    try:
        modelc=RandomForestClassifier(n_estimators=50, oob_score=True,n_jobs=-1,bootstrap=True)
        modelc.fit(X,Y)
        class_score=modelc.oob_score_.astype(float)
    except:
        "Ooops, looks like it's not a classification problem!"
        class_score=-5.0
    
    modelr=RandomForestRegressor(n_estimators=50, oob_score=True,n_jobs=-1,bootstrap=True)
    modelr.fit(X,Y)
    reg_score=modelr.oob_score_.astype(float)
    model=[]
    score1=[]
    if class_score >= modelr.oob_score_:
        
        print "Classifier", np.subtract(class_score,reg_score)*100., " %% certainity"
        model=RandomForestClassifier(n_estimators=1000, oob_score=True,n_jobs=-1,bootstrap=True)
        model.fit(X,Y)
        score1=model.oob_score_
        print "Classifier RF"
        typ='classifier'
    elif reg_score > class_score:
        print "Regressor",np.subtract(reg_score,class_score)*100., " %% certainity"
        model=RandomForestRegressor(n_estimators=1000, oob_score=True,n_jobs=-1,bootstrap=True)
        model.fit(X,Y)
        score1=model.oob_score_
        print "Regression RF"
        typ='regressor'
    print score1, "full RF score"

    """identify and sort most important features and errors"""
    feature_importances=pd.Series(model.feature_importances_,index=X.columns)
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0) # calculate tree errors
    feats=np.vstack((X.columns,model.feature_importances_))
    idx = np.argsort(feats[1])
    feats=feats[:,idx]
    var=feats[0,:] #sorted numvar by importance
    feature_importances.sort_values(inplace=True)
    print predictor, model.oob_score_, "Out-of-bag SCORE full RF system"
    #print "Sorted all features"
    for i in (-5,-4,-3,-2,-1): #print top 5 features
        print feature_importances[i],var[i], "RF features"
    #print feats[-5:]
    print "Average feature importance for top 5", np.mean(feats[1][-5:])  #feats[1][i]
    return model,score1, feats,typ
    

def OOB_vs_feats(X,Y,fea,typ): #recursive feature extraction: order the importances of features, eliminate a feature, calculate OOB score, repeat and plot. Thus allowing the relationship between OOB score and number of features to be examined
    scores=[]
    av_imps=[]
    feats=fea
    print len(feats[0][:])
    if typ=='classifier':
        print "FEATURE EXTRACTION", len(feats[0][:])
        for i in range(len(feats[0][:])-1):
            cnt=i
            print cnt,
            fes=feats[0]
            fes=fes[cnt:-1]
            slim=X[fes] 
            m=RandomForestClassifier(n_estimators=50, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)
            m.fit(slim,Y)
            scores.append(m.oob_score_)
            feature_importances=pd.Series(m.feature_importances_,index=slim.columns)
            std = np.std([tree.feature_importances_ for tree in m.estimators_],axis=0) # calculate tree errors
            f=np.vstack((slim.columns,m.feature_importances_))
            idx = np.argsort(f[1])
            f=f[:,idx]
            var=f[0,:] #sorted numvar by importance
            feature_importances.sort_values(inplace=True)
            av_imps.append(np.mean(f[1][-5:]))            
    elif typ=='regressor':
        print "FEATURE EXTRACTION", len(feats[0][:])
        for i in range(len(feats[0][:])-1):            
            print i,
            cnt=i
            fes=feats[0]
            fes=fes[cnt:-1]
            slim=X[fes] 
            m=RandomForestRegressor(n_estimators=50, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)
            m.fit(slim,Y)
            scores.append(m.oob_score_)
            feature_importances=pd.Series(m.feature_importances_,index=slim.columns)
            std = np.std([tree.feature_importances_ for tree in m.estimators_],axis=0) # calculate tree errors
            f=np.vstack((slim.columns,m.feature_importances_))
            idx = np.argsort(f[1])
            f=f[:,idx]
            var=f[0,:] #sorted numvar by importance
            feature_importances.sort_values(inplace=True)
            av_imps.append(np.mean(f[1][-5:]))
    else:
        print "last argument must be classifier or regressor string"
    
    fig=plt.figure("RF OOB and features")
    scores=np.asarray(scores)
    av_imps=np.asarray(av_imps)
    x=range(len(feats[0][:])-1)
    x=np.asarray(x,dtype=np.float32)
    x=(x/x.shape[0])*100
    #x=x[::-1] # reverse X order
    combi=scores*av_imps
    ax=plt.subplot(111)
    ax.plot(x, scores)
    ax.plot(x, av_imps)
    ax.plot(x, combi)
    fig.suptitle('Scoring RF', fontsize=20)
    plt.xlabel('Features excluded (%)', fontsize=18)
    plt.ylabel('Normalised score', fontsize=16)
    ax.legend(["OOB_scores","Av. top feature importances","overall score"]) 
    plt.show()  
 
def Ridge_a_plot(X,Y,typ,niters): #Plot R2 value vs penalisation for Ridge classifier
    print "Testing Ridge regularization..."
    rid_al_scores=[]
    
    #make range of alphas
    al0=0.0,0.0000001,0.0001,0.001,0.01
    al1=np.linspace(0.1,1.0,niters)
    al1=np.concatenate((al0,al1))
    al_arr=np.linspace(1.0,50.0,niters)
    al_arr=np.concatenate((al1,al_arr))
    if typ=='classifier':
        for i in range(niters):
            rid_al=lm.RidgeClassifier(alpha=al_arr[i],max_iter=1000)
            rid_al.fit(X,Y)
            rid_al_scores.append(np.nanmean(cross_validation.cross_val_score(rid_al, holdout, y_holdout , cv=5)))     
           
    else:
        print "last argument must be classifier or regressor string"
    
    fig=plt.figure("Ridge score and L2 penalisation")
    scores=np.asarray(rid_al_scores)

    x=range(len(rid_al_scores))
    x=np.asarray(x,dtype=np.float32)
    ax=plt.subplot(111)
    ax.plot(x, scores)
    fig.suptitle('Ridge penalisation', fontsize=20)
    plt.xlabel('Penalisation  value', fontsize=18)
    plt.ylabel('R2 score', fontsize=16)
    
    plt.show()     
          
def shuffle_frame(data_,num):  #sub-feature
    headers=[]
    for i in range(num):
        pos=random.randint(0,len(data_.columns)-1)
        headers.append(data_.columns[pos])
    return(data_[headers])


# function to report best scores from optimiser
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
#Sanity check - randomly shufle features and re-run a slimmed down RF on only those features- crude test for correlated features                            
def shuffle_test(data_,predy_df,typ,numbers): 
    master=[]
    master_scores=[]
    
    print "Shuffle " ,numbers
    print "Shuffle test - i.e. randomly pick 5 features and report OOB scores"
    for i in range(numbers):
        print i,
        #print "-----------------------------------------------------"
        datanew=shuffle_frame(data_,4)      
        score1=[]
        if typ=='classifier':
            model=RandomForestClassifier(n_estimators=50, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)
            RFshu, RF_full_feats=RF_params(model,10,data_,predy_df,typ)
            #model.fit(datanew,predy_df)
            score1=RFshu.oob_score_    
        elif typ=='regressor':
            model=RandomForestRegressor(n_estimators=50, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)
            model.fit(datanew,predy_df)
            score1=model.oob_score_
        else:
            print "Unknown ML type; choose classifier or regressor"
        #for i in datanew.columns:
            #print "%s," %(i)
        #print " "
        #print score1, "SHUFFLED SCORE #######"
        #print "---------------------------"
        master.append((datanew.columns)) 
        master_scores.append(score1)
    print ""
    print "Best shuffle result =",
    print np.max(master_scores)
    print master[np.argmax(master_scores)]
    return master[np.argmax(master_scores)]
    
      
    
                                              

#optimise RF parameters through random-search and cross validation
def RF_params(models,n_iter,X,Y,typ): 
    if typ=='classifier':
        print "RF param search..."
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                    "max_features": sp_randint(1, len(X.columns)),
                    "min_samples_split": sp_randint(1, 11),
                    "min_samples_leaf": sp_randint(1, 11),
                    "criterion": ["gini", "entropy"],
                    "min_impurity_split": [0.00000001,0.00001,0.0001,0.001,0.1]
                    
                    }
        
        # run randomized search
        random_search = RandomizedSearchCV(models, param_distributions=param_dist,n_iter=n_iter,refit=True)
        random_search.fit(X, Y)
        report(random_search.cv_results_)
        pRF=random_search.best_estimator_
    if typ=='regressor':
        print "RF param search..."
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                    "max_features": sp_randint(1, len(X.columns)),
                    "min_samples_split": sp_randint(1, 11),
                    "min_samples_leaf": sp_randint(1, 11),
                    "criterion": ["gini", "entropy"]}
    
        # run randomized search
        random_search = RandomizedSearchCV(models, param_distributions=param_dist,n_iter=n_iter,refit=True)
        random_search.fit(X, Y)
        report(random_search.cv_results_)
        pRF=random_search.best_estimator_
    feature_importances=pd.Series(pRF.feature_importances_,index=X.columns)
    std = np.std([tree.feature_importances_ for tree in pRF.estimators_],axis=0) # calculate tree errors
    feats=np.vstack((X.columns,pRF.feature_importances_))
    idx = np.argsort(feats[1])
    feats=feats[:,idx]
    var=feats[0,:] #sorted numvar by importance
    feature_importances.sort_values(inplace=True)
    print predictor, pRF.oob_score_, "Out-of-bag SCORE optimized RF system"
    #print "Sorted all features"
    for i in (-5,-4,-3,-2,-1):
        print feature_importances[i],var[i]
    #print feats[-5:]
    print "Average feature importance for top 5", np.mean(feats[1][-5:])  #feats[1][i]
    return pRF,feats

#optimise the parameters of gradient boosted classifier, with random searcha and cross-validation  
def gbc_params(models,n_iter,X,Y,typ): 
    if typ=='classifier':
        print "GBC param search..."
        # specify parameters and distributions to sample from
        param_dist = {"learning_rate": [0.000000000000000000001,0.0000000001,0.000001,0.0001,0.01,0.1,1.0,10.0],
                    "max_depth": sp_randint(1, 11),
                    "min_samples_split": [0.02,0.05,0.1,0.2,0.4,0.5],
                    "min_impurity_split": [0.00000001,0.00001,0.0001,0.001,0.1],
                    "max_features": sp_randint(1, len(X.columns)),
                    "loss": ["deviance","exponential"]
                    
                    }
        
        # run randomized search
        random_search = RandomizedSearchCV(models, param_distributions=param_dist,n_iter=n_iter,refit=True)
        random_search.fit(X, Y)
        report(random_search.cv_results_)
        pRF=random_search.best_estimator_
    if typ=='regressor':
        print "GBR param search..."
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                    "max_features": sp_randint(1, len(X.columns)),
                    "min_samples_split": sp_randint(1, 11),
                    "min_samples_leaf": sp_randint(1, 11),
                    "criterion": ["gini", "entropy"]}
    
        # run randomized search
        random_search = RandomizedSearchCV(models, param_distributions=param_dist,n_iter=n_iter,refit=True)
        random_search.fit(X, Y)
        report(random_search.cv_results_)
        pRF=random_search.best_estimator_
    feature_importances=pd.Series(pRF.feature_importances_,index=X.columns)
    feats=np.vstack((X.columns,pRF.feature_importances_))
    idx = np.argsort(feats[1])
    feats=feats[:,idx]
    var=feats[0,:] #sorted numvar by importance
    feature_importances.sort_values(inplace=True)
    #print "Sorted all features"
    for i in (-5,-4,-3,-2,-1):
        print feature_importances[i],var[i]
    #print feats[-5:]
    print "Average feature importance for top 5", np.mean(feats[1][-5:])  #feats[1][i]
    return pRF,feats
    
#optimise the parameters of ridge, with random search and cross-validation          
def rid_params(models,n_iter,X,Y,typ):  
    if typ=='classifier':
        print "RIDGE C param search ... "
        # specify parameters and distributions to sample from
        param_dist = {"alpha": [1,5,20,50.],
                    "fit_intercept": [True, False],
                    "normalize": [True, False],
                    "solver": ["auto", "svd", "cholesky","lsqr","sparse_cg","sag"]
                    
                    
                    }
        
        # run randomized search
        random_search = RandomizedSearchCV(models, param_distributions=param_dist,n_iter=n_iter,refit=True)
        random_search.fit(X, Y)
        report(random_search.cv_results_)
        prid=random_search.best_estimator_
    if typ=='regressor':
        print "Ridge Reg param search ..."
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                    "max_features": sp_randint(1, len(X.columns)),
                    "min_samples_split": sp_randint(1, 11),
                    "min_samples_leaf": sp_randint(1, 11),
                    "criterion": ["gini", "entropy"],
                    "tol":[0.0001,0.001,0.1,1.0, None]
                    
                    }
    
        # run randomized search
        random_search = RandomizedSearchCV(models, param_distributions=param_dist,n_iter=n_iter,refit=True)
        random_search.fit(X, Y)
        #report(random_search.cv_results_)
        prid=random_search.best_estimator_
    cofs=np.abs(prid.coef_) # might need [0] if giving 1D error
    cofs=np.absolute(cofs)
    cofs=list(cofs[0])      
    coeffs=pd.Series(cofs,index=X.columns)
    ridge_feat=np.vstack((X.columns,cofs))
    idx = np.argsort(ridge_feat[1]) 
    ridge_feat=ridge_feat[:,idx]
    prid_scor=prid.score(X,Y)
    print predictor, prid_scor, "Ridge R2 after optimization"
    return prid,ridge_feat[0]


#Clustering algo linked to RF classifier - i.e. look at sub-populations in group and plot predictors
def cluster_RF(datax,no_clust,ynom,numfeatures,x123): #datax must include target variable
    kmeans = KMeans(n_clusters=no_clust,random_state=42) 
    
    ypredicting=datax.pop(ynom) 
    clusters = kmeans.fit(datax)
    datax['cluster'] = pd.Series(clusters.labels_, index=datax.index)
    dataxc=datax.copy()
    dataxc=pd.concat([dataxc,ypredicting],axis=1)
    clust_datax=dataxc["cluster"].value_counts()
    remove=clust_datax[clust_datax <=10].index
    dataxc["cluster"].replace(remove, np.nan,inplace=True)
    datax.drop('cluster',inplace=True,axis=1) 
    print "####################################"
    for ic in dataxc["cluster"].value_counts().index:
        print "------- cluster %0.f %% ---------" %(((float(dataxc[dataxc['cluster']==ic]["Age"].count())/float(dataxc["Age"].count()))*100.))
        x_pred = dataxc[dataxc["cluster"] == ic]             
        ypred=x_pred.pop(ynom) 
        x_pred.drop("cluster",axis=1,inplace=True)                 
        RF_full,RF_full_score_,RF_full_feats = RF_type(x_pred,ypred)
        x_pred=x_pred[RF_full_feats[0][-4:]] #INPUT APPROP FEATURES         
        RFslim=RandomForestClassifier(n_estimators=500, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)      # RFslim takes in only the most important features as identified (in this case) through recursive feature extraction  
        RFslim.fit(x_pred,ypred)
        score2=RFslim.oob_score_
        
        
        """identify and sort most important features and errors"""
        feature_importances=pd.Series(RFslim.feature_importances_,index=x_pred.columns)
        std = np.std([tree.feature_importances_ for tree in RFslim.estimators_],axis=0) # calculate tree errors
        feats=np.vstack((x_pred.columns,RFslim.feature_importances_))
        idx = np.argsort(feats[1])
        feats=feats[:,idx]
        var=feats[0,:] #sorted numvar by importance
        feature_importances.sort_values(inplace=True)
        print "##################", predictor, RFslim.oob_score_, "Out-of-bag SCORE slim RF", "###################"
        for i in range(len(var)):
            print "Name: %s, val %.3f, mean cluster %.3f, mean all %.3f"   %(feats[0][i], feats[1][i], np.mean(x_pred[feats[0][i]]), np.mean(dataxc[feats[0][i]]) )
        print "Average feature importance for top 5 slim RF", np.mean(feats[1][-5:])  #feats[1][i]
        
        """PLOT FOREST"""
        fig=plt.figure("Reduced forest for %s cluster %s %.0f %%" %(predictor, ic, (float(dataxc[dataxc['cluster']==ic]["Age"].count())/float(dataxc["Age"].count()))*100.))
        ax=plt.subplot(111)
        barnames= var
        bar_height= (feature_importances*RFslim.oob_score_)*100.
        indes=range(len(barnames))
        bar_errors=(std*100.)/2.
        ax.bar(indes,bar_height,yerr=bar_errors)
        ax.set_xticklabels(barnames)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Predictive Power (%)")
        fig.suptitle("Reduced forest %s" %(predictor), fontsize=24)
        plt.xlabel('Features', fontsize=18)
        plt.ylabel('Real predictive value (%)', fontsize=16)
        ax.text(0.2, 0.9, "Total power %.0f " %(RFslim.oob_score_*100), horizontalalignment='center',verticalalignment='center',transform = ax.transAxes) 
        tempdat=pd.concat([x_pred,ypred],axis=1)
        p=sns.pairplot(data=tempdat,hue=predictor,palette=("red","green")) #plot pairwise relationships to interrogate underlying relationships
        p.map_upper(sns.regplot)
        p.fig.suptitle('%s cluster: %s %.0f %%' %(predictor, ic, (float(dataxc[dataxc['cluster']==ic]["Age"].count())/float(dataxc["Age"].count()))*100.), verticalalignment='top', fontsize=20)
    print dataxc['cluster'].value_counts().divide(dataxc['cluster'].count())*100.                         
                                                                 
                                                                                                 
                                                                                                                                 
                                                                                                                                                            
 
    
    
##########################################################################################            
                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
inny=pd.ExcelFile("cleaned_data.xlsx") # read in excel
df=inny.parse("Sheet1") #load sheet into dataframe
inny=pd.ExcelFile("cleaned_data.xlsx") # read in excel
df=inny.parse("Sheet1") #load sheet into dataframe

TNMdata=pd.ExcelFile("TNM_customers.xlsx")
usage_df=TNMdata.parse("Usages") # all usage data from TNM
recharge_df=TNMdata.parse("recharge_kw")

#a little data cleaning (most of this is done in cleaner.py prior to running this script)

#dropping super sparse data 
df.drop(["Point_(1-5)","Interviewer"],axis=1,inplace=True)
df.drop("Smart_phone_basic_phone_internet_enabled_2nd_phone",axis=1,inplace=True)
df.drop("did_you_buy_your_2nd_phone_new_or_used",axis=1,inplace=True)
df.drop("What_is_the_2nd_reason_you_use_your_phone",axis=1,inplace=True)


#drop features dependent on other features
df.drop(["What_is_the_brand_of_your_solar_product","What_mode_of_transport_do_you_use_to_purchase_light_Motorbike,_minibus,_bike,_walk,_car,_other","What_could_be_more_important_to_have_a_solarlight_or_phone_charger_if_same_price_","Where_did_you_buy_your_solar_product_shop_trading_center,_shop_city,_shop_in_village,_shop_trading_center,_friend,_NGO,_church","What_is_the_second_use_for_your_solar_product","What_is_the_primary_use_of_your_solar_product"],axis=1,inplace=True)
#df.drop(["Phone_number"], axis=1, inplace=True)


#initialise a few copies of data for different uses
df=df
train=df
dat=df  


#remove low-freq airtime spends (as likely being errors)
air=train["How_much_MK_do_you_spend_on_airtime_in_a_month"].value_counts()
remove=air[air <=2].index #remove anything that appears less than 2 times = 0.5% % of responses
train["How_much_MK_do_you_spend_on_airtime_in_a_month"].replace(remove, train["How_much_MK_do_you_spend_on_airtime_in_a_month"].median(skipna=True),inplace=True)

#binarise Sex
train["Sex"].replace('male', 1,inplace=True)
train["Sex"].replace('female', 0,inplace=True)

#fill blanks with appropriate response
train["do_you_own_any_solar_"].fillna('no',inplace=True)

#convert instances of internet phonet to smart phone (understood as the same thing by Malawians)
train["Smart_phone_basic_phone_internet_enabled_1st_phone"][train["Smart_phone_basic_phone_internet_enabled_1st_phone"]=='internetenabled']='smartphone'

#################

dat=train  

catvar=list(train.dtypes[train.dtypes == "object"].index) #list of cat columns
numvar=list(train.dtypes[train.dtypes != "object"].index) #list of numeric columns


print "##########################################"

#Simple plots to look at relationships in data
if 1==0: #switch for on or off
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ###########Simple plots
    #1st phone type and gift or not
    phones=train.groupby(['Smart_phone_basic_phone_internet_enabled_1st_phone', 'How_did_you_get_your_first_phone_(gift_purchased)'])['Smart_phone_basic_phone_internet_enabled_1st_phone'].count().unstack('How_did_you_get_your_first_phone_(gift_purchased)')
    phones=phones.divide(train['Smart_phone_basic_phone_internet_enabled_1st_phone'].count())*100
    ax=phones.plot(kind='bar', stacked=True, ax=axes[0,0])
    ax.set_xlabel("Phone type")
    ax.set_ylabel("Ownership (%)")
    ax.set_ylim([0,100])
    
    #ever gifted phone
    phones=train.groupby(['Have_you_ever_given_a_phone_as_a_gift_Yes_no', 'If_yes,_to_whom'])['Have_you_ever_given_a_phone_as_a_gift_Yes_no'].count().unstack('If_yes,_to_whom')
    phones=phones.divide(train['Have_you_ever_given_a_phone_as_a_gift_Yes_no'].count())*100
    ax=phones.plot(kind='bar', stacked=True, ax=axes[1,1])
    ax.set_xlabel("Gifted phone")
    ax.set_ylabel("Ownership (%)")
    ax.set_ylim([0,100])
    
    newphones=train.groupby(["Why_did_you_buy_this_phone_theftupgradebattery_died_cell_died_first_phone_other"])
    #newphones=train["Why_did_you_buy_this_phone_theftupgradebattery_died_cell_phone_died_first_phone_other"]
    newphones=train["Why_did_you_buy_this_phone_theftupgradebattery_died_cell_died_first_phone_other"].value_counts().divide(train["Why_did_you_buy_this_phone_theftupgradebattery_died_cell_died_first_phone_other"].count())*100
    ax2=newphones.plot(kind='bar', ax=axes[0,1])
    ax2.set_xlabel("Reasons for buying latest phone")
    ax2.set_ylabel("%")
    ax.set_ylim([0,100])
    
    fig2, axes = plt.subplots(nrows=2, ncols=3)
    #solar ownership and brands
    solar=train.groupby(["do_you_own_any_solar_", 'If_yes,_is_it_(solartorch,_5-10w,_10-25w,_25-50,_50-100w)'])['do_you_own_any_solar_'].count().unstack('If_yes,_is_it_(solartorch,_5-10w,_10-25w,_25-50,_50-100w)')
    solar=solar.divide(train['do_you_own_any_solar_'].count())*100
    ax=solar.plot(kind='bar', stacked=True, ax=axes[0,0])
    ax.set_xlabel("Solar ownership")
    ax.set_ylabel("Ownership (%)")
    ax.set_ylim([0,100])
    
    brand=train["What_is_the_brand_of_your_solar_product"].value_counts()
    remove=brand[brand <=5].index
    train["What_is_the_brand_of_your_solar_product"].replace(remove, np.nan,inplace=True)
    quality=train.groupby(["What_is_the_brand_of_your_solar_product", 'do_you_think_solar_is_good_quality_Yesno'])["What_is_the_brand_of_your_solar_product"].count().unstack('do_you_think_solar_is_good_quality_Yesno')
    #quality=quality.divide(train["do_you_think_solar_is_good_quality_Yesno"].count())*100
    ax=quality.plot(kind='bar', stacked=True, ax=axes[0,1])
    ax.set_xlabel("Solar brand")
    ax.set_ylabel("Ownership (%)")
    
    ax=(train["What_is_your_source_of_light_electricity,_battery_torch,_candle,_paraffin,_solar_light,_cell_phone,_other"].value_counts().divide(train["What_is_your_source_of_light_electricity,_battery_torch,_candle,_paraffin,_solar_light,_cell_phone,_other"].count())*100).plot(kind='bar',ax=axes[1,0])
    ax.set_xlabel("Light source")
    ax.set_ylabel("Populartity %")
    
    #solar quality by brokenness
    phones=train.groupby(['do_you_think_solar_is_good_quality_Yesno', 'do_you_own_or_do_you_know_someone_who_owns_a_broken_solar_product'])['do_you_think_solar_is_good_quality_Yesno'].count().unstack('do_you_own_or_do_you_know_someone_who_owns_a_broken_solar_product')
    phones=phones.divide(train['do_you_think_solar_is_good_quality_Yesno'].count())*100
    ax=phones.plot(kind='bar', stacked=True, ax=axes[1,1])
    ax.set_xlabel("Phone type")
    ax.set_ylabel("Ownership (%)")
    
    #Change source of light
    phones=train.groupby(['Are_you_satisfied_with_the_source_of_light_or_would_you_like_to_change_Satisfied_Change', 'Why_Change_light'])['Are_you_satisfied_with_the_source_of_light_or_would_you_like_to_change_Satisfied_Change'].count().unstack('Why_Change_light')
    phones=phones.divide(train['Are_you_satisfied_with_the_source_of_light_or_would_you_like_to_change_Satisfied_Change'].count())*100
    ax=phones.plot(kind='bar', stacked=True, ax=axes[0,2])
    ax.set_xlabel("Satisfaction with light")
    ax.set_ylabel(" (%)")
    
    #histo of spend on lighting
    ax=train["How_much_do_you_spend_on_light_in_a_month"].plot(kind='hist', bins=50,bottom=0.1,ax=axes[1,2])
    ax.set_xlabel("Monthly light spend")
    ax.set_ylabel("Frequency")
    #ax.set_xscale('log')
    ax.set_xlim([0,4000])

"""Fancy Part"""

#Dummify categorical variables
for varx in catvar:
    dummies=pd.get_dummies(dat[varx],prefix=varx)
    dat=pd.concat([dat,dummies],axis=1)
    dat.drop([varx],axis=1,inplace=True) 
    

#Impute  missing data in most conservative way (i.e. just labelled as missing - this may or may not cause headaches later, but is the imputation with least assumptions about underying data)
for varx in catvar: # fillNA with most common value
    train[varx].fillna('Missing',inplace=True)
    
for i in numvar[2:]: #fill in missing numerical values with means
    train[i][np.isnan(train[i])] = np.nanmean(train[i])
 
    
""" create sub dataframes of people of interest for either T-testing or ML on sub-samps"""
print "########### SUB DF ##########################"

solar_owners = dat[dat["do_you_own_any_solar__yes"] == 1.0] 
solar_losers = dat[dat["do_you_own_any_solar__no"] == 1.0]

rural = dat[dat["Rural_peri-urban_rural"] == 1.0]
urban = dat[dat["Rural_peri-urban_peri-urban"] == 1.0]
geo=sim=pd.concat([rural,urban])

for i in geo.columns:
    print i
sys.exit()


""" This part is to look at technology adoption """
gift_givers = dat[dat["Have_you_ever_given_a_phone_as_a_gift_Yes_no_yes"] == 1.0]
gift_losers = dat[dat["Have_you_ever_given_a_phone_as_a_gift_Yes_no_no"] == 1.0]

gift_rec = dat[dat["How_did_you_get_your_first_phone_(gift_purchased)_gift"] == 1.0]
gift_nonrec = dat[dat["How_did_you_get_your_first_phone_(gift_purchased)_purchased"] == 1.0]
"""   """


solar_light= dat[dat["What_is_your_source_of_light_electricity,_battery_torch,_candle,_paraffin,_solar_light,_cell_phone,_other_solarlight"]==1.0]

solar_owners = dat[dat["do_you_own_any_solar__yes"] == 1.0] 
solar_losers = dat[dat["do_you_own_any_solar__no"] == 1.0]

boost_yes = dat[dat["do_you_use_a_booster_charge_(15_minutes_to_charge_your_phone)_no"] == 1.0] 
boost_no = dat[dat["do_you_use_a_booster_charge_(15_minutes_to_charge_your_phone)_yes"] == 1.0]

solar_qual=pd.concat([ dat[dat["do_you_think_solar_is_good_quality_Yesno_yes"] == 1.0], dat[dat["do_you_think_solar_is_good_quality_Yesno_no"] == 1.0]]) # only people who answered question - no blanks

solar_yes = solar_owners[solar_owners["do_you_think_solar_is_good_quality_Yesno_yes"] == 1.0] 
solar_no= solar_owners[solar_owners["do_you_think_solar_is_good_quality_Yesno_no"] == 1.0]


TNM = dat[dat["Which_network_SIM_card_do_you_have_tnm"] == 1.0] 
Airtel = dat[dat["Which_network_SIM_card_do_you_have_airtel"] == 1.0]
sim=pd.concat([TNM,Airtel])

Mal=dat[dat["Sex"] == 1.0]
Fem=dat[dat["Sex"] == 0.0] 



for i in dat.columns:
    print "%s: %s +- %s" %(i, dat[i].mean(), dat[i].sem())

#Sub sub data frames 

TNMsolar_owners = TNM[TNM["do_you_own_any_solar__yes"] == 1.0]
TNMsolar_losers = TNM[TNM["do_you_own_any_solar__no"] == 1.0]

#Simple T-test plots for data exploration: test binary combinations and plot anything which shows up with P-value of <0.05  . 
if 1==1:
    for t in gift_rec.columns:
        A=gift_rec
        B=gift_nonrec
        #if stats.ttest_ind(gift_rec[i],gift_nonrec[i])[1] <= 0.1:
        print t,"A %.2f, B  %.2f " %(A[i].mean(), B[i].mean())
        print "======================"
        barnames="Gift rec", "non"
        plt.figure(t)
        ax=plt.subplot(111)
        height=np.mean(A[t]),np.mean(B[t])
        yerr=stats.sem(A[t],nan_policy='omit'),stats.sem(B[t],nan_policy='omit')
        ax.bar(range(2),height,yerr=yerr)
        ax.set_xticklabels(barnames)
plt.show()
sys.exit()     
if 1==0:
    for t in dat.columns.values:
        if stats.ttest_ind(solar_yes[t],solar_no[t] )[1] < 0.05:
            print t, stats.ttest_ind(solar_yes[t],solar_no[t] ), np.mean(solar_yes[t]), np.mean(solar_no[t]), "Good quality solar, bad quality"
            #plt.figure(t)
            #ax=plt.subplot(111)
            #height=np.mean(solar_yes[t]),np.mean(solar_no[t])
            #yerr=stats.sem(solar_yes[t]),stats.sem(solar_no[t])
            #ax.bar(range(2),height,yerr=yerr)
            #barnames="Good quality solar", "bad quality"
            #ax.set_xticklabels(barnames)

    
if 1==0:    
    print "---------------------"
    for t in dat.columns.values:
        if stats.ttest_ind(gift_givers[t],gift_losers[t] )[1] < 0.05:
            print t, stats.ttest_ind(gift_givers[t],gift_losers[t] ), np.mean(gift_givers[t]), np.mean(gift_losers[t]), "Gifters, non-gifters"
            #plt.figure("%s" %(t))
            #A=gift_givers
            #B=gift_losers
            #barnames="Gift givers", "non-gifters"
            #ax=plt.subplot(111)
            #height=np.mean(A[t]),np.mean(B[t])
            #yerr=stats.sem(A[t]),stats.sem(B[t])
            #ax.bar(range(2),height,yerr=yerr)
            #ax.set_xticklabels(barnames)
            

    print "---------------------"
    for t in dat.columns.values:
        print "Urban v Rural"
        A=urban
        B=rural
        if stats.ttest_ind(A[t],B[t] )[1] < 0.05:
            print t, stats.ttest_ind(A[t],B[t] ), np.mean(A[t]), np.mean(B[t]), "urban rural"
            #plt.figure("%s" %(t))
            #barnames="urban", "rural"
            #ax=plt.subplot(111)
            #height=np.mean(A[t]),np.mean(B[t])
            #yerr=stats.sem(A[t]),stats.sem(B[t])
            #ax.bar(range(2),height,yerr=yerr)
            #ax.set_xticklabels(barnames)    

            print "---------------------"
      
            
    print "---------------------"
    #find t-test wehre p<0.05
    for t in dat.columns.values:
        if stats.ttest_ind(TNM[t],Airtel[t] )[1] < 0.05:
            print t, stats.ttest_ind(TNM[t],Airtel[t] ), np.mean(TNM[t]), np.mean(Airtel[t]), "TNM, AIRTEL"
    
    
    print "---------MF------------"
    for t in dat.columns.values:
        if stats.ttest_ind(Mal[t],Fem[t] )[1] < 0.05:
            print t, stats.ttest_ind(Mal[t],Fem[t] ), np.mean(Mal[t]), np.mean(Fem[t]), "M, F"
            plt.figure("%s" %(t))
            #A=Mal
            #B=Fem
            #barnames="Male", "Female"
            #ax=plt.subplot(111)
            #height=np.mean(A[t]),np.mean(B[t])
            #yerr=stats.sem(A[t]),stats.sem(B[t])
            #ax.bar(range(2),height,yerr=yerr)
            #ax.set_xticklabels(barnames) 
    
    
    print "---------------------"
    for t in dat.columns.values:
        if stats.ttest_ind(solar_owners[t],solar_losers[t] )[1] < 0.05:
            print t, stats.ttest_ind(solar_owners[t],solar_losers[t] ), np.mean(solar_owners[t]), np.mean(solar_losers[t]), "solar_owners, solar_losers"
    
            
    print "---------------------"
    for t in dat.columns.values:
        if stats.ttest_ind(solar_yes[t],solar_no[t] )[1] < 0.05:
            print t, stats.ttest_ind(solar_yes[t],solar_no[t] ), np.mean(solar_yes[t]), np.mean(solar_no[t]), "solar good quality, solar poor"
    


############################
#test=train

test=dat

niters_para=25  # number of iterations to optimise algos 
if 1==0: #switch for if using telecom data
    ixs="real_airtime","reported_monthly_income"   
    for i in ixs:
        test[i].replace(np.nan, np.nanmean(test[i]),inplace=True)
else:       
    test.drop(["real_airtime","reported_monthly_income"],axis=1,inplace=True)





########################################################


#drop uneeded
test.drop(["Phone_number","Why_Change_light_tooexpensive","If_yes,_to_whom_family","If_yes,_to_whom_friend","If_yes_from_whom_family","If_yes_from_whom_friend"], axis=1,inplace=True) # drop related variable
test.drop("How_much_MK_do_you_spend_on_airtime_in_a_month",axis=1,inplace=True)
test.drop(["How_many_times_a_month_do_you_purchase_light","Has_anyone_ever_given_you_a_phone_as_a_gift_yes","If_yes,_to_whom_business","Has_anyone_ever_given_you_a_phone_as_a_gift_no"],axis=1,inplace=True)
test.drop(["district_ntchisi","district_nkhotakota","district_dowa","If_you_had_an_internet_phone,_what_would_you_use_it_for_dontknow","Smart_phone_basic_phone_internet_enabled_1st_phone_basicphone"],axis=1,inplace=True)
test.drop(["Is_your_solar_sold_piece_by_piece_yes","Is_your_solar_sold_piece_by_piece_no","Where_do_you_charge_your_phone_(home,_neighbor,_friend,_barbershop,_shop,_work,_other)_barbershop"],axis=1,inplace=True)
test.drop(["If_yes,_is_it_(solartorch,_5-10w,_10-25w,_25-50,_50-100w)","What_is_your_source_of_light_electricity,_battery_torch,_candle,_paraffin,_solar_light,_cell_phone,_other_solarlight","If_not_home,_how_many_minutes_walk_from_your_house_(one_way)_to_charge_phone","What_is_your_2nd_source_of_light_electricity,_battery_torch,_candle,_paraffin,_solar_light,_cell_phone,_other_solarlight"],axis=1,inplace=True)







#drop anything with very sparse answers (i.e. less that 2.5% response rate)
for tx in test.columns:
    if test[tx].mean()< 0.025:
        print tx
        test.drop(tx,axis=1,inplace=True)


############################

########CREATE TRAIN AND TEST SETS 80/20 #####################
test, holdout=train_test_split(test,test_size=0.15)

#Keep an original
testx=test





print "####### MACHINE LEARNING PART ##########"




predictor="Which_network_SIM_card_do_you_have_tnm" #var to predict
olsvar=test[predictor]
testori=test
#pop off data for training, and also from holdout
y_predict=test.pop(predictor)
y_holdout=holdout.pop(predictor)

test.drop("Which_network_SIM_card_do_you_have_airtel",axis=1,inplace=True) #drop mutually exclusive category




print "##############  FEATURE SELECTION  ######################"


print "%s Starting # features" %(len(test.columns) )


#possible variance threshold to employ
"""
sel = VarianceThreshold()
sel.fit_transform(test)
print "Variances"
print sel.variances_
print "###"
"""



RF_full,RF_full_score_,RF_full_feats,typ = RF_type(test,y_predict)




#Feature selection using select K best and Chi2 to see which features most accurately capture the data

feature_limit=0.2 # percent of feats t keep
k_number=int((len(testx.columns)*feature_limit))
Kbest=SelectKBest(chi2, k=k_number)
Kbest.fit_transform(test, y_predict)
kfeats=np.vstack((testx.columns,Kbest.scores_))
#sort by most important
idx = np.argsort(kfeats[1])
kfeats=kfeats[:,idx]
selectedk=[]
for i in range(len(testx.columns)):
    if np.isnan(kfeats[1][i])==False:
        print kfeats[0][i], kfeats[1][i]
        selectedk.append(kfeats[0][i])
K_number=float(len(testx.columns)-k_number)-float(len(testx.columns))
k_number=np.negative(k_number)
selected_by_k= kfeats[0][k_number:] 
selectedk=selectedk[k_number:]


#Make test only contain features selected by Kbest scoring
test=test[selected_by_k]

#Feature selection using recursive feature elimination (generates plots only)
OOB_vs_feats(test,y_predict,RF_full_feats,typ) 


#Run RF to find type of problem (classification/regression)
RF_full,RF_full_score_,RF_full_feats,typ = RF_type(test,y_predict)
#optimise parameters of RF
RF_full, RF_full_feats=RF_params(RF_full,niters_para,test,y_predict,typ)


 
""" SLIM LINE ML"""

#Various things to play with for sanity e.g. how does randomly shuffling and selecting features affect OOB score stability?
#shuffled_feats=shuffle_test(testx,y_predict,typ,25)
#slimjim=test[shuffled_feats] #INPUT APPROP FEATURES


#Select only top 5 features to keep in data after SelectKbest and recursive feature extraction
test=test[RF_full_feats[0][-5:]] 

""" Recap: Feature selection through a combo of recursive feature extraction, variance threshhold and Kbest feature extraction has yielded slimmed down data set (recast as test), we still have testx which contains all features for use in other ML methods"""

print "####### Model testing #############"

if typ=='regressor':
    RFslim=RandomForestRegressor(n_estimators=1000, oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)    
elif typ=='classifier':
    RFslim=RandomForestClassifier(n_estimators=1000,oob_score=True,n_jobs=-1,bootstrap=True,random_state=42)
else:
    print "Unknown RF typ (Reg/Class)"
    
#optimise RF on data with minimal features
RFslimp, RF_slim_feats=RF_params(RFslim,niters_para,test,y_predict,typ)

score2=RFslimp.oob_score_ #store OOB score for later

#identify and sort most important features and errors
feature_importances=pd.Series(RFslimp.feature_importances_,index=test.columns)
std = np.std([tree.feature_importances_ for tree in RFslimp.estimators_],axis=0) # calculate tree errors
feature_importances.sort_values(inplace=True)



"""PLOT FOREST"""
fig=plt.figure("Reduced forest %s" %(predictor))
ax=plt.subplot(111)
barnames= RF_slim_feats[0] 
CV=np.nanmean(cross_validation.cross_val_score(RFslimp, holdout, y_holdout , cv=8))
bar_height= (feature_importances*CV)*100.
indes=range(len(barnames))
bar_errors=(std*100.)/2.
ax.bar(indes,bar_height,yerr=bar_errors)
ax.set_xticklabels(barnames)
ax.set_xlabel("Feature")
ax.set_ylabel("Predictive Power (%)")
fig.suptitle("Reduced forest %s" %(predictor), fontsize=24)
plt.xlabel('Features', fontsize=18)
plt.ylabel('Real predictive value (%)', fontsize=16)

ax.text(0.2, 0.9, "Total power %f " %(score2*100.), horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)

""" """
#plots#


#manually enter any appropriate sub data frames for generating crude (but easy to communicate) comparisons
t1=solar_owners
t2=solar_losers
tnoms="Solar","no solar"

#feature plots
plt.figure("Factor analysis - histos of %s" %(predictor))
try:
    ax=plt.subplot(421)
    d=t2[test.columns[0]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="black",normed=True)
    
    d=t1[test.columns[0]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="green",normed=True)
    #ax.set_xlim([0,10000])
    ax.text(0.9, 0.2, "%s" %(test.columns[0]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xscale('log')
    ax.legend(tnoms)
except:
    print "plot error, continuing"
    
try:
    ax=plt.subplot(422)
    d=t2[test.columns[4]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="black",normed=True)

    d=t1[test.columns[4]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="green",normed=True)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
    ax.text(0.9, 0.2, "%s" %(test.columns[4]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(423)
    d=t2[test.columns[1]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="black",normed=True)

    d=t1[test.columns[1]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="green",normed=True)
    ax.text(0.9, 0.2, "%s" %(test.columns[1]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(424)
    d=t2[test.columns[2]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="black",normed=True)

    d=t1[test.columns[2]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="green",normed=True)
    ax.text(0.9, 0.2, "%s" %(test.columns[2]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(425)
    d=t2[test.columns[3]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="black",normed=True)

    d=t1[test.columns[3]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="green",normed=True)
    ax.text(0.9, 0.2, "%s" %(test.columns[3]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(426)
    d=t2[test.columns[5]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15,facecolor= "black",normed=True)

    d=t1[test.columns[5]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="green",normed=True)
    ax.text(0.9, 0.2, "%s" %(test.columns[5]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(428)
    d=t2[test.columns[6]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(norm, 15, facecolor="black",normed=True)
    
    d=t1[test.columns[6]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.hist(d, 15,facecolor= "green",normed=True)
    ax.text(0.9, 0.2, "%s" %(test.columns[6]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
    

plt.figure("Factor analysis - curves of %s" %(predictor))
try:
    ax=plt.subplot(421)
    d=t2[test.columns[0]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")
    
    d=t1[test.columns[0]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    #ax.set_xlim([0,10000])
    ax.text(0.9, 0.2, "%s" %(test.columns[0]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xscale('log')
    ax.legend(tnoms)
except:
    print "plot error, continuing"
    
try:
    ax=plt.subplot(422)
    d=t2[test.columns[4]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")

    d=t1[test.columns[4]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
    ax.text(0.9, 0.2, "%s" %(test.columns[4]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(423)
    d=t2[test.columns[1]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")

    d=t1[test.columns[1]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    ax.text(0.9, 0.2, "%s" %(test.columns[1]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(424)
    d=t2[test.columns[2]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")

    d=t1[test.columns[2]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    ax.text(0.9, 0.2, "%s" %(test.columns[2]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(425)
    d=t2[test.columns[3]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")

    d=t1[test.columns[3]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    ax.text(0.9, 0.2, "%s" %(test.columns[3]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(426)
    d=t2[test.columns[5]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")

    d=t1[test.columns[5]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    ax.text(0.9, 0.2, "%s" %(test.columns[5]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(428)
    d=t2[test.columns[6]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="black")
    
    d=t1[test.columns[6]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    ax.plot(x_fit, norm, color="green")
    ax.text(0.9, 0.2, "%s" %(test.columns[6]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "plot error, continuing"
        
plt.figure("Factor analysis of %s" %(predictor))
try:
    ax=plt.subplot(421)
    d=t2[test.columns[0]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')

    d=t1[test.columns[0]]
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[0]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    ax.legend(tnoms)
except:
    print "plot error, continuing"
    
try:
    ax=plt.subplot(422)
    d=t2[test.columns[4]]
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')

    d=t1[test.columns[4]]
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[4]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
except:
    print "plot error, continuing"
try:
    ax=plt.subplot(423)
    d=t2[test.columns[1]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')
    d=t1[test.columns[1]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[1]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "bar error, continuing"
try:
    ax=plt.subplot(424)
    d=t2[test.columns[2]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')
    d=t1[test.columns[2]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[2]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "bar error, continuing"
try:
    ax=plt.subplot(425)
    d=t2[test.columns[3]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')
    d=t1[test.columns[3]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[3]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "bar error, continuing"
try:
    ax=plt.subplot(426)
    d=t2[test.columns[5]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')
    d=t1[test.columns[5]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[5]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "bar error, continuing"
try:
    ax=plt.subplot(428)
    d=t2[test.columns[6]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(1,d.mean(),yerr=yer, color='black')
    d=t1[test.columns[6]]
    mu, std = stats.norm.fit(d)
    x_fit=np.linspace(np.min(d),np.max(d),d.shape[0])
    norm=stats.norm.pdf(x_fit, mu, std)
    yer=d.sem()
    ax.bar(2,d.mean(),yerr=yer, color='green')
    ax.text(0.9, 0.2, "%s" %(test.columns[6]),horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    #ax.set_xlim([0,15000])
    #ax.set_xscale('log')
except:
    print "bar error, continuing"
    

####

#Try some other ML methods to compare accuracy


""" TRY OTHER ML and cross-val with holdout"""
if typ=='classifier':
    print "SVC"
    #Need to add some optimisation and feature selection around these - work in progress
    model_svm=SVC(penalty='l2', loss='squared_hinge', dual=True,tol=1e-4) #L2 penalised support vector machine
    model_svm.fit(testx,y_predict)
    svm_score= model_svm.score(testx,y_predict)   
    print "RIDGE C"
    rid=lm.RidgeClassifier(max_iter=1000)
    rid, rid_feats_=rid_params(rid,niters_para,testx,y_predict,typ) # optimise parameters for Ridge (L1 optimisation)
    slimrid,slimrid_feats=rid_params(rid,niters_para,test,y_predict,typ) #also try a Ridge classifier with 'slim' data (i.e. after feature extraction)
    rid_score=rid.score(testx,y_predict)
    print "GBC"
    gbc=GBC(n_estimators=110, random_state=41)
    gbc, gbcfeats=gbc_params(gbc,niters_para,test,y_predict,typ) # optimise parameters for Gradient boosted
    gbc.fit(test, y_predict)
    gbc_scores=np.nanmean(cross_validation.cross_val_score(gbc, holdout, y_holdout , cv=10))


            
elif typ=='regressor':
    #work in progress
    model_svm=SVR()
    model_svm.fit(testx,y_predict)
    svm_score= model_svm.score(testx,y_predict)   
    rid=lm.Ridge(alpha=0.1, fit_intercept=True,max_iter=10000, tol=0.0001,solver='auto')
    rid.fit(testx,y_predict)
    rid_score=rid.score(testx,y_predict)

#Plot Ridge results
Ridge_a_plot(testx,y_predict,'classifier',20)

#sort Ridge results
cofs=np.abs(rid.coef_) # might need [0] if giving 1D error
cofs=np.absolute(cofs)
cofs=list(cofs[0])
coeffs=pd.Series(cofs,index=testx.columns)
ridge_feat=np.vstack((testx.columns,cofs))
idx = np.argsort(ridge_feat[1])
ridge_feat=ridge_feat[:,idx]

slimcofs=np.abs(slimrid.coef_) # might need [0] if giving 1D error
slimcofs=np.absolute(slimcofs)
slimcofs=list(slimcofs[0])

slimcoeffs=pd.Series(slimcofs,index=test.columns)
slimridge_feat=np.vstack((test.columns,slimcofs))
idx = np.argsort(slimridge_feat[1])
slimridge_feat=slimridge_feat[:,idx]


#Some cross val scores
Ridge_scores=cross_validation.cross_val_score(rid, holdout, y_holdout , cv=10)
slimrid_CV=np.nanmean(cross_validation.cross_val_score(slimrid, holdout, y_holdout , cv=10))

#GBC plots
plt.figure("GBC feature importance")
plt.suptitle("GBC importances %s" %(predictor))
heights=gbcfeats[1][-5:]
barnom=gbcfeats[0][-5:]
ax=plt.subplot(111)
#ax.suptitle("Ridge ML coefficients")
ax.bar(range(len(heights)),heights, color='green')
ax.set_xticklabels(barnom)
ax.set_xlabel("Feature")
ax.set_ylabel("Predictive Power (%)")
ax.text(0.2, 0.9, "Total power %.0f " %(np.nanmean(gbc_scores*100)), horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)



#Some summaries for inspection
print "##########################################################################"
print "          ######## RFslim feature importances #########                    "
for i in (-5,-4,-3,-2,-1):
    print RF_slim_feats[0][i], RF_slim_feats[1][i]
print np.nanmean(CV), predictor,  "RFslim CV score"


print "##########################################################################"
print "          ######## GBC feature importances #########                    "
for i in (-5,-4,-3,-2,-1):
    print gbcfeats[0][i], gbcfeats[1][i]
print np.nanmean(gbc_scores), predictor,  "GBC CV score"


print "##########################################################################"
print "          ######## Ridge feature importances #########                    "
for i in (-5,-4,-3,-2,-1):
    print ridge_feat[0][i], ridge_feat[1][i]
print np.nanmean(Ridge_scores), predictor,  "Ridge CV score"
     
                         
co=np.abs(model_svm.coef_) # might need [0] if giving 1D error
co=np.absolute(co)
co=list(co[0])

co_svm=pd.Series(co,index=testx.columns)
svm_feat=np.vstack((testx.columns,co))
idx = np.argsort(svm_feat[1])
svm_feat=svm_feat[:,idx]
svm_var=svm_feat[0,:] #sorted numvar by importance
#ridge_feat.sort_values(inplace=True)

print "##########################################################################"
print "          ######## SVC/SVM feature importances #########   "
print predictor, svm_score, "svm R2"
for i in (-5,-4,-3,-2,-1):
    print svm_feat[0][i], svm_feat[1][i]
    






#Some more cross-val scores for inspection

RF_scores=cross_validation.cross_val_score(RFslimp, holdout, y_holdout , cv=10)
svm_scores=cross_validation.cross_val_score(model_svm, holdout, y_holdout , cv=10)
RF_full_holdscore=cross_validation.cross_val_score(RF_full, holdout, y_holdout , cv=10)

#plotting Ridge results
plt.figure("Ridge coeffs")
plt.suptitle("Ridge normalised coeffs")
maxval=np.sum(ridge_feat[1][:])
top_rid=ridge_feat[1][-5:]
heights=((np.divide(top_rid,maxval))*100.)*np.nanmean(Ridge_scores)

barnom=ridge_feat[0][:]
ax=plt.subplot(111)
#ax.suptitle("Ridge ML coefficients")
ax.bar(range(len(heights)),heights, color='red')
ax.set_xticklabels(barnom[-5:])
ax.set_xlabel("Feature")
ax.set_ylabel("Predictive Power (%)")
ax.text(0.2, 0.9, "Total power %.0f " %(np.nanmean(Ridge_scores*100)), horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)




#Some simple decision trees (mostly for fun - these overfit terribly, as expected)
print "#############DT features ##################"
clf = DecisionTreeRegressor(max_depth=2)
clf.fit(test, y_predict)
print "###################"
tree_importances=pd.Series(clf.feature_importances_,index=test.columns)
tree_importances.sort_values(inplace=True)
print tree_importances[-5:]
print clf.score(test,y_predict), "Tree score"
print "#################################"

#Some simple linear regressions - consider this a diagnostic baseline - I expect ML to always do better than this, if ML doesnt then there are issues somewhere
Reg=LR(fit_intercept=True, normalize=True,  n_jobs=-1)
Reg.fit(testx,y_predict)
coeffs=pd.Series(Reg.coef_,index=testx.columns)
coeffs=np.absolute(coeffs)
coeffs.sort_values(inplace=True)
print "Top LR full coefficients..."
print coeffs[-5:]

LRfull_CV=cross_validation.cross_val_score(Reg, holdout, y_holdout , cv=15)
score3=Reg.score(testx,y_predict)


print score3, predictor, "Total R2 linear regression for all vars"
f,pval= f_regression(test,y_predict)

pval=pd.Series(pval,index=test.columns)
pval.sort_values(inplace=True)
print "################ Linear Regression Pvals <0.05 ######################"
imp_var=[]
for i in range(len(pval)):
    if pval[i] <= 0.05:
        print pval.keys()[i], pval[i], coeffs[pval.keys()[i]], "var, pval, coeff"
        imp_var.append(pval.keys()[i])

print "Total Multiple Regression analysis"



Reg.fit(test,y_predict)
score4=Reg.score(test,y_predict)
LRslim_CV=cross_validation.cross_val_score(Reg, holdout, y_holdout , cv=15) #cross validate linear regs with fewer features





""" Print summary of scores and cross-val scores, as well as feature averages"""
print "Means and predictiors"
for i in range(len(test.columns)):
    print "Feature: %s , Means: %s=  %f , %s= %f"%(test.columns.values[i],tnoms[0],  np.mean(t1[test.columns[i]]), tnoms[1],np.mean(t2[test.columns[i]]))

print "############################################################"
print "          ##############################                    "
print "                 ###############                            "
print "Predicting: ", predictor
print "Summary internal R2 scores, full RF= %s, slim RF= %s, full LR= %s, slim LR= %s, SVM/SVC: %s, Ridge: %s" %(RF_full_score_,score2,score3,score4,svm_score,rid_score)
print "Cross Validate SCORES"
print "Holdout score Ridge: %s,   slimRidge: %s ,  slimRF: %s, fatRF %s,   SVC: %s, slim LR: %s, full LR: %s"   %(np.mean(Ridge_scores),slimrid_CV,np.mean(RF_scores),np.mean(RF_full_holdscore),np.mean(svm_scores),np.mean(LRslim_CV),np.mean(LRfull_CV))
print "GBC CV score: %s" %(np.nanmean(gbc_scores))

plt.show()
sys.exit()
