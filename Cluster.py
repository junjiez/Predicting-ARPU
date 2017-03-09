#WDI model
""" TO DO
holdout data
"""
import pandas as pd
import operator
import statsmodels.formula.api as smf
import sys
from sklearn.decomposition import PCA
import seaborn as sns
import sklearn.linear_model  as lm
#from mpl_toolkits.mplot3d import Axes3D
#import clustering
from patsy import dmatrices, dmatrix, demo_data
import numpy as np
import statsmodels.api as sm
import statsmodels.stats as sms
from sklearn.cross_validation import train_test_split
import random
import pylab as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest as KBest
from sklearn.feature_selection import f_regression

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
import sklearn.feature_selection as fs
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans
from sklearn import cluster
import math
from scipy import stats
import matplotlib.image as mpimg
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import colors
import six
#mpl.rcParams['pdf.fonttype'] = 42
#sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
old_settings = np.seterr(all='ignore')


def get_spaced_colours(n): #Random colour generator
    col=[]
    colours_ = list(six.iteritems(colors.cnames))
    for i in range(n):
        col.append(colours_[i][0])
    col=np.asarray(col)
    return col
    
    
inny=pd.ExcelFile("cleaned_data.xlsx") # read in excel
df=inny.parse("Sheet1") #load sheet into dataframe
df=df
train=df
dat=train   

#remove extreme airtime spends

air=train["How_much_MK_do_you_spend_on_airtime_in_a_month"].value_counts()
remove=air[air <=2].index #remove anything that appears less than 2 times = 3% of responses
train["How_much_MK_do_you_spend_on_airtime_in_a_month"].replace(remove, train["How_much_MK_do_you_spend_on_airtime_in_a_month"].median(skipna=True),inplace=True)

#################

train.drop(["Smart_phone_basic_phone_internet_enabled_2nd_phone","did_you_buy_your_2nd_phone_new_or_used","What_is_the_2nd_reason_you_use_your_phone","What_is_the_brand_of_your_solar_product"],axis=1,inplace=True)
  
  
  
dat=train  

#convert instances of internet phonet o smart phone
dat["Smart_phone_basic_phone_internet_enabled_1st_phone"][dat["Smart_phone_basic_phone_internet_enabled_1st_phone"]=='internetenabled']='smartphone'
dat.drop(["reported_monthly_income","Interviewer","Phone_number","Point_(1-5)","district"], axis=1, inplace=True)
dat.drop(["Is_your_solar_sold_piece_by_piece","If_yes,_is_it_(solartorch,_5-10w,_10-25w,_25-50,_50-100w)","What_mode_of_transport_do_you_use_to_purchase_light_Motorbike,_minibus,_bike,_walk,_car,_other","What_could_be_more_important_to_have_a_solarlight_or_phone_charger_if_same_price_","Where_did_you_buy_your_solar_product_shop_trading_center,_shop_city,_shop_in_village,_shop_trading_center,_friend,_NGO,_church","What_is_the_second_use_for_your_solar_product","What_is_the_primary_use_of_your_solar_product"],axis=1,inplace=True)
catvar=list(train.dtypes[train.dtypes == "object"].index) #Select cat columns
numvar=list(train.dtypes[train.dtypes != "object"].index) #Select numeric columns


for varx in catvar:
    dummies=pd.get_dummies(dat[varx],prefix=varx)
    dat=pd.concat([dat,dummies],axis=1)
    dat.drop([varx],axis=1,inplace=True) 
    

    
        
                
""" create sub dataframes of people of interest"""
print "########### SUB DF ##########################"
#people who own solar

solar_owners = dat[dat["do_you_own_any_solar__yes"] == 1.0] 
solar_losers = dat[dat["do_you_own_any_solar__no"] == 1.0]

rural = dat[dat["Rural_peri-urban_rural"] == 1.0]
urban = dat[dat["Rural_peri-urban_peri-urban"] == 1.0]

gift_givers = dat[dat["Have_you_ever_given_a_phone_as_a_gift_Yes_no_yes"] == 1.0]
gift_losers = dat[dat["Have_you_ever_given_a_phone_as_a_gift_Yes_no_no"] == 1.0]

solar_light= dat[dat["What_is_your_source_of_light_electricity,_battery_torch,_candle,_paraffin,_solar_light,_cell_phone,_other_solarlight"]==1.0]

"""
temp= train[train["do_you_own_any_solar_"] == 'yes']
temp["What_is_the_brand_of_your_solar_product"].fillna('other',inplace=True)
temp= temp["What_is_the_brand_of_your_solar_product"]

solar_ownersx=dat[dat["do_you_own_any_solar__yes"] == 1.0] 
solar_ownersx.drop(["What_is_the_brand_of_your_solar_product_draft","What_is_the_brand_of_your_solar_product_ecco","What_is_the_brand_of_your_solar_product_goldstone","What_is_the_brand_of_your_solar_product_sunking"],axis=1,inplace=True)
solar_ownersx = pd.concat((solar_ownersx,temp),axis=1)
"""


solar_owners = dat[dat["do_you_own_any_solar__yes"] == 1.0] 
solar_losers = dat[dat["do_you_own_any_solar__no"] == 1.0]



boost_yes = dat[dat["do_you_use_a_booster_charge_(15_minutes_to_charge_your_phone)_no"] == 1.0] 
boost_no = dat[dat["do_you_use_a_booster_charge_(15_minutes_to_charge_your_phone)_yes"] == 1.0]

solar_qual=pd.concat([ dat[dat["do_you_think_solar_is_good_quality_Yesno_yes"] == 1.0], dat[dat["do_you_think_solar_is_good_quality_Yesno_no"] == 1.0]])

solar_yes = solar_owners[solar_owners["do_you_think_solar_is_good_quality_Yesno_yes"] == 1.0] 
solar_no= solar_owners[solar_owners["do_you_think_solar_is_good_quality_Yesno_no"] == 1.0]
TNM_old = dat[dat["Which_network_SIM_card_do_you_have_tnm"] == 1.0]
TNM=dat[dat["real_airtime"] >=0.0] # made like this as some errors in TNM yes no question.

Airtel = dat[dat["Which_network_SIM_card_do_you_have_airtel"] == 1.0]
Mal=dat[dat["Sex_male"] == 1.0]
Fem=dat[dat["Sex_female"] == 1.0] 
sim=pd.concat([TNM,Airtel])


#fill blanks in TNM dat
TNM["real_airtime"].fillna(np.nanmean(TNM["real_airtime"]),inplace=True)
dat.drop("real_airtime",axis=1,inplace=True)
#dat=TNM

#Impute to deal with missing values
for varx in catvar: # fillNA with most common value
    #x=train[varx].value_counts().keys()[0]
    #train[varx].fillna(x,inplace=True)
    train[varx].fillna('Missing',inplace=True)
    
for i in numvar[2:]: #fill in missing numerical values, skipping point and date
    #print i
    train[i][np.isnan(train[i])] = np.nanmean(train[i])
    
#Clustering on whole data set 
#chage dat to cluster and cl= to the DB of choice
#dat_to_cluster=pd.concat([dat["How_much_MK_do_you_spend_on_airtime_in_a_month"],dat["Which_network_SIM_card_do_you_have_tnm"]],axis=1)
dat_to_cluster=TNM
Ncl=13  #number of clusters wanted for supervised
direc='figs\clusters'
plt.figure()
plt.scatter(dat["Which_network_SIM_card_do_you_have_tnm"],train["How_much_MK_do_you_spend_on_airtime_in_a_month"])
cl=TNM["How_much_MK_do_you_spend_on_airtime_in_a_month"].reshape(-1, 1)


if 1==1:
    #get ideal # clusters
    Nclust=np.linspace(1,cl.shape[0]-2,20,dtype=int) 
    
    KM_error_rate=[]
    for i in Nclust:
        k_test = cluster.KMeans(n_clusters=i)
        test_fit=k_test.fit(cl)
        KM_error_rate.append(test_fit.inertia_)
    
    plt.figure("Kmeans cluster analysis - full data")
    ax=plt.subplot(211)
    ax.plot(Nclust,KM_error_rate,lw=3)  
    ax.set_ylim([0,np.max(KM_error_rate)])
    ax.set_ylabel("Distance to centroid")
    ax.set_xlabel("Number of clusters")
    plt.savefig("Kmeans.pdf")


    

    
    kmeans = KMeans(n_clusters=Ncl)
    clusters = kmeans.fit(cl)
    cl=pd.DataFrame(cl)
    dat_to_cluster['cluster'] = pd.Series(clusters.labels_, index=dat_to_cluster.index)
    #test.plot(kind='scatter',x='Age',y='How_much_MK_do_you_spend_on_airtime_in_a_month', c=get_spaced_colours(5), figsize=(16,8))
    dat_to_clusterc=dat_to_cluster.copy()
    print dat_to_cluster['cluster'].value_counts()
    clust_dat_to_cluster=dat_to_clusterc["cluster"].value_counts()
    cutoff=dat_to_cluster['cluster'].count()/10.
    remove=clust_dat_to_cluster[clust_dat_to_cluster <=cutoff].index #remove anythin less than 10%
    dat_to_clusterc["cluster"].replace(remove, 99.0 ,inplace=True)
    dat_to_cluster.drop('cluster',inplace=True,axis=1)
    print dat_to_clusterc['cluster'].value_counts().divide(dat_to_clusterc['cluster'].count())*100.
    
    ##fout=file("%s\cluster_summary.txt" %(direc),'w+')
       
    print "####################################"
    for i in dat_to_clusterc["cluster"].value_counts().index:
        print "---------------- %0.f %% ------------" %(((float(dat_to_clusterc[dat_to_clusterc['cluster']==i]["cluster"].count())/float(dat_to_clusterc["cluster"].count()))*100.))
        #fout.write("----------------size:  %0.f %%------------ \n" %(((float(dat_to_clusterc[dat_to_clusterc['cluster']==i]["cluster"].count())/float(dat_to_clusterc["cluster"].count()))*100.)))
        brand = dat_to_clusterc[dat_to_clusterc["cluster"] == i] 
        sol_ = dat_to_clusterc
        print "mean spend", np.nanmean(brand["real_airtime"]), "vs average:", np.nanmean(dat_to_cluster["real_airtime"]), "TNM and Airtel share %s %s %%:" %(np.mean(brand["Which_network_SIM_card_do_you_have_tnm"])*100.,np.mean(brand["Which_network_SIM_card_do_you_have_airtel"])*100.)
        #fout.write("mean spend %2.f, TNM and Airtel share %s %s %% \n" %(np.nanmean(brand["How_much_MK_do_you_spend_on_airtime_in_a_month"]),np.mean(brand["Which_network_SIM_card_do_you_have_tnm"])*100.,np.mean(brand["Which_network_SIM_card_do_you_have_airtel"])*100.))
        pc=float(dat_to_clusterc[dat_to_clusterc['cluster']==i]["cluster"].count())/float(dat_to_clusterc["cluster"].count())*100
        pc=np.round(pc,1)
        #pc2= np.round(float(brand[brand["Which_network_SIM_card_do_you_have_tnm"] == 1.0]["cluster"].count())/float(brand["cluster"].count()) *100.,1)
        #pc3= np.round(float(brand[brand["Which_network_SIM_card_do_you_have_airtel"] == 1.0]["cluster"].count())/float(brand["cluster"].count()) *100.,1)
        pc2= (float(brand.count()[0])/float(dat_to_cluster.count()[0]))*100
        pc3=100.
        ##fout.write("TNM:Airtel split: %0.f : %0.f\n" %(pc2,pc3))
        for t in dat_to_clusterc.columns:
            if t != 'cluster':
                if stats.ttest_ind(brand[t],sol_[t] )[1] < 0.05: 
                    A=brand
                    B=dat_to_cluster
                    
                    #fout.write("\t %s \n" %(t))
                    #fout.write(" \t Cluster differences: This %s to Average %s \n" %(np.mean(brand[t]), np.mean(sol_[t])))
                    print "%s this cluster: %s  Average: %s"       %(t, np.mean(A[t]), np.mean(B[t]))
                    #fout.write(" \t cluster: %s  %% av: %s  \n"       %( np.mean(A[t]), np.mean(B[t]) ))
                    #plt.figure("%s - cluster %s %0.f %%" %(t, i, pc))
                    #barnames='%s cluster' %(i),'average'
                    #ax=plt.subplot(111)
                    #height=np.mean(A[t]),np.mean(B[t])
                    #yerr=stats.sem(A[t]),stats.sem(B[t])
                    #ax.bar(range(2),height,yerr=yerr)
                    #ax.set_xticklabels(barnames)
                    #plt.savefig("%s\%s-%0.f.pdf" %(direc, t, pc))
                if stats.ttest_ind(brand[t],sol_[t] )[1] < 0.5 and stats.ttest_ind(brand[t],sol_[t] )[1] > 0.05: 
                    A=brand
                    B=dat_to_cluster
                    print "Secondary (p 0.2): %s this cluster: %s  Average: %s"       %(t, np.mean(A[t]), np.mean(B[t]))
                
    dat_to_clusterc.drop('cluster',inplace=True,axis=1)
    #fout.close()

full=dat_to_cluster
predictor="How_much_MK_do_you_spend_on_airtime_in_a_month"
y_predict=full.pop(predictor)
#full.drop("Which_network_SIM_card_do_you_have_airtel",axis=1,inplace=True)
cnter=np.asarray(y_predict)

plt.figure("PCA plot")
ax=plt.subplot(211)

pca = PCA(n_components=2).fit(full)
pca_2d = pca.transform(full)

for i in range(len(cnter)):
    if bool(cnter[i]==1.0) == True:
        c1 = ax.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    marker='o')
    else:
        c2 = ax.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    marker='o')
ax.set_xlim([np.mean(pca_2d[:,0])-3*np.std(pca_2d[:,0]),np.mean(pca_2d[:,0])+3*np.std(pca_2d[:,0])])
ax.set_ylim([np.mean(pca_2d[:,1])-3*np.std(pca_2d[:,1]),np.mean(pca_2d[:,1])+3*np.std(pca_2d[:,1])])

"""  """
kmeans = KMeans(n_clusters=Ncl)
clusters = kmeans.fit(cl)
cl=pd.DataFrame(cl)
dat_to_cluster['cluster'] = pd.Series(clusters.labels_, index=dat_to_cluster.index)
#test.plot(kind='scatter',x='Age',y='How_much_MK_do_you_spend_on_airtime_in_a_month', c=get_spaced_colours(5), figsize=(16,8))
dat_to_clusterc=dat_to_cluster.copy()
clust_dat_to_cluster=dat_to_clusterc["cluster"].value_counts()
#cutoff=dat_to_cluster['cluster'].count()/10.
remove=clust_dat_to_cluster[clust_dat_to_cluster <=cutoff].index #remove anythin less than 10%
dat_to_clusterc["cluster"].replace(remove, 99.0,inplace=True)
dat_to_cluster.drop('cluster',inplace=True,axis=1)
print dat_to_clusterc['cluster'].value_counts().divide(dat_to_clusterc['cluster'].count())*100.
dataxc=dat_to_clusterc.copy()
xyz=np.asarray(dataxc["cluster"] )  

#plt.figure("Kmean PCA")
ax2=plt.subplot(212)
colors=get_spaced_colours(9)
cl_nom=[]
for ic in dataxc["cluster"].value_counts().index:
    cl_nom.append(ic)
    
for i in range(len(cnter)):
    if xyz[i] == cl_nom[0]:
         x=1
    elif xyz[i]== cl_nom[1]:
         ax2.scatter(pca_2d[i,0],pca_2d[i,1],c=colors[0],marker='o')
    elif xyz[i]==cl_nom[2]:
         ax2.scatter(pca_2d[i,0],pca_2d[i,1],c=colors[2], marker='o')
    elif xyz[i]==cl_nom[3]:
         ax2.scatter(pca_2d[i,0],pca_2d[i,1],c=colors[4], marker='o')
    elif xyz[i]==cl_nom[4]:
         ax2.scatter(pca_2d[i,0],pca_2d[i,1],c=colors[8], marker='o')
    elif xyz[i]==cl_nom[5]:
         ax2.scatter(pca_2d[i,0],pca_2d[i,1],c='black', marker='o')

for ic in dataxc["cluster"].value_counts().index:
    cl_nom.append(ic)
    
            
                            
#ax2.legend([ca,cb,cc],['Cluster 0', 'Cluster 1',   'Cluster 2'])
#ax2.title('K-means clusters PCA')
ax2.set_xlim([np.mean(pca_2d[:,0])-3*np.std(pca_2d[:,0]),np.mean(pca_2d[:,0])+3*np.std(pca_2d[:,0])])
ax2.set_ylim([np.mean(pca_2d[:,1])-3*np.std(pca_2d[:,1]),np.mean(pca_2d[:,1])+3*np.std(pca_2d[:,1])])
print dataxc["cluster"].value_counts().divide(dataxc["Age"].count())*100.





plt.show()
sys.exit()