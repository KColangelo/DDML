# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:04:04 2020

This file provides the main double debiased machine learning estimator for
continuous treatments. The class "DDMLCT" performs the estimation when the
.fit method is called. 

DDMLCT is initialized by passing 2 models (such as sklearn models) which have
.fit and .predict methods. One for estimating the generalized propensity score (GPS)
and one for estimation gamma. model1 is used for estimating gamma and model2 is 
used for estimation of the GPS


Comments on packages used:
    -copy is used to make copies of the models used to initiate DDMLCT. 
    -pandas is used during rescaling in order to rescale non-dummies
    -scipy.stats.norm is used for the computation of the gaussian kernel
    -numpy is used for the storage of most of the data and attributes. If data is
     passed to the .fit method as a pandas dataframe it is converted to numpy arrays
     before being passed to the models to fit. 
    


@author: Kyle Colangelo
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import copy
import sklearn
#This function evaluates the gaussian kernel wtih bandwidth h at point x
def gaussian_kernel(x,h):
    k = (1/h)*norm.pdf((x/h),0,1)
    return k

#This function evaluates the epanechnikov kernel
def e_kernel(x,h):
    k = (1/h)*(3/4)*(1-((x/h)**2))
    k = k*(abs(x/h)<=1)
    return k


    
class DDMLCT:
    def __init__(self,model1,model2):
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.beta = np.array(())
        self.std_errors = np.array(())
        self.Vt = np.array(())
        self.Bt = np.array(())
        self.summary = None
        self.scaling = {'mean_Y':0,
                     'sd_Y':1,
                     'mean_T':0,
                     'sd_T':1}
        self.naive_count = 0
        self.L = 5
        self.gamma_models = []
        
    def reset(self):
        self.beta = np.array(())
        self.std_errors = np.array(())
        self.Vt = np.array(())
        self.Bt = np.array(())
        self.summary = None
        self.scaling = {'mean_Y':0,
                     'sd_Y':1,
                     'mean_T':0,
                     'sd_T':1}
        self.naive_count = 0
        self.L = 5
        
    def naive(self,Xf,XT,Xt,Y,I,I_C,L):
        if self.naive_count < self.L:
            self.gamma_models.append(self.model1.fit(np.column_stack((XT[I_C],Xf[I_C])),Y[I_C]))
            self.naive_count +=1
        gamma = self.gamma_models[L].predict(np.column_stack((Xt[I],Xf[I])))
        return gamma
    
    def ipw(self,Xf,g,I,I_C):
        self.model2.fit(Xf[I_C],g[I_C])
        gps = self.model2.predict(Xf[I])
        
        return gps
    

    def fit_L(self,Xf,XT,Xt,Y,g,K,I,I_C,L):
        gamma = self.naive(Xf,XT,Xt,Y,I,I_C,L)
        gps = self.ipw(Xf,g,I,I_C)
        self.kept = np.concatenate((self.kept,I[gps>0]))
        
        # Compute the summand
        psi = np.mean(gamma[gps>0]) + np.mean(((Y[I][gps>0]-gamma[gps>0])*(K[I][gps>0]/gps[gps>0])))

        # Average over all indexes to get an estimate of beta hat
        beta_hat = np.mean(psi)
        
        return beta_hat, gamma, gps
    
        
    def fit_t(self,X,T,Y,trep,L,XT,Xt):
        # If no bandwidth is specified, use rule of thumb
        self.kept = np.array((),dtype=int)
        
        

        T_t = T-trep
        g = gaussian_kernel(T_t,self.h)
        K = e_kernel(T_t,self.h)
        gamma = np.zeros(self.n)
        gps = np.zeros(self.n)
        beta_hat = np.zeros(L)
        
        
            
        for i in range(L):
            if L==1:
                I = self.I_split[0]
                I_C = self.I_split[0]
            else:
                I=self.I_split[i]
                # Define the complement as the union of all other sets
                I_C = [x for x in np.arange(self.n) if x not in I]
                

            beta_hat[i], gamma[I], gps[I] = self.fit_L(X,XT,Xt,Y,g,K,I,I_C,i) 
        
        self.n = len(self.kept)
        beta_hat = np.mean(beta_hat)
        self.beta = np.append(self.beta,beta_hat)
        IF =(K[self.kept]/gps[self.kept])*(Y[self.kept]-gamma[self.kept]) + gamma[self.kept] - beta_hat
        std_error = np.sqrt((1/((self.n)**2))*np.sum(IF**2))
        self.Bt = np.append(self.Bt,(1/(self.n*(self.h**2)))*(np.sum((K[self.kept]/gps[self.kept])*(Y[self.kept]-gamma[self.kept]))))
        self.Vt = np.append(self.Vt,(std_error**2)*(self.n*self.h))
        self.std_errors = np.append(self.std_errors,std_error)
        self.gps.loc[self.kept,str(trep[0])] = gps[self.kept]
        
        
    def fit(self,X,T,Y,t_list,L=5,h=None,basis=False,standardize=False):
        self.reset()
        self.naive_count = 0
        self.n = len(Y)
        self.t_list = np.array(t_list,ndmin=1)
        self.L = L
        self.I_split = np.array_split(np.array(range(self.n)),L)

        if h==None:
            self.h = np.std(T)*(self.n**-0.2)
        else:
            self.h = h
            
        X,T,Y,t_list = self.reformat(X,T,Y,t_list,standardize)
        
        

        self.gps = pd.DataFrame(index = range(self.n))
        if basis==True:
            XT,Xf,ind = self.augment(X,T)
            if standardize == True:
                Xf = self.scale_non_dummies(Xf)[0]
                XT, scaler = self.scale_non_dummies(XT)
        else:
            XT = T
            Xf = X
            
        for t in np.array((t_list),ndmin=1):
            self.n = len(Y)
            trep = np.repeat(t,self.n)
            if basis==True:
                Xt = self.augment(X,trep,ind)[0]
                if standardize == True:
                    Xt = self.scale_non_dummies(Xt,scaler)[0]
            else:
                Xt = trep
            self.fit_t(Xf,T,Y,trep,L,XT,Xt)
            
        self.h_star = ((np.mean(self.Vt)/(4*(np.mean(self.Bt)**2)))**0.2)*(self.n**-0.2)
    
        if standardize==True:
            self.descale()
        
        self.gps.columns = self.t_list
        
    def augment(self,X,T,ind=None):
        T = T.reshape(len(T),1)
        XT= np.column_stack((T,(T**2),(T**3),T*X))
        Xf = np.column_stack((X,X**2,X**3))
        Xf = np.unique(Xf,axis=1)
        if np.array_equal(ind,None):
            XT,ind = np.unique(XT,axis=1,return_index=True)
        else: 
            XT = XT[:,ind]
        return XT, Xf, ind

    def scale_non_dummies(self,D,scaler=None):
        D = pd.DataFrame(D)
        if scaler==None:
            scaler = sklearn.preprocessing.StandardScaler()  
            D[D.select_dtypes('float64').columns] = scaler.fit_transform(D.select_dtypes('float64')) 
        else:
            D[D.select_dtypes('float64').columns] = (D[D.select_dtypes('float64').columns]-scaler.mean_)/scaler.scale_
        return np.array(D), scaler
    
    def reformat(self,X,T,Y,t_list,standardize):
        if standardize==True:
            df = pd.DataFrame(data = np.column_stack((Y,T,X)))
            self.scaling = {'mean_Y':np.mean(df[0]),
                     'sd_Y':np.std(df[0]),
                     'mean_T':np.mean(df[1]),
                     'sd_T':np.std(df[1])}
            df[df.select_dtypes('float64').columns] = sklearn.preprocessing.StandardScaler().fit_transform(df.select_dtypes('float64'))
            
            Y = df[0]
            T = df[1]
            X = df.loc[:,2:]
            del df
            t_list = (t_list-self.scaling['mean_T'])/self.scaling['sd_T']
            self.h = self.h/self.scaling['sd_T']
        X = np.array((X))
        T = np.array((T))
        Y = np.array((Y))
        return X,T,Y,t_list
    
    def descale(self):
        self.std_errors = self.std_errors*self.scaling['sd_Y']
        self.h_star = self.h_star*self.scaling['sd_T']
        self.beta = (self.beta*self.scaling['sd_Y']) +self.scaling['mean_Y']
        self.h = self.h*self.scaling['sd_T']
        
        
        
        

        
        
    
        

    

    
