import sys, os
sys.path.append("tslib")
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.src.synthcontrol.multisyntheticControl import MultiRobustSyntheticControl
from tslib.tests import testdata
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt




def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]
    


def run_RSC(sing_vals, trainDF, testDF, treat_key, ctrls): 
    # model
    length=len(trainDF)
    #hyperparameter
    singvals = sing_vals

    rscModel = RobustSyntheticControl(treat_key, singvals, length, probObservation=1,  modelType='svd', svdMethod='numpy', otherSeriesKeysArray=ctrls)

    # fit the model
    rscModel.fit(trainDF)

    # save the denoised training data
    denoisedDF = rscModel.model.denoisedDF()

    # predict - all at once
    predictions = rscModel.predict(testDF)
    return [denoisedDF, predictions]




def plot_RSC_res_placebo(sc_predicted_preintervention, predictions, treat_key, trainDF, test_DF,yearsToPlot,end_date, lockdown,
method='RSC',
overplot=True, loess=True, loess_frac=0.05, cat='',horizon=4,save=True, show=True,color = 'black'): 
    # ax=plt.gca()
    horizon=5
    real_nyc = np.append(trainDF[treat_key], test_DF[treat_key], axis=0)
    predicted=np.zeros(len(predictions))
    if method=='RSC':
        predicted = np.append(sc_predicted_preintervention, predictions, axis=0)
    elif method=='linreg':
        predicted = np.append(sc_predicted_preintervention, predictions, axis=0)
    if not overplot:
        plt.figure()
        
    data = pd.DataFrame(real_nyc,columns=['real_nyc'])
    if loess: 
        loess_predicted=sm.nonparametric.lowess(predicted.flatten(), range(len(yearsToPlot)),return_sorted=False,frac=loess_frac)
        data['synth_nyc'] = loess_predicted
        loess_predicted=sm.nonparametric.lowess(real_nyc, range(len(yearsToPlot)),return_sorted=False,frac=loess_frac)
        data['real_nyc'] = loess_predicted
    else: 
        data['synth_nyc'] = predicted


    if not overplot: 
        plt.plot_date(yearsToPlot, data['real_nyc'], color='red', label='observed NYC',ls='solid',markersize=2);
        plt.plot_date(yearsToPlot, data['synth_nyc'], color='blue', label='predicted NYC',ls='solid',markersize=2);
        plt.axvline(x=end_date, linewidth=1, color='black', label='Intervention');plt.axvline(x=lockdown, linewidth=1, color='black', label='lockdown')
        legend = plt.legend(loc=(1,0.7), shadow=False)
    else: 
        # plt.plot_date(yearsToPlot[len(sc_predicted_preintervention)-1:-1], test_DF[treat_key].values-predictions.flatten(),alpha=0.3, markersize=0,ls='--')#,ls='solid',);

        plt.plot_date(yearsToPlot, data['real_nyc']-data['synth_nyc'],alpha=0.2, markersize=1.5,ls='--',color='black');
        # plt.axvline(x=end_date, linewidth=1, color='black', label='Intervention');plt.axvline(x=lockdown, linewidth=1, color='black', label='lockdown')


    
#     plt.plot_date(yearsToPlot, , color='blue', label='predicted NYC',ls='solid',markersize=2);plt.axvline(x=end_date, linewidth=1, color='black', label='Intervention');plt.axvline(x=lockdown, linewidth=1, color='black', label='lockdown')

    # plt.setp(ax.get_xticklabels()[::4], visible=False)
    # plt.xticks(rotation=45)
    # plt.plot_date(yearsToPlot, real_nyc, color='red', label='observed',ls='solid',markersize=2)
    # plt.plot_date(yearsToPlot, predicted, color='blue', label='predicted',ls='solid',markersize=2)
    # plt.axvline(x=end_date, linewidth=1, color='black', label='Intervention')
    # plt.axvline(x=lockdown, linewidth=1, color='black', label='lockdown')

    
    # plt.title('Synthetic control placebo check, '+treat_key+' treated')
    # plt.tight_layout()
    # if save: 
    #     plt.savefig('synthcontrol_'+method+'-treated-'+treat_key+'_'+cat+'.pdf',bbox_to_inches='tight')
    # if show: 
        # plt.show()


def validate(treat_key, trainDF, testDF, singvals, ctrls,end_val_date,end_date):
    length=len(trainDF)
    rscModel = RobustSyntheticControl(treat_key, singvals, length, probObservation=1,  modelType='svd', svdMethod='numpy', otherSeriesKeysArray=ctrls)

    # fit the model
    rscModel.fit(trainDF)

    # save the denoised training data
    denoisedDF = rscModel.model.denoisedDF()

    # predict - all at once
    predictions = rscModel.predict(testDF)

    y = testDF[treat_key]
    return np.mean( np.square(predictions - y) ) # return mse 



def lm_res(predictions, testDF, treat_key): 
    n = len(predictions)
    X = sm.add_constant(range(n))
    res = testDF[treat_key] - predictions
    model = sm.OLS(res,X)
    results = model.fit()
    param = results.params.values # intercept, coefficient 
    return [results, param]

import cvxpy as cvx

def SC(treatedKey, otherKeys,trainDF,nonneg=True):
    # control (times x paths array)
    # outcome (times x 1 array)
    (n,p) = trainDF.shape
    control = trainDF[otherKeys].values
    outcome = trainDF[treatedKey].values.reshape([n,1])
    

    w = cvx.Variable((control.shape[1],1), nonneg=nonneg)
    
    objective = cvx.Minimize(cvx.sum_squares(outcome - control*w))
    constraints = [cvx.sum(w) == 1]
    prob = cvx.Problem(objective, constraints)
    # The optimal objective value is returned by prob.solve()
    result = prob.solve(verbose=False)
    return w.value

def SC_ridge(treatedKey, otherKeys,trainDF,lmbda,nonneg=True):
    # control (times x paths array)
    # outcome (times x 1 array)
    (n,p) = trainDF.shape
    control = trainDF[otherKeys].values
    outcome = trainDF[treatedKey].values.reshape([n,1])
    
    w = cvx.Variable((control.shape[1],1), nonneg=nonneg)
    objective = cvx.Minimize(cvx.sum_squares(outcome - control*w) + lmbda*cvx.sum_squares(w) )
    constraints = [cvx.sum(w) == 1]

    prob = cvx.Problem(objective,constraints)
    # The optimal objective value is returned by prob.solve()
    result = prob.solve(solver='GUROBI', verbose=False)
    return w.value

def run_SC(trainDF, testDF, treat_key, ctrls,nonneg=True): 
    # model
    length=len(trainDF)
    w = SC(treat_key, ctrls, trainDF,nonneg)
    sc_pre = np.matmul(trainDF[ctrls].values, w)
    predictions = np.matmul(testDF[ctrls].values, w)
    return [w, sc_pre, predictions]



def get_data_dicts(incident_series,cat, treat_key, ctrls, cities_, start_date, end_date, start_2, end_2_date,
    city_pop,daily=True,weekly_agg='W'):
    '''
    start_date, end_date, start_2, end_2_date
    '''    
    trainDataMasterDict = {}
    trainDataDict = {}
    testDataDict = {}; means = {}
    for city in cities_:
        violent = incident_series[ (incident_series.city==city) & (incident_series['category']==cat )].iloc[:,3:]
        
        violent = violent.transpose()
        violent.index = pd.to_datetime(violent.index)
        # print(city, len(violent[start_date:end_date].values.flatten()))
        violent = violent / (city_pop[city_pop['city']==city]['population'].values[0]*1.0)
        means[city] = city_pop[city_pop['city']==city]['population'].values[0]*1.0
        violent = violent - np.mean(violent[start_date:end_date])

        if daily: 
            # print('after divide', city, len(violent[start_date:end_date].values.flatten()))
            if len(violent[start_date:end_date].values.flatten()) > 0: 
                city_n = len(violent[start_date:end_date].values.flatten())
                test_n = len(violent[start_2:end_2_date].values.flatten()) 
            trainDataDict.update({city: violent[start_date:end_date].values.flatten()})
            testDataDict.update({city: violent[start_2:end_2_date].values.flatten()})
        
        else:
            # weekly aggregation 
            violent_sum = violent.resample(weekly_agg, how='sum')
            # print('after divide',city, len(violent_sum[start_date:end_date].values.flatten()))
            if len(violent_sum[start_date:end_date].values.flatten()) > 0: 
                city_n = len(violent_sum[start_date:end_date].values.flatten())
                test_n = len(violent_sum[start_2:end_2_date].values.flatten()) 
            trainDataDict.update({city: violent_sum[start_date:end_date].values.flatten()})
            testDataDict.update({city: violent_sum[start_2:end_2_date].values.flatten()})

    
    trainDataDict = {key: value for key, value in trainDataDict.items() if len(value) > 0}
    testDataDict = {key: value for key, value in testDataDict.items() if len(value) > 0}
    supported_cities = trainDataDict.keys()
    # print(supported_cities)
    # print( [len(val) for val in trainDataDict.values() ] )

    if daily:
        trainDataDict.update({'intercept':np.ones( city_n  )})    
        testDataDict.update({'intercept':np.ones( test_n )})
    else: 
        trainDataDict.update({'intercept':np.ones( city_n )})    
        testDataDict.update({'intercept':np.ones( test_n )})
    # print( [len(val) for val in trainDataDict.values() ] )
    trainDF = pd.DataFrame(data=trainDataDict)
    testDF = pd.DataFrame(data=testDataDict)
    if daily: 
        return [trainDF, testDF,violent, supported_cities, means]
    else:
        return [trainDF, testDF,violent_sum, supported_cities, means]

def plot_res_placebo_diff(sc_pre, yearsToPlot, predictions, treat_key, trainDF, test_DF, axs, save=True, show=True): 
    
    real_nyc = np.append(trainDF[treat_key], testDF[treat_key], axis=0)
    predicted = np.append(sc_pre, predictions, axis=0)
    diff = real_nyc - predicted
    axs[ind].plot_date(yearsToPlot, (diff), color='blue', label=treat_key,ls='solid',markersize=2)
#     plt.plot_date(yearsToPlot, predicted, color='blue', label='predicted',ls='solid',markersize=2)
    axs[ind].axvline(x=end_date, linewidth=1, color='black')
    axs[ind].axhline(0, linewidth=1, color='red', ls='--')
    axs[ind].legend(loc=(1,0.7))

    

def get_placebo_distributions(residual_params, cities_,city_to_ind, cat='violent',plot=True):
    params = np.zeros((len(cities_),2))
    for ind,x in enumerate(cities_): 
        params[ind,:] = residual_params[x][1]
    if plot: 

        plt.figure()
        plt.hist(params[:,0])
        plt.axvline(params[city_to_ind['New York City'], 0],color='black',label='NYC')
        plt.legend()
        plt.title('Intercept distribution')
        plt.tight_layout(); plt.savefig('figs/intercept_dist_'+cat+'.pdf')

    ecdf_int = ECDF(params[:,0])
    ecdf_coef = ECDF(params[:,1])

    print ecdf_int(params[city_to_ind['New York City'], 0])
    print ecdf_int(params[city_to_ind['New York City'], 1])

    if plot:
        plt.figure()
        plt.hist(params[:,1])
        plt.axvline(params[city_to_ind['New York City'], 1],color='black',label='NYC')
        plt.title('Coefficient distribution')
        plt.legend()
        plt.tight_layout(); plt.savefig('figs/coefficient_dist_'+cat+'.pdf')
    return [ecdf_int(params[city_to_ind['New York City'], 0]),ecdf_int(params[city_to_ind['New York City'], 1]),ecdf_int, ecdf_coef]

from matplotlib.pyplot import cm
import matplotlib as mpl

def mse_(a1,a2):

    return np.sqrt(np.mean(np.square(a1 - a2))) 

def get_placebo_checks(residual_params,placebo_ates,not_nyc, cities_, trainDF, 
    testDF, yearsToPlot, end_date, lockdown, overplot=True,loess=False,loess_frac=0.05,
    nonneg=True,horizon=1,cat='',method='linreg',singvals=0,save=True,plot=True,ridge=False,ridge_params=None):
    '''
    Return residual parameters of fit
    Return placebo ATEs
    '''
    horizon=1
    mses = np.zeros(len(not_nyc))
    mse_ratios = np.zeros(len(not_nyc))
    # color=cm.rainbow(np.linspace(0,1,len(not_nyc)))
    ax_ = plt.axes()
    ax_.set_prop_cycle('color',[plt.cm.magma(i) for i in np.linspace(0, 1, len(not_nyc))])

    for ind,city in enumerate(not_nyc): 
        # c = next(color)
        treat_key = city
        ctrls = [x for x in cities_ if x!=treat_key] 
        if method=='linreg':
            if ridge: 
                [trainDF_pre, testDF_pre, lmbdas] = ridge_params
                ws_lmbda = [None]*len(lmbdas); mses_=np.zeros(len(lmbdas))
                for lmbda_ind,lmbda in enumerate(lmbdas): 
                    # linear reg
                    sc_ctrls_ = ctrls + ['intercept']
                    w = SC_ridge(treat_key, sc_ctrls_, trainDF_pre, lmbda,nonneg=nonneg); 
                    ws_lmbda[lmbda_ind]=w.flatten()
                    predictions = np.matmul(testDF_pre[sc_ctrls_].values, w)
                    mses_[lmbda_ind] = np.sqrt(np.mean(np.square(testDF_pre[treat_key].values - predictions.flatten())))
                best_lmbda = lmbdas[np.argmin(mses_)] 
                w_ridge = ws_lmbda[np.argmin(mses_)]
                w = w_ridge 
                sc_pre = np.matmul(trainDF[sc_ctrls_].values, w)
                predictions = np.matmul(testDF[sc_ctrls_].values, w)
            else: 
                [w, sc_pre, predictions] = run_SC(trainDF, testDF, treat_key, ctrls,nonneg)

        elif method=='RSC': 
            [denoisedDF, predictions] = run_RSC(singvals, trainDF, testDF, treat_key, ctrls)
            sc_pre = denoisedDF[treat_key].values

        mses[ind] = np.sqrt(np.mean(np.square(trainDF[treat_key] - sc_pre.flatten())))
        mse_ratios[ind] = mse_(testDF[treat_key],predictions.flatten()) / mse_(trainDF[treat_key], sc_pre.flatten())

        # print city
        if plot: 
            plot_RSC_res_placebo(sc_pre, predictions, treat_key, trainDF, testDF,yearsToPlot,end_date, lockdown,
                save=save,overplot=overplot,loess=loess,loess_frac=loess_frac,horizon=horizon, cat = cat, show=False,method=method)
        tau = np.mean(testDF[treat_key]-predictions.flatten()) 
        placebo_ates[treat_key] = tau
        # fit linear model
        [res,param] = lm_res(predictions.flatten(),testDF, treat_key)
        residual_params[treat_key] = [res,param]
    # plt.show()
    # plt.savefig('placebo_'+method+'_'+cat+'.pdf',bbox_to_inches='tight')
    return [residual_params, placebo_ates,mses, mse_ratios ]


def init_placebo_params(og_treat_key, nyres,nyparam, testDF, pred): 
    residual_params = {}
    residual_params[og_treat_key] = [nyres,nyparam]
    placebo_ates = {}
    placebo_ates[og_treat_key ] = np.mean(testDF[og_treat_key]-pred) 
    return [residual_params, placebo_ates]
