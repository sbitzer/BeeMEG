#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:22:13 2016

@author: bitzer
"""
import os
import numpy as np
import pandas as pd
import helpers

def get_dot_level_measures(fname='infres_collapse_201612121759.npz', 
                           resultsdir=helpers.resultsdir, S=2000, Smin=20):
    """Returns dot-level posterior model measures.
    
        Measures are computed from the model for each subject, trial and 
        presented dot. See "Returns" for details of measures.
        
        Parameters are sampled from their posterior, the model is used to 
        compute the measures for each trial and sampled parameter set and then
        the measures are averaged over the sampled parameter sets.
        
        Parameters
        ----------
        fname : str, default 'infres_collapse_201612121759.npz'
            file name of model inference result to be used as basis
        resultsdir : str, default helpers.resultsdir
            path to results directory where fname lies
        S : int, optional
            number of samples to draw from the parameter posterior
            ignored when measures loaded from file
        Smin : int, optional
            all returned measure averages which are based on less than Smin 
            values will be NaN, supposed to prevent too large variance in
            returned averages
            
        Returns
        -------
        dot_level : DataFrame
            the posterior model measures with following columns
                
        - logpost_left: log posterior probability of the left choice
        - loglik_left: log likelihood of dot under left option
        - m_evidence_left: momentary evidence that dot provides for left option, 
          equals loglik_left-loglik_right
        - surprise: negative log-probability of the observed dot under the
          model (including both options)
    """
    
    resfile = os.path.join(resultsdir, fname[:-4]+'_dot_level.h5')
    
    if os.path.isfile(resfile):
        with pd.HDFStore(resfile, mode='r') as store:
            dot_level = store['dot_level']
            S = store['S']
            Smin = store['Smin']
        
        print('Loaded dot-level measures from file (S=%d, Smin=%d)!' % (S, Smin))
    else:
        with np.load(os.path.join(resultsdir, fname)) as data:
            subjects = data['subjects']
            model = data['model'][()]
            pars = data['pars'][()]
            ep_means = data['ep_means']
            ep_covs = data['ep_covs']
        
        trind = np.arange(model.L)
        
        dot_level = pd.DataFrame([], index=pd.MultiIndex.from_product(
                [subjects, trind+1, np.arange(model.S)+1], names=['subject', 
                'trial', 'dot']))
    
        with pd.HDFStore(resfile, mode='w', complevel=7, complib='zlib') as store:
            store['S'] = pd.Series(S, name='S')
            store['Smin'] = pd.Series(Smin, name='Smin')
            for i, sub in enumerate(subjects):
                print('computing dot-level measures for subject %d ...' % sub)
                
                # sample transformed parameters from posterior
                parsamp = pars.transform(np.random.multivariate_normal(
                        ep_means[i, :], ep_covs[i, :, :], S))
                
                # initialise stores
                logpost = np.zeros((model.S, model.C, model.L, S))
                loglik = np.zeros((model.S, model.C, model.L, S))
                surprise = np.zeros((model.S, model.L, S))
                
                # generate measures for each sampled parameter set and trial
                for s in range(S):
                    # set model parameters to the sampled values
                    for p, name in enumerate(pars.names):
                        setattr(model, name, parsamp[s, p])
                        
                    # log-posteriors and log-likelihoods
                    lp, ll = model.compute_logpost_from_features(trind)
                    logpost[:, :, :, s] = lp[:, :, :, 0]
                    loglik[:, :, :, s] = ll[:, :, :, 0]
                           
                    # surprise (needs to be inside the loop, because it also
                    # uses prior stored in model)
                    surprise[:, :, s] = model.compute_surprise(logpost[:, :, :, s], 
                                                               loglik[:, :, :, s])
                    
                    # sample decisions from the computed log-posteriors
                    ch, rt = model.gen_response_from_logpost(logpost[:, :, :, s],
                            ndt=False, lapses=True)
                    # nan those values which are past the decisions
                    # (the returned RT is in real time; by dividing through the
                    # model's dt you get the first time bin past the decision,
                    # because index 0 in log_post is after dt in real time and
                    # only lapse trials can have rt below dt)
                    ind = np.rint(rt[:, 0] / model.dt).astype(int)
                    for tr, t in enumerate(ind):
                        logpost[t:, :, tr, s] = np.nan
                        loglik[t:, :, tr, s] = np.nan
                        surprise[t:, tr, s] = np.nan
                    
                    # print progress
                    if ( np.floor(s / S * 100) < np.floor((s+1) / S * 100) ):
                        print('\r... %3d%% completed' % (np.floor((s+1) / S * 100)), end='');
                print('')
                
                # expected values
                dot_level.loc[sub, 'logpost_left'] = np.nanmean(
                        logpost[:, 0, :, :], axis=2).flatten('F')
                dot_level.loc[sub, 'logpost_right'] = np.nanmean(
                        logpost[:, 1, :, :], axis=2).flatten('F')
                dot_level.loc[sub, 'loglik_left'] = np.nanmean(
                        loglik[:, 0, :, :], axis=2).flatten('F')
                dot_level.loc[sub, 'loglik_right'] = np.nanmean(
                        loglik[:, 1, :, :], axis=2).flatten('F')
                dot_level.loc[sub, 'm_evidence_left'] = np.nanmean(
                        loglik[:, 0, :, :] - loglik[:, 1, :, :], 
                        axis=2).flatten('F')
                dot_level.loc[sub, 'surprise'] = np.nanmean(surprise, 
                        axis=2).flatten('F')
                
                # set all those averages to NaN which are only based on less
                # than Smin values
                ind = (S - np.isnan(surprise).sum(axis=2)) < Smin
                # NOTE that this provides a view of dot_level for subject sub
                dot_level_sub = dot_level.loc[sub]
                # this change will affect dot_level
                dot_level_sub.loc[ind.flatten('F'), :] = np.nan
                
                store['dot_level'] = dot_level
    
    return dot_level


def get_entropies(fname='infres_collapse_201612121759.npz', 
                  resultsdir=helpers.resultsdir, S=15000, entropy_bins=50):
    """Returns posterior model entropies for each subject and trial.
    
        Posterior entropies measure the uncertainty that the model has about
        the response of the subject on that trial, i.e., if the model predicts
        a wide range of responses in that trial, the entropy will be large.
        
        If a corresponding cache file exists, the entropies are loaded and 
        returned from there.
        
        Parameters
        ----------
        fname : str, default 'infres_collapse_201612121759.npz'
            file name of model inference result to be used as basis
        resultsdir : str, default helpers.resultsdir
            path to results directory where fname lies
        S : int, default 15000
            number of response samples per trial for estimating entropy,
            ignored when entropies loaded from file
        entropy_bins : int, default 50
            number of bins in the response histogram to estimate response
            probabilities used in estimating entropy,
            ignored when entropies loaded from file
        
        Returns
        -------
        entropies : ndarray
            estimated posterior entropies
            shape = (number of subjects, number of trials)
            in order stored in model in corresponding inference result
    """
    
    
    ent_fname = os.path.join(resultsdir, fname[:-4]+'_entropies.npz')
    
    if os.path.isfile(ent_fname):
        with np.load(ent_fname) as data:
            entropies = data['entropies']
            S = data['S'][()]
            entropy_bins = data['entropy_bins'][()]
            
        print('Loaded entropies from file (S=%d, entropy_bins=%d)!' %
              (S, entropy_bins))
    else:
        with np.load(os.path.join(resultsdir, fname)) as data:
            subjects = data['subjects']
            model = data['model'][()]
            pars = data['pars'][()]
            ep_means = data['ep_means']
            ep_covs = data['ep_covs']
            
        entropies = np.zeros((subjects.size, model.L))
        for si, sub in enumerate(subjects):
            print('subject %d ...' % sub, flush=True)
            for tri, tr in enumerate(np.arange(model.L)):
                # sample from posterior
                choices, rts = model.gen_response_from_Gauss_posterior(tr, 
                    pars.names, ep_means[si, :], ep_covs[si, :, :], S, pars.transform)
                
                # estimate entropy
                entropies[si, tri] = model.estimate_entropy(choices, rts, entropy_bins)
                
            np.savez_compressed(os.path.join(resultsdir, fname[:-4]+'_entropies.npz'), 
                                entropies=entropies, S=S, entropy_bins=entropy_bins)
            
    return entropies


# run as script, if called
if __name__ == '__main__':
    entropies = get_entropies()
    dot_level = get_dot_level_measures()