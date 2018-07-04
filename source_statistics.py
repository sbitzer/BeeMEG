#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:28:44 2017

@author: bitzer
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import itertools
import os
import seaborn as sns
from scipy.interpolate import interp1d
import scipy.stats
import math
import mne
from warnings import warn
import matplotlib.pyplot as plt


if os.name == 'posix':
    subjects_dir = 'mne_subjects'
else:
#    subjects_dir = os.path.expanduser('~\\BeeMEG\\mne_subjects')
    subjects_dir = 'E:\bitzer\mne_subjects'
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')
inf_dir = os.path.join('data', 'inf_results')

Glasser_sections = pd.read_csv('Glasser2016_sections.csv', index_col=0)
Glasser_areas = pd.read_csv('Glasser2016_areas.csv', index_col=0)

basefile_measures = ['mean', 'absmean', 'mlog10p', 'std', 'tval', 'abstval',
                     'mlog10p_fdr', 'p_fdr']

# the number of subjects with suitable MEG data
S = 34

def get_Glasser_section(area):
    if area.startswith('??'):
        section = 'whole hemisphere'
    else:
        section = Glasser_areas[Glasser_areas['area name'] == area[2:-7]]['main section']
        section = Glasser_sections.loc[section].name.values[0]
    
    return section


def add_measure(second_level, measure):
    if isinstance(second_level.columns, pd.MultiIndex):
        for r_name in second_level.columns.levels[1]:
            if measure == 'tval':
                second_level[(measure, r_name)] = (
                          second_level[('mean', r_name)] 
                        / second_level[('std', r_name)] * np.sqrt(S))
            elif measure == 'abstval':
                second_level[(measure, r_name)] = (
                          second_level[('mean', r_name)] 
                        / second_level[('std', r_name)] * np.sqrt(S)).abs()
            elif measure == 'absmean':
                second_level[(measure, r_name)] = (
                        second_level[('mean', r_name)].abs())
            elif measure == 'mlog10p_fdr':
                _, pval = mne.stats.fdr_correction(
                        10**(-second_level[('mlog10p', r_name)]), 0.01)
                second_level[(measure, r_name)] = -np.log10(pval)
            elif measure == 'p_fdr':
                _, pval = mne.stats.fdr_correction(
                        10**(-second_level[('mlog10p', r_name)]), 0.01)
                second_level[(measure, r_name)] = pval
            else:
                raise ValueError("You try to add an unknown basefile measure!")
    else:
        if measure == 'tval':
            second_level[measure] = (
                      second_level['mean'] 
                    / second_level['std'] * np.sqrt(S))
        elif measure == 'abstval':
            second_level[measure] = (
                      second_level['mean'] 
                    / second_level['std'] * np.sqrt(S)).abs()
        elif measure == 'absmean':
            second_level[measure] = second_level['mean'].abs()
        elif measure == 'mlog10p_fdr':
                _, pval = mne.stats.fdr_correction(
                        10**(-second_level['mlog10p']), 0.01)
                second_level[measure] = -np.log10(pval)
        elif measure == 'p_fdr':
                _, pval = mne.stats.fdr_correction(
                        10**(-second_level['mlog10p']), 0.01)
                second_level[measure] = pval
        else:
            raise ValueError("You try to add an unknown basefile measure!")


def find_slabs_threshold(basefile, measure='mu_p_large', quantile=0.99, 
                         bemdir=bem_dir, regressors=None, 
                         exclude=None, verbose=2, return_cdf=False,
                         use_basefile=None, perm=0):
    """Finds a value threshold for a measure corresponding to a given quantile.
    
        Pools values of the measure across all fitted regressors. Then finds 
        the value corresponding to the given quantile.
    """
    
    if use_basefile is None:
        use_basefile = measure in basefile_measures
    
    if regressors is None:
        try:
            with pd.HDFStore(os.path.join(bemdir, basefile), 'r') as store:
                regressors = store.first_level_src.columns.levels[1]
        except (FileNotFoundError, OSError):
            with pd.HDFStore(os.path.join(inf_dir, basefile), 'r') as store:
                regressors = store.first_level.columns.levels[2]
    
    if exclude:
        if exclude == 'trialregs':
            exclude = ['intercept', 'entropy', 'response', 'trial_time', 'motoprep']
        elif exclude == 'dotregs':
            exclude = ['abs_dot_x', 'abs_dot_y', 'accev', 'accev_cflip', 
                       'accsur_pca', 'dot_x', 'dot_x_cflip', 'dot_y', 'move_dist',
                       'sum_dot_y_prev']
        if verbose:
            print('excluding:')
            print('\n'.join(exclude), end='\n\n')
        
        regressors = np.setdiff1d(regressors, exclude)
    
    if use_basefile:
        with pd.HDFStore(os.path.join(inf_dir, basefile), 'r') as store:
            if verbose:
                for r_name in regressors:
                    print('adding ' + r_name)
            second_level = store.second_level.loc[
                    perm, (slice(None), list(regressors))]
            
        if measure not in second_level.columns.levels[0]:
            add_measure(second_level, measure)
        
        values = (second_level.loc[:, (measure, list(regressors))]
                  .values.flatten())
    else:
        values = np.array([], dtype=float)
        for r_name in regressors:
            if verbose:
                print('adding ' + r_name)
            
            fname = basefile[:-3] + '_slabs_' + r_name + '.h5'
            
            with pd.HDFStore(os.path.join(bemdir, fname), 'r') as store:
                values = np.r_[values, store.second_level_src[measure].values]
    
    if verbose:
        print('N = %d' % values.size)
    if verbose > 1:
        fig, ax = sns.plt.subplots()
        ax = sns.distplot(values, ax=ax)
    
    qval = np.percentile(values, quantile * 100)
    
    if verbose > 1:
        ax.plot(qval*np.r_[1, 1], ax.get_ylim())
    
    if return_cdf:
        values.sort()
        cdf = interp1d(values, np.arange(values.size) / float(values.size),
                       assume_sorted=True)
        return qval, cdf
    else:
        return qval
    
    
def get_time_clusters(measure_series, threshold, cdf=None):
    
    clusters = []
    
    for label in measure_series.index.levels[0]:
        label_tc = measure_series.xs(label, level='label')
        
        start = 0
        for key, group in itertools.groupby(label_tc >= threshold):
            length = sum(1 for _ in group)
            if key == True:
                if cdf is None:
                    logps = label_tc.values[start : start+length]
                else:
                    logps = np.log10(
                            1 - cdf(label_tc.values[start : start+length]))
                clusters.append([label, label_tc.index[start], 
                                 label_tc.index[start+length-1], logps.sum()])
                
            start += length
    
    return pd.DataFrame(clusters, 
                        columns=['label', 'start_t', 'end_t', 'log10p'])


def get_fdrcorr_clusters(basefile, regressors, fdr_alpha, measure=None, 
                         threshold=None, cdf=None, use_basefile=None, perm=0):
    
    if use_basefile is None:
        use_basefile = measure in basefile_measures
    
    if threshold is None:
        threshold = -np.log10(fdr_alpha)
        if measure is not None:
            warn("Specific measure was given, but no threshold. Will ignore "
                 "measure and cdf and compute clusters directly based on "
                 "FDR-corrected p-values.")
        measure = 'mlog10p_fdr'
        cdf = None
    
    if use_basefile:
        second_level = pd.read_hdf(os.path.join(inf_dir, basefile),
                                   'second_level')
        second_level = second_level.loc[perm]
        
        if measure not in second_level.columns.levels[0]:
            add_measure(second_level, measure)
        
        clusters = [
                get_time_clusters(second_level.loc[:, (measure, r_name)],
                                  threshold, cdf)
                for r_name in regressors]
    else:
        clusters = []
        for r_name in regressors:
            srcfile = basefile[:-3] + '_slabs_%s.h5' % r_name
            file = os.path.join(bem_dir, srcfile)
            src_df = pd.read_hdf(file, 'second_level_src')
            clusters.append(get_time_clusters(src_df[measure], threshold, cdf))
    
    clusters = pd.concat(clusters, keys=regressors, 
                         names=['regressor', 'cluster'])
    
    # if clusters were defined directly based on FDR-correct p-values
    if cdf is None:
        assert measure.startswith('mlog10p')
        # get_time_clusters returns the sum of mlog10p-values in the cluster
        # so these need to be multiplied by -1 to get log10p
        clusters['log10p'] = -clusters.log10p
        clusters['pval_corrected'] = 10 ** (clusters.log10p)
    # if clusters were defined based on an empirical cdf
    else:
        reject, pval = mne.stats.fdr_correction(10**clusters.log10p, fdr_alpha)
        clusters['pval_corrected'] = pval
        clusters = clusters[reject]
    
    try:
        clusters['region'] = clusters.label.apply(get_Glasser_section)
    except IndexError:
        print('No brain regions found for area, skipping that step.')
    
    return clusters
    

def apply_cluster_mask(src_df, clusters, regressor):
    """Set values in src_df to nan which do not belong to a cluster."""
    mask = np.ones(src_df.index.size, dtype=bool)
    
    # remove clusters from mask
    for cl in clusters.xs(regressor, level='regressor').itertuples():
        ind = src_df.index.get_locs((cl.label, slice(cl.start_t, cl.end_t)))
        mask[ind] = False
        
    src_df[mask] = np.nan
    

def load_src_df(basefile, regressor, clusters=None, use_basefile=False, perm=0):
    """Load src_df for given regressor and result and mask by clusters, if given."""
    if use_basefile:
        src_df = pd.read_hdf(os.path.join(inf_dir, basefile), 'second_level')
        src_df = src_df.loc[perm].xs(regressor, axis=1, level='regressor')
    else:
        file = os.path.join(bem_dir, basefile[:-3] + '_slabs_%s.h5' % regressor)
        src_df = pd.read_hdf(file, 'second_level_src')
    
    if clusters is not None:
        apply_cluster_mask(src_df, clusters, regressor)
    
    return src_df


def null_inconsistent(values, v_threshold, s_threshold=3):
    """Sets all values to 0 that do not belong to a sequence of length 
    s_threshold of values above v_threshold."""
    mask = np.zeros_like(values, dtype=bool)
    
    start = 0
    for key, group in itertools.groupby(values >= v_threshold):
        length = sum(1 for _ in group)
        if key == True and length >= s_threshold:
            mask[start : start+length] = True
        start += length
    
    values = values.copy()
    
    # if the value threshold is below 0, then values were log-transformed
    # so that 0 becomes -inf
    if v_threshold < 0:
        values[~mask] = -np.inf
    else:
        values[~mask] = 0
    
    return values


def null_inconsistent_measure(vseries, v_threshold, s_threshold=3):
    """Applies null_inconsistent to all labels in the series, but not across 
    labels."""
    vseries = vseries.reset_index(level='label')
    vdf = vseries.pivot(columns='label')
    
    vdf = vdf.apply(lambda x: null_inconsistent(x, v_threshold, s_threshold))
    vdf = vdf.stack().reorder_levels(['label', 'time']).sort_index()
    
    return vdf[vdf.columns[0]]


def get_significant_areas_in_time(measure_series, threshold, timewindow):
    values = null_inconsistent_measure(measure_series, threshold)
    values_tmp = values.loc[(slice(None), slice(*timewindow))]
    values_tmp = values_tmp.reset_index(level='label').pivot(columns='label')
    labels = values_tmp.columns.get_level_values('label')[values_tmp.sum() > 0]
    del values_tmp
    
    areas = pd.Series(np.unique(labels.map(lambda l: l[2:-7])))
    
    area_values = pd.DataFrame(
            [], dtype=float, index=values.index.levels[1], columns=
            pd.MultiIndex.from_product([['L', 'R'], areas], 
                                       names = ['hemi', 'region']))

    for hemi in ['l', 'r']:
        # create full label names
        labels = areas.map(
                lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
        
        # set values of all areas to that of aggregated value
        avals = values.loc[(labels, slice(None))]
        avals = avals.reset_index(level='time').pivot(columns='time')
        area_values.loc[:, (hemi.upper(), slice(None))] = avals.values.T

    return area_values.drop(area_values.columns[area_values.sum() == 0], axis=1)


def get_significant_areas_in_region(measure_series, threshold, region):
    areas = Glasser_areas[Glasser_areas['main section'] == region]
    areas = areas.sort_values('area name')
    
    values = null_inconsistent_measure(measure_series, threshold)
    
    area_values = pd.DataFrame(
            [], dtype=float, index=values.index.levels[1], columns=
            pd.MultiIndex.from_product([['L', 'R'], areas['area name']], 
                                       names = ['hemi', 'region']))

    for hemi in ['l', 'r']:
        # create full label names
        labels = areas['area name'].map(
                lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
        
        # set values of all areas to that of aggregated value
        avals = values.loc[(labels, slice(None))]
        avals = avals.reset_index(level='time').pivot(columns='time')
        area_values.loc[:, (hemi.upper(), slice(None))] = avals.values.T

    return area_values.drop(area_values.columns[area_values.sum() == 0], axis=1)


def get_significant_regions(measure_series, threshold, 
                            region_aggfun=lambda a: np.max(a, axis=0)):
    values = null_inconsistent_measure(measure_series, threshold)
    
    regions = pd.DataFrame(
            [], dtype=float, index=values.index.levels[1], columns=
            pd.MultiIndex.from_product([['L', 'R'], Glasser_sections.name], 
                                       names = ['hemi', 'region']))
    for section in Glasser_sections.index:
        # find all areas in region
        areas = Glasser_areas[Glasser_areas['main section'] == section]
        
        for hemi in ['l', 'r']:
            # create full label names
            labels = areas['area name'].map(
                    lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
            
            # set values of all areas to that of aggregated value
            avals = values.loc[(labels, slice(None))]
            avals = avals.reset_index(level='time').pivot(columns='time')
            regions.loc[:, (hemi.upper(), Glasser_sections.loc[section, 'name'])] = (
                    region_aggfun(avals).values)
    
    return regions.drop(regions.columns[regions.sum() == 0], axis=1)


#%% statistical test for difference between correlations
#   taken from http://www.philippsinger.info/?p=347
#   https://github.com/psinger/CorrelationStats
#   see also:
# [4] J. H. Steiger, "Tests for comparing elements of a correlation matrix.," 
#     Psychological bulletin, vol. 87, iss. 2, p. 245, 1980.
# [5] G. Y. Zou, "Toward using confidence intervals to compare correlations.," 
#     Psychological methods, vol. 12, iss. 4, pp. 399-413, 2007. 
def rz_ci(r, n, conf_level = 0.95):
    zr_se = pow(1/(n - 3), .5)
    moe = scipy.stats.norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
    zu = math.atanh(r) + moe
    zl = math.atanh(r) - moe
    return np.tanh((zl, zu))

def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy,2)-pow(rxz,2)-pow(ryz,2))+pow(ryz,3)
    den = (1 - pow(rxy,2)) * (1 - pow(rxz,2))
    return num/float(den)

def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, 
                   method='steiger'):
    """
    Calculates the statistical significance for the null hypothesis that the
    difference between correlations xy - xz = 0 given that variables y and z 
    are themselves correlated
    
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, 
           only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz)/2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin 
                         + av * av * cube)))
        p = 1 - scipy.stats.t.cdf(abs(t2), n - 3)

        if twotailed:
            p *= 2

        return t2, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow(
                (pow((xy - L1), 2) + pow((U2 - xz), 2) 
                 - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
        upper = xy - xz + pow(
                (pow((U1 - xy), 2) + pow((xz - L2), 2) 
                - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')
        
        
#%% sparse Gaussian process analyses for functional relationships
try: 
    import GPy
except ImportError:
    print('No GPy found. Skipping definition of GP functions in '
          'source_statistics.')
else:
    def plot_gp_result(ax, xpred, gpm, color='C0', label='GP prediction', 
                       return_significant=False, xx=None, coverage=0.95):
        if xx is None:
            xx = xpred
            
        gpmean, gpvar = gpm.predict_noiseless(xpred)
        
        gpmean = gpmean[:, 0]
        gpstd = np.sqrt(gpvar[:, 0])
        
        lw = plt.rcParams["lines.linewidth"] * 1.5
        
        stdmult = scipy.stats.norm.ppf([0.5 - coverage/2, 0.5 + coverage/2])
        lower = gpmean + stdmult[0] * gpstd
        upper = gpmean + stdmult[1] * gpstd
        
        # Draw the regression line and confidence interval
        ax.plot(xx, gpmean, color=color, lw=lw, label=label)
        ax.fill_between(xx[:, 0], lower, upper, facecolor=color, alpha=.15)
        
        if return_significant:
            return xpred[(lower > 0) | (upper < 0)]
        
        
    def fitgp(xvals, data, Zvar=None, biasstd=1.0, smooth=True, gptype='sparse'):
        if data.ndim < 2:
            data = data[:, None]
        if xvals.ndim < 2:
            xvals = xvals[:, None]
        if Zvar.ndim < 2:
            Zvar = Zvar[:, None]
        
        # choose basic covariance function, the squared exponential (i.e. RBF) is 
        # known to be very smooth while the Matern-class allows for local bumps;
        # the global features / overall shape of the function should be very 
        # similar, though
        if smooth:
            kern = GPy.kern.RBF(1)
        else:
            kern = GPy.kern.Matern32(1)
        
        # reasonable initial value for lengthscale: half of the xrange
        kern.lengthscale = (xvals.max() - xvals.min()) / 2
        
        if biasstd:
            # the bias kernel adds a constant to the covariance function which
            # corresponds to adding an offset/intercept to the underlying function
            # note however, that if the underlying linear model is y = f(x) + b, 
            # then b has a Gaussian prior with b ~ N(0, kern.Bias.variance); thus, 
            # you cannot interpret kern.Bias.variance as the fitted intercept!
            kern += GPy.kern.Bias(1, biasstd ** 2)
        
        if gptype == 'sparse':
            gpm = GPy.models.SparseGPRegression(xvals, data, kernel=kern, 
                                                Z=Zvar)
            
            # set noise variance to a reasonable initial value for optimisation
            gpm.likelihood.variance = data.var()
            
            # do not optimise inducing inputs (makes no big difference at least for 
            # comparison of dot_x and sum_dot_x in individual subjects, or pooled, 
            # because the inducing inputs move only minimally from their initial 
            # values and the results are very similar)
#            gpm.Z.fix()
        elif gptype == 'heteroscedastic':
            gpm = GPy.models.GPHeteroscedasticRegression(xvals, data,
                                                         kernel=kern)
            gpm.het_Gauss.variance = Zvar
            gpm.het_Gauss.variance.fix()
        
        gpm.optimize()
        
        return gpm
        
    
    def gpregplot(r_name, label, rdata, x_estimator=None, x_bins=10, ax=None, 
                  plot_kws=None, scatter_kws=None, xrange=[-1.5, 1.5], 
                  gp_kws={'gptype': 'sparse'}):
        
        if gp_kws['gptype'] == 'sparse':
            X = rdata[r_name]
            Y = rdata[label]
            Zvar = np.linspace(*xrange, num=15)
        elif gp_kws['gptype'] == 'heteroscedastic':
            # get mean and variance of data points associated with distinct x
            grouped = rdata.groupby(r_name)
            
            Y = grouped[label].mean()
            X = Y.index.values
            Zvar = grouped[label].var() / grouped[label].count()
            
        if not 'return_significant' in plot_kws.keys():
            plot_kws['return_significant'] = False
        
        gpm = fitgp(X, Y, Zvar, **gp_kws)
        
        xpred = np.linspace(*xrange, num=200)[:, None]
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # plot regression function
        significant = plot_gp_result(ax, xpred, gpm, **plot_kws)
        
        # scatterplot
        bin_edges = np.linspace(*xrange, num=x_bins+1)
        binx = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax = sns.regplot(r_name, label, rdata, x_estimator=np.mean, x_bins=binx, 
                         ax=ax, scatter_kws=scatter_kws, fit_reg=False)
        
        if plot_kws['return_significant']:
            return gpm, significant
        else:
            return gpm