#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:19:02 2018

@author: bitzer
"""

from __future__ import print_function
import helpers
import source_visualisations as sv
import source_statistics as ss
import numpy as np
import pandas as pd
import os
import mne
import scipy.stats

try:
    from surfer import Brain
except:
    pass

figdir = helpers.figdir


#%% some options
# newest results are based on this parcellation, cf. 'srcfile' in basefile
parc = 'HCPMMP1_5_8'

r_name = 'response'

if r_name == 'response':
    # loose source orientations, but cortex normal currents
    # baseline (-0.3, 0), response-aligned
    # timewins build-up: [-500, -120], response: [-30, 100], 
    # exclude time-outs, trial_time, intercept, response
    # trial normalisation of data, local normalisation of DM
    basefile = 'source_singledot_201808291410.h5'

show_measure = 'abstval'

fdr_alpha = 0.01


#%% determine statistically significant effects
sl = pd.read_hdf(os.path.join(
        helpers.resultsdir, basefile), 'second_level').loc[0]

winnames = sl.index.get_level_values('time').unique()

ss.add_measure(sl, 'mlog10p_fdr')

sig = sl[sl[('mlog10p_fdr', r_name)] > -np.log10(fdr_alpha)].xs(
        r_name, level='regressor', axis=1)

srcdf = sl.xs(r_name, level='regressor', axis=1).copy()
srcdf[srcdf['mlog10p_fdr'] < -np.log10(fdr_alpha)] = 0
if show_measure not in srcdf.columns:
    ss.add_measure(srcdf, show_measure)
    

#%% define some plotting functions
def get_colorinfo(srcdf, measure):
    srcdf = srcdf[(srcdf[measure] != 0) & srcdf[measure].notna()]
    
    return {'fmin': srcdf[measure].abs().min(),
            'fmid': srcdf[measure].abs().median(), 
            'fmax': srcdf[measure].abs().max(), 
            'transparent': True,
            'colormap': 'auto'}
    
    
def make_brain_figures(srcdf, measure, cinfo, filepat_base, top=5):
    views = {'rh': ['medial', 'lateral'],
             'lh': ['lateral', 'medial']}
    
    toplabels = {}
    
    for hemi in ['lh', 'rh']:
        brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                      subjects_dir=sv.subjects_dir, background='w', 
                      foreground='k')
            
        sv.show_labels_as_data(srcdf, measure, brain, 
                               time_label=None, parc=parc, **cinfo)
        
        if top:
            ind = srcdf.index.get_level_values('label').map(
                    lambda x: x.startswith(hemi[0].upper()))
            
            topl = srcdf[ind].abs().sort_values(
                    measure, ascending=False).head(top)
            topl = (
                    topl[(topl[measure] != 0) 
                              & topl[measure].notna()]
                    .index.droplevel('time'))
            toplabels[hemi] = topl
            
            alllabels = mne.read_labels_from_annot(
                    'fsaverage', parc=parc, hemi=hemi)
            
            brain.remove_labels()
            for label in alllabels:
                if label.name in topl:
                    brain.add_label(label, borders=1, hemi=hemi, alpha=0.8, 
                                    color='k')
        
        # increase font size of colorbar - this only works by increasing the 
        # colorbar itself and setting the ratio of colorbar to text
        brain.data['colorbar'].scalar_bar.bar_ratio = 0.35
        brain.data['colorbar'].scalar_bar_representation.position = [0.075, 0.01]
        brain.data['colorbar'].scalar_bar_representation.position2 = [0.85, 0.12]
        
        filepat = os.path.join(figdir, filepat_base + '_{}.png'.format(hemi))
        
        brain.save_montage(filepat, views[hemi], colorbar=0)
        
        brain.close()
        
    infiles = [
            os.path.join(figdir, filepat_base + '_{}.png'.format(hemi))
            for hemi in ['lh', 'rh']]
    outfile = os.path.join(figdir, filepat_base + '.png')
    os.system("montage -tile 2x1 -geometry +0+0 {} {}".format(
            ' '.join(infiles), outfile))
    
    return toplabels


#%% show results in brain
interactive = False
normalise = True

toplstr = lambda ls: ', '.join([l[2:-7] for l in ls])

srcdf_normed = srcdf.copy()
if normalise:
    for wname in winnames:
        normed, _, _ = helpers.normalise_magnitudes(
                srcdf.loc[(slice(None), wname), :].dropna())
        srcdf_normed.loc[normed.index] = normed

if interactive:
    hemi = 'lh'
    wname = 'build-up'
    
    brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                  subjects_dir=sv.subjects_dir, background='w', foreground='k')
    
    sv.show_labels_as_data(srcdf_normed.loc[(slice(None), wname), :], 
                           show_measure, brain, time_label=None, parc=parc,
                           **get_colorinfo(srcdf, show_measure))
else:
    toplstrs = []
    for wname in winnames:
        filepat_base = 'win_brain_{}_{}_{}'.format(r_name, show_measure, wname)
        
        df = srcdf_normed.loc[(slice(None), wname), :]
        if normalise:
            cinfo = get_colorinfo(srcdf_normed, show_measure)
        else:
            cinfo = get_colorinfo(df, show_measure)
            
        toplabels = make_brain_figures(df, show_measure, cinfo, filepat_base)
        
        toplstrs.append('{}: left - {}; right - {}'.format(
                wname, toplstr(toplabels['lh']), toplstr(toplabels['rh'])))


#%% investigate differences

# whether to shift the magnitudes before scaling
shift = True
        
data = pd.read_hdf(os.path.join(helpers.resultsdir, basefile), 'first_level')
data = data.loc[0].xs('beta', level='measure', axis=1).xs(
        r_name, level='regressor', axis=1).copy()

# scale data so that for each time point and subject the mean magnitude of
# betas across brain areas with significant effects is 1
magnitudes = data.unstack('time').abs()
siglabels = sig.index.get_level_values('label').unique()
#siglabels = magnitudes.index
#siglabels = ['L_4_ROI-lh', 'R_4_ROI-rh']

zero = magnitudes.iloc[0].copy()
scale = magnitudes.iloc[0].copy()
for wname in winnames:
    sigl = sig.xs(wname, level='time').index
    
    _, zero.loc[(slice(None), wname)], scale.loc[(slice(None), wname)] = (
            helpers.normalise_magnitudes(magnitudes.loc[
                    sigl, (slice(None), wname)]))

datanorm = (magnitudes - zero) / scale
datanorm.loc[magnitudes.index.difference(siglabels)] = 0
datanorm = datanorm.stack('time')

# calculate magnitude difference
diff = (datanorm.xs(winnames[0], level='time')
        - datanorm.xs(winnames[1], level='time'))

# perform t-test
tvals, pvals = scipy.stats.ttest_1samp(diff, 0, axis=1)

# put results into standard container
srcdfdiff = pd.DataFrame(
        np.c_[diff.mean(axis=1), diff.std(axis=1), tvals, pvals],
        index=pd.MultiIndex.from_product(
                [diff.index, [0]],
                names=['label', 'time']),
        columns=['mean', 'std', 'tval', 'pval'])

# determine significant differences with FDR correction
srcdfdiff['mlog10p'] = -np.log10(srcdfdiff['pval'])
srcdfdiffsig = srcdfdiff.loc[list(siglabels), :].copy()
ss.add_measure(srcdfdiffsig, 'mlog10p_fdr')

# zero-out non-significant differences
srcdfdiff.loc[srcdfdiffsig[
        srcdfdiffsig['mlog10p_fdr'] < -np.log10(fdr_alpha)].index] = 0


#%% show in brain
diff_show_measure = 'tval'
if interactive:
    if 'brain' not in locals():
        hemi = 'rh'
        brain = Brain(
                'fsaverage', hemi, 'inflated', cortex='low_contrast',
                 subjects_dir=sv.subjects_dir, background='w', foreground='k')
    sv.show_labels_as_data(srcdfdiff, diff_show_measure, brain, 
                           time_label=None, parc=parc, center=0);
else:
    filepat_base = 'windiff_brain_{}_{}_{}-{}'.format(
            r_name, diff_show_measure, winnames[0], winnames[1])
    cinfo = get_colorinfo(srcdfdiff, diff_show_measure)
    cinfo['center'] = 0
    make_brain_figures(srcdfdiff, diff_show_measure, cinfo, filepat_base, 
                       top=0);
                       
print(', '.join(srcdfdiff[srcdfdiff['mean'] != 0].index
      .get_level_values('label').map(lambda l: l[2:-7]).unique()))