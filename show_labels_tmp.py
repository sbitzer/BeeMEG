import source_visualisations as sv
import pandas as pd
import mne

srcfile = 'source_HCPMMP1_allsubs_201703301614_slabs_dot_x.h5'
file = 'mne_subjects/fsaverage/bem/' + srcfile
src_df = pd.read_hdf(file, 'second_level_src')

labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', hemi='both')
labels = {l.name: l for l in labels}

brain = sv.show_labels(srcfile, 'mu_p_large', src_df=src_df, labels=labels)

show = lambda time: sv.show_labels(srcfile, 'mu_p_large', src_df=src_df, 
                                   labels=labels, brain=brain, time=time)
