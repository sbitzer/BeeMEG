#!/bin/bash

fsavdir=$SUBJECTS_DIR/fsaverage/Glasser
wb=$fsavdir/connectome_workbench/bin_linux64/wb_command
datadir=$fsavdir/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k
meshdir=$fsavdir/standard_mesh_atlases/resample_fsaverage
outdir=$fsavdir/out

border=$datadir/Q1-Q6_RelatedParcellation210.L.SubAreas.32k_fs_LR.border
oldsphere=$meshdir/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii
newsphere=$meshdir/fsaverage_std_sphere.L.164k_fsavg_L.surf.gii
oldarea=$meshdir/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii
newarea=$meshdir/fsaverage.L.midthickness_va_avg.164k_fsavg_L.shape.gii

# turn borders into metric ROIs
$wb -border-to-rois $oldsphere $border $outdir/tmp.func.gii

# create table list file
$wb -border-export-color-table $border $outdir/label_list.txt

# turn metric ROIs into labels
$wb -metric-label-import $outdir/tmp.func.gii $outdir/color.tab $outdir/tmp.label.gii -discard-others

# combine all labels into one column
#$wb -label-merge $outdir/tmp_merged.label.gii -label $outdir/tmp.label.gii

# tranform labels into fsaverage164
$wb -label-resample $outdir/tmp.label.gii $oldsphere $newsphere ADAP_BARY_AREA $outdir/tmp.fsaverage164.label.gii -area-metrics $oldarea $newarea

# convert into freesurfer annot-file
mris_convert --annot $outdir/tmp.fsaverage164.label.gii $newsphere $outdir/lh.HCPMMP1_motor.annot

read -p "Enter to close ..." response
