#!/bin/bash

fsavdir=$SUBJECTS_DIR/fsaverage/Glasser
wb=$fsavdir/connectome_workbench/bin_linux64/wb_command
datadir=$fsavdir/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k
meshdir=$fsavdir/standard_mesh_atlases
outdir=$fsavdir/out

# separate out labels from cifti
$wb -cifti-separate $datadir/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii COLUMN -label CORTEX_LEFT $outdir/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.label.gii

# tranform labels into fsaverage164
$wb -label-resample $outdir/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.label.gii $meshdir/L.sphere.32k_fs_LR.surf.gii $meshdir/fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii BARYCENTRIC $outdir/left.fsaverage164.label.gii

# convert into freesurfer annot-file
mris_convert --annot $outdir/left.fsaverage164.label.gii $meshdir/fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii $outdir/lh.HCP-MMP1.annot

read -p "Enter to close ..." response
