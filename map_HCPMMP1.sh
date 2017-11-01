#!/bin/bash

fsavdir=$SUBJECTS_DIR/fsaverage/Glasser
wb=$fsavdir/connectome_workbench/bin_linux64/wb_command
datadir=$fsavdir/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k
meshdir=$fsavdir/standard_mesh_atlases/resample_fsaverage
outdir=$fsavdir/out

label=$datadir/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii
oldsph=$meshdir/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii
newsph=$meshdir/fsaverage_std_sphere.L.164k_fsavg_L.surf.gii
oldarea=$meshdir/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii
newarea=$meshdir/fsaverage.L.midthickness_va_avg.164k_fsavg_L.shape.gii

# separate out labels from cifti
$wb -cifti-separate $label COLUMN -label CORTEX_LEFT $outdir/tmp.label.gii

# tranform labels into fsaverage164
$wb -label-resample $outdir/tmp.label.gii $oldsph $newsph ADAP_BARY_AREA $outdir/tmp.fsaverage164.label.gii -area-metrics $oldarea $newarea

# convert into freesurfer annot-file
mris_convert --annot $outdir/tmp.fsaverage164.label.gii $newsph $outdir/lh.HCPMMP1_5_8.annot

#rm $outdir/tmp.label.gii $outdir/tmp.fsaverage164.label.gii

label=$datadir/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii
oldsph=$meshdir/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii
newsph=$meshdir/fsaverage_std_sphere.R.164k_fsavg_R.surf.gii
oldarea=$meshdir/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii
newarea=$meshdir/fsaverage.R.midthickness_va_avg.164k_fsavg_R.shape.gii

# separate out labels from cifti
$wb -cifti-separate $label COLUMN -label CORTEX_RIGHT $outdir/tmp.label.gii

# tranform labels into fsaverage164
$wb -label-resample $outdir/tmp.label.gii $oldsph $newsph ADAP_BARY_AREA $outdir/tmp.fsaverage164.label.gii -area-metrics $oldarea $newarea

# convert into freesurfer annot-file
mris_convert --annot $outdir/tmp.fsaverage164.label.gii $newsph $outdir/rh.HCPMMP1_5_8.annot

#rm $outdir/tmp.label.gii $outdir/tmp.fsaverage164.label.gii

read -p "Enter to close ..." response
