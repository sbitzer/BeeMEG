#!/bin/bash

fsavdir=$SUBJECTS_DIR/fsaverage/Glasser
wb=$fsavdir/connectome_workbench/bin_linux64/wb_command
datadir=$fsavdir/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k
meshdir=$fsavdir/standard_mesh_atlases/resample_fsaverage
outdir=$fsavdir/out

for hemi in L R
do
	border=$datadir/Q1-Q6_RelatedParcellation210.$hemi.SubAreas.32k_fs_LR.border
	oldsphere=$meshdir/fs_LR-deformed_to-fsaverage.$hemi.sphere.32k_fs_LR.surf.gii
	newsphere=$meshdir/fsaverage_std_sphere.$hemi.164k_fsavg_$hemi.surf.gii
	oldarea=$meshdir/fs_LR.$hemi.midthickness_va_avg.32k_fs_LR.shape.gii
	newarea=$meshdir/fsaverage.$hemi.midthickness_va_avg.164k_fsavg_$hemi.shape.gii

	# turn borders into metric ROIs
	$wb -border-to-rois $oldsphere $border $outdir/rois.func.gii

	# combine columns for different ROIs into a single column with integer indices
	$wb -metric-reduce $outdir/rois.func.gii INDEXMAX $outdir/indexmax.func.gii
	$wb -metric-reduce $outdir/rois.func.gii MAX $outdir/max.func.gii
	$wb -metric-mask $outdir/indexmax.func.gii $outdir/max.func.gii $outdir/integers.func.gii

	# turn metric ROIs into labels
	$wb -metric-label-import $outdir/integers.func.gii HCP_motor_color.tab $outdir/tmp.label.gii

	# tranform labels into fsaverage164
	$wb -label-resample $outdir/tmp.label.gii $oldsphere $newsphere ADAP_BARY_AREA $outdir/tmp.fsaverage164.label.gii -area-metrics $oldarea $newarea

	# convert into freesurfer annot-file
	mris_convert --annot $outdir/tmp.fsaverage164.label.gii $newsphere $outdir/${hemi,,}h.HCPMMP1_motor.annot

	# copy to fsaverage label directory
	cp $outdir/${hemi,,}h.HCPMMP1_motor.annot $fsavdir/../label/
done

rm $outdir/rois.func.gii $outdir/indexmax.func.gii $outdir/max.func.gii $outdir/integers.func.gii $outdir/tmp.label.gii $outdir/tmp.fsaverage164.label.gii

