#!/bin/bash

fsavdir=$SUBJECTS_DIR/fsaverage/Glasser
wb=$fsavdir/connectome_workbench/bin_linux64/wb_command
datadir=$fsavdir/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k
meshdir=$fsavdir/standard_mesh_atlases/resample_fsaverage
outdir=$fsavdir/out

for hemi in L R
do
	label=$datadir/Q1-Q6_RelatedParcellation210.$hemi.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii
	oldsph=$meshdir/fs_LR-deformed_to-fsaverage.$hemi.sphere.32k_fs_LR.surf.gii
	newsph=$meshdir/fsaverage_std_sphere.$hemi.164k_fsavg_$hemi.surf.gii
	oldarea=$meshdir/fs_LR.$hemi.midthickness_va_avg.32k_fs_LR.shape.gii
	newarea=$meshdir/fsaverage.$hemi.midthickness_va_avg.164k_fsavg_$hemi.shape.gii

	# separate out labels from cifti
	if [ "$hemi" == "L" ]; then
		$wb -cifti-separate $label COLUMN -label CORTEX_LEFT $outdir/tmp.label.gii
	else
		$wb -cifti-separate $label COLUMN -label CORTEX_RIGHT $outdir/tmp.label.gii
	fi

	# tranform labels into fsaverage164
	$wb -label-resample $outdir/tmp.label.gii $oldsph $newsph ADAP_BARY_AREA $outdir/tmp.fsaverage164.label.gii -area-metrics $oldarea $newarea

	# convert into freesurfer annot-file
	mris_convert --annot $outdir/tmp.fsaverage164.label.gii $newsph $outdir/${hemi,,}h.HCPMMP1_5_8.annot

	# copy to fsaverage label directory
	cp $outdir/${hemi,,}h.HCPMMP1_5_8.annot $fsavdir/../label/
done

rm $outdir/tmp.label.gii $outdir/tmp.fsaverage164.label.gii

