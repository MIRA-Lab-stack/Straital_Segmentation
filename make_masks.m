rois={'anp_l', 'anp_r', 'dca_l', 'dca_r', 'pca_l', 'pca_r', 'pop_l', 'pop_r', 'vst_l', 'vst_r'}
studies = dir('MR*');

for i = 1 : numel(studies)
    
    i/numel(studies)

    cd(studies(i).name)
    delete *acin*
    delete r*
    delete *T2*
    delete *cer*
    delete *gen*
    delete *hip*
    delete *3mm*
    delete *amy*
    delete *ent*
    delete *med*
    delete *par*
    delete *mid_c*
    delete *orb*
    delete kds*
    delete *dor*
    delete *tem*
    delete *orb*
    tmp=dir('*crop*hdr');
    if isempty(tmp)
        tmp=dir('*rot*hdr');
    end
    tmp=load_untouch_nii(tmp(1).name);
    mask=zeros(size(tmp.img)); mask=int16(mask);
    
    for j = 1:numel(rois)
        
        tmp_roi=dir(['*'  rois{j}  '*hdr']);
        tmp_roi=load_untouch_nii(tmp_roi(1).name);
        mask=mask+int16(tmp_roi.img);
        
    end
    
    mask(mask>0)=1;
    tmp.img=mask;
    save_untouch_nii(tmp, 'kds_striatum_mask.nii')
    cd ..
    
end