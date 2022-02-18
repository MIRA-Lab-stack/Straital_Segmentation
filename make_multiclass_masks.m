% folders = {'validation','training','testing'};
 rois = {'*anp*hdr',  '*dca*hdr', '*pca*hdr', '*pop*hdr', '*vst*hdr'};
% 
% for i = 1 : numel(folders)
%     cd(folders{i})
    
    pid = dir('study*'); 
    pid = [pid; dir('0*')]; 
    pid = [pid; dir('1*')];
    pid = [pid; dir('2*')];
    pid = [pid; dir('3*')];
    pid = [pid; dir('MR*')];
    pid
    
    for j = 1 : numel(pid)
        cd(pid(j).name)
        pwd
        aa=load_untouch_nii('kds_striatum_mask');
        aa.img=zeros(size(aa.img));
        for k = 1:numel(rois)
            tmp = dir(rois{k})
            for l = 1:numel(tmp)
           
                tmp_mri = load_untouch_nii(tmp(l).name);
                aa.img(tmp_mri.img>0)=k;
            end
        end
        save_untouch_nii(aa,'kds_mask_multiclass')
        cd ..
    end
    cd ..
% end
