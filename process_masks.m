%this scruipt will take the TF output for Jared's dataset, where
%each channel is 1 roi prediction, and will make it into a normal mask
cwd = pwd;
data_dir = '/media/mira/Data/karl/striatum/patches/Mario_data/HRAC/matlab_2020_03_27/';

cd(data_dir)

tf_out_list = dir('MRP*mat');

%% First we prune them to the same size as the input MRI
disp('First we are stripping the extra zero padding off ofthe masks and "scrubbing" some high prediction voxels at the edge of the FOV')
for pid = 1:numel(tf_out_list)
    disp(['Pruning mask ' num2str(pid) ' out of ' num2str(numel(tf_out_list))])
    a = load(tf_out_list(pid).name) ;
    a.roi = squeeze(a.roi); a.out=squeeze(a.out);
    if size(a.out(1,:,:,1)) ~= size(a.roi(1,:,:))
        error('Failure! The size of the output and roi maps are not the same')
    end
    
    x = size(a.out,1) - size(a.mri,1); a.out(1:x,:,:,:)=[]; a.roi(1:x,:,:)=[];
    y = size(a.out,2) - size(a.mri,2); a.out(:,1:y,:,:)=[]; a.roi(:,1:y,:)=[];
    z = size(a.out,3) - size(a.mri,3); a.out(:,:,1:z,:)=[]; a.roi(:,:,1:z)=[];
    out=a.out; roi=a.roi; mri=a.mri;
    out = out .* repmat(mri>0,[1,1,1,1,size(out,5)]);
    
    save(['processed_' tf_out_list(pid).name],'out','roi','mri')

    
end

disp('Pruning complete')

%% Now we will actually adjust the mask
disp('Pocessing masks now-- takes a few min!: Converting 4D prediction (x,y,z,roi) to 3D mask:')
for pid = 1:numel(tf_out_list)
    disp(['Processing (step 2) mask ' num2str(pid) ' out of ' num2str(numel(tf_out_list))])
    disp(pid)
    a = load(['processed_' tf_out_list(pid).name]);
    tf_out = a.out;   %the network pred
    tf_out_modified = zeros(size(tf_out,1),size(tf_out,2),size(tf_out,3)); %this is what we will save
    for vox = 1:numel(tf_out_modified)
        [x,y,z] = ind2sub(size(tf_out_modified),vox);
        [v, id] = max(tf_out(x,y,z,:));
        tf_out_modified(vox) = id - 1;   
    end
    out=tf_out_modified;
    mri=a.mri;
    roi=a.roi;
    save(['processed2_' tf_out_list(pid).name],'out','mri','roi')
end

%% Now we can do the F1 analysis easily

disp('Calculating F1 scores')
f1=zeros(numel(tf_out_list),numel(6));
dsc=f1;

for pid = 1:numel(tf_out_list)
    pid
    a = load(['processed2_' tf_out_list(pid).name]);

    c = confusionmat(a.roi(:),a.out(:));
    tp = diag(c); %true positives
    
    %take out between
    %f1=zeros(numel(tf_out_list),numel(tp));
    %dsc=f1;
    %    
    for ii = 1:numel(tp)
        p = tp(ii) ./ sum(c(:,ii)); %precision
        r = tp(ii) ./ sum(c(ii,:)); %recall
        f1(pid,ii) = (2*p*r)/(p+r);
        dsc(pid,ii)=dice(a.roi(:)==ii-1,a.out(:)==ii-1);
    end
end

%% Makes some plots
close all
roi={'ANP','DCA','PCA','POP','VST'};

figure; title('F1 data'); ylabel('F1 score \pm 1 std'); xticks(1:5); xticklabels(roi)

hold on
bar(1:5,mean(f1(:,2:end),1))
errorbar(1:5,mean(f1(:,2:end),1),std(f1(:,2:end),1),'.')

figure; title('Dice  data'); ylabel('Dice score \pm 1 std'); xticks(1:5); xticklabels(roi)

hold on
bar(1:5,mean(dsc(:,2:end),1))
errorbar(1:5,mean(dsc(:,2:end),1),std(dsc(:,2:end),1),'.')

addpath('/home/mira/Documents/matlab_functions/notBoxPlot-master/code/')
addpath('/home/mira/Documents/matlab_functions/notBoxPlot-master/tests/')

figure; notBoxPlot(dsc(:,2:end));
title('Dice  data'); ylabel('Dice score'); xticks(1:5); xticklabels(roi)


cd(cwd)
    