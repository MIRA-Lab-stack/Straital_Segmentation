%%% DO NOT RUN THIS CODE COMPLETELY, RUN PER SECTION!!!


%% For visually inspecting segmentations
launch = pwd;
work_dir = '/media/mira/Data/karl/striatum/patches/Mario_data/HRAC/matlab_2020_03_27/';
cd(work_dir);
files = dir('processed2_*');
for i=1:numel(files)
%     name = files(i).name;
%     cnn_file = strcat(work_dir,'/',name,'/',name,'_CNN_output.nii');
%     temp = strcat(work_dir,'/',name,'/*norm*');
%     mri = dir(temp); mri = mri.name;
%     mri_file = strcat(work_dir,'/',name,'/',mri);
%     CNN = load_untouch_nii(cnn_file);
%     cnn = CNN.img;
%     MRI = load_untouch_nii(mri_file);
%     mri_image = MRI.img;

    load(files(i).name);
    name = files(i).name;
    
    %% Always change this below to edit
     out(1:80,:,:)=0;
     out(160:end,:,:)=0;
     out(:,1:105,:)=0;
     out(:,155:end,:)=0;
    
    % change indices below
    figure; subplot(1,2,1); imshow(out(:,:,65),[]); axis on; subplot(1,2,2); imshow(roi(:,:,65),[]); axis on;title(name);
    figure; subplot(1,2,1); imshow(out(:,:,60),[]); axis on; subplot(1,2,2); imshow(roi(:,:,60),[]); axis on;title(name);
    figure; subplot(1,2,1); imshow(out(:,:,55),[]); axis on; subplot(1,2,2); imshow(roi(:,:,55),[]); axis on;title(name);
    figure; subplot(1,2,1); imshow(out(:,:,50),[]); axis on; subplot(1,2,2); imshow(roi(:,:,50),[]); axis on;title(name);
    
    
    save(files(i).name,'mri','out','roi')
end    
    
%% After visually inspecting images, editting    
 
addpath('/home/mira/Documents/matlab_functions/');
cd(launch)
main_dir = pwd; %always should be where the matlab code is
work_dir = '/media/mira/Data/karl/striatum/patches/Mario_data/HRAC/matlab_2020_03_27/'; %point to folder you want to edit
originalMRI_dir = '/media/mira/Data/karl/striatum/patches/Mario_data/HRAC/';
dir_for_newSegs = '/media/mira/Data/karl/striatum/patches/Mario_data/03_27_2020_StraitalSeg/';
matlab_files = [work_dir 'M*.mat'];
cd(work_dir)
files = dir(['processed2_*.mat']);
cd(main_dir)

for i=1:numel(files)
    cd(work_dir)
    name = files(i).name;
    load(name);
    Cnn = out;
%     Cnn_file = strcat(work_dir,'/',name,'/',name,'_CNN_output.nii');
%     CNN = load_untouch_nii(Cnn_file);
%     Cnn = CNN.img;

    % Comment out if you dont need cropping
     Cnn(1:80,:,:)=0;
     Cnn(160:end,:,:)=0;
     Cnn(:,1:105,:)=0;
     Cnn(:,155:end,:)=0;
    
    mri_name = name(12:end-4);
    temp = strcat(originalMRI_dir,mri_name,'/*cropped*');
    mri_dir = dir(temp); mri_dir = mri_dir.name;
    mri_file = strcat(originalMRI_dir,mri_name,'/',mri_dir);
    MRI = load_untouch_nii(mri_file);
    cnn = MRI;
    cnn.img = Cnn;
    
    Rois = MRI;
    Rois.img = roi;
    
    cnn.hdr.hist.descrip = 'CNN_output';
    Rois.hdr.hist.descrip = 'Original_Rois';
    %CNN_output = make_nii(cnn);
    file_name = strcat(mri_name,'_CNN_output.nii');
    roi_name = strcat(mri_name,'_original_ROIs.nii');
    new_dir = strcat(dir_for_newSegs,mri_name);
    mkdir(new_dir);
    cd(new_dir);
    save_untouch_nii(cnn, file_name);
    save_untouch_nii(Rois, roi_name);
    copyfile(temp,new_dir)
    cd(main_dir);
end

 