%% After visually inspecting images, editing    
 
% Out of preference, we use the 'load_untouch_nii' commands to work with nifti's in matlab 
% This code could be changed to any preference one may have

original_mri_dir = '/Your_data_directory/';
work_dir = '/Your_data_directory/CNN_outputs/';
% This next step is optional, you can always change where to save the ROI nifti files
dir_for_newSegs = '/Make_new_directory_for_segmentations/'
mkdir(dir_for_newSegs)
cd(work_dir)
files = dir(['/Your_data_directory/CNN_outputs/*.mat']);

for i=1:numel(files)
    cd(work_dir)
    name = files(i).name;
    load(name);
    Cnn = out;

    % Comment out if you dont need cropping
     Cnn(1:80,:,:)=0;
     Cnn(160:end,:,:)=0;
     Cnn(:,1:105,:)=0;
     Cnn(:,155:end,:)=0;
    

% The following code loads the original MRI header and data
    mri_name = name(1:end-4);
    temp = strcat(original_mri_dir,mri_name,'/*MRI_volume*');
    mri_dir = dir(temp); mri_dir = mri_dir.name;
    mri_file = strcat(originalMRI_dir,mri_name,'/',mri_dir);
    MRI = load_untouch_nii(mri_file);
    cnn = MRI;
    cnn.img = Cnn;
    
    cnn.hdr.hist.descrip = 'CNN_output';
    file_name = strcat(mri_name,'_CNN_output.nii');
    cd(dir_for_newSegs);
    save_untouch_nii(cnn, file_name);

end

 
