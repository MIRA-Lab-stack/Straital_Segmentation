Hello!

This project uses a deep learning Convolutional Neural Network for Multiclass Striatal Segmentation from T1 MRI images. 

Multitask Learning Based Three-Dimensional Striatal Segmentation of MRI: fMRI and PET Objective Assessments
Journal of Magnetic Resonance Imaging

Serrano-Sosa M, Van Snellenberg JX, Meng J, Luceno JR, Spuhler K, Weinstein JJ, Abi-Dargham A, Slifstein M, Huang C. Multitask Learning Based Three-Dimensional 	Striatal Segmentation of MRI: fMRI and PET Objective Assessments. J Magn Reson Imaging. 2021 Nov;54(5):1623-1635. doi: 10.1002/jmri.27682. Epub 2021 May 		10. PMID: 33970510.


1) 

The first steps in implementing this code is to download the appropriate python libraries:

tensorflow: v. 1.14.0
numpy: v. 1.16.5
nibabel: v. 2.4.1
scipy: v. 1.2.2

We also recommend entering the 'network_weights' folder and editting the file 'checkpoint'

This can be edited in any txt file editor. Tensorflow looks for the checkpoint prior to reading the weights.

Change any mention of '/Your_directory/network_weights/' to the directory where you have downloaded this github repository.

2)

The next step is data organization. You will need to have a data directory structured as:

-> Data directory (that contains all subjects that will be used to create segmentations)
----> MRI_#1 (Individual subject folders)
------->MRI_#1_cropped_volume.hdr
------->MRI_#1_cropped_volume.img

----> MRI_#2
------->MRI_#2_cropped_volume.hdr
------->MRI_#2_cropped_volume.img

----> MRI_#xxx
------->MRI_#xxx_cropped_volume.hdr
------->MRI_#xxx_cropped_volume.img


3) 

Next you will need to run the python script 'NNeval.py'
This script has been modified and the only thing that should be changed is where it states '/Your_directory/' and '/Your_data_directory/'

Depending on how you have saved the following files on your system
a) '/Your_directory/' should point to the entire filepath of this downloaded github repository
b) '/Your_data_directory/' should point to the Data directory previously stated in step 2 of this READ_ME

Running this code will create a new folder that contains the Striatal segmentations of all subjects in the Data directory. However, these will be in '*.mat' files.

It is strongly recommended to visually inspect the segmentations. In rare instances, it was noticed that voxels outside of the brain were assigned an ROI value. These can be easily cropped away.

Depending on the way one wants to conduct further analysis, they can keep in '*.mat' form or convert them back into the original nifti/analyze75 format


4) (Optional)


Our pipeline used data in analyze75 format, the code "After_inspecting_segs.m" will help in cropping the unwanted ROI voxels mentioned in step 3 and also create analyze75 format volumes of the Striatal ROI's

This code has already been modified and only a few things need to be changed"

a) '/Your_directory/' should point to the entire filepath of this downloaded github repository
b) '/Your_data_directory/' should point to the Data directory previously stated in step 1 of this READ_ME
c) '/Make_new_directory_for_segmentations/' can be changed to create a new directory only containing analyze75 format segmentations


Thank you!!


If there are any questions in the implementation of the code, please email miralab.sbu@gmail.com





