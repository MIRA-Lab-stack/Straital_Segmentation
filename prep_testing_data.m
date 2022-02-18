% Choose a directory
testList = '/media/mira/Data/karl/striatum/patches/Mario_data/HRAC/matlab_2020_03_27/';
cd(testList)
a=dir('*mat');

for i = 1:numel(a)
    aa=load(a(i).name);
    aa=aa.roi;
    k=aa;
    size(k)
    k1 = zeros(size(k));
    k2=k1; k3=k1; k4=k1; k5=k1; k6=k1;
    for pix = 1 : numel(k)
        if k(pix)==0
            k1(pix)=1;
            
        elseif k(pix)==1
            k2(pix)=1;
            
        elseif k(pix)==2
            k3(pix)=1;
            
        elseif k(pix)==3
            k4(pix)=1;
            
        elseif k(pix)==4
            k5(pix)=1;
         elseif k(pix)==5
            k6(pix)=1;
        end
    end
    
   mask = cat(5, k1,k2,k3,k4,k5,k6);
   save(a(i).name,'mask','-append')
end