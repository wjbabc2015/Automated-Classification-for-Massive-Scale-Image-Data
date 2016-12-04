%author Sai Kiran Thati

% Number of images present in the folder
%( all the images have to be renamed in increasing order of number starting from 1..example; 1,2,3,..n)
n = 4;
statsMtx=[];
for i=1:n
%Converting number to string to read images. example; '1.bmp','2.bmp'...'n.bmp')
filename = [num2str(i) '.bmp'];
%reading the image in matlab
I=imread(filename);
%graycomatrix function creates GLCM for the image read
%NumLevels can be set to the desired Gray levels. Note: Matlab does uniform quantization
% Angle and offset(pixel distance) 'D' can be set by 
% 0              [  0 D ]
% 45             [ -D D ]
% 90             [ -D 0 ]
% 135            [ -D -D] 
% below code uses d=16 and all four orientations
% when symmetric true computes graycomatrix counts 
% both 1,2 and 2,1 pairings when calculating the number of times the value 1 is adjacent to the value 2
GLCM2=graycomatrix(I, 'NumLevels',256,'offset', [0 1; -1 1; -1 0; -1 -1],'symmetric',true);
%taking the average of all four orientations.
GLCM2=(GLCM2(:,:,1)+GLCM2(:,:,2)+GLCM2(:,:,3)+GLCM2(:,:,4))/4;
% calling the method GLCM_Features4  calculates all the GLCM features from the matrices. 
stats2=GLCM_Features4(GLCM2,0);
% converting from struct to array 
t=struct2array(stats2);
% storing the array in stats matrix
statsMtx =[statsMtx ; t];
end
