GLCM Calc Tool.exe (version 0.1)
July 28, 2015

The GLCM calculation tool receives input in the form of 32 bit “.bmp” images. The output of the tool is a set of “.csv” files (readable in Excel) containing 17 textural features per image. It is currently executable only on Windows.
The directory structure of the input should be the following:

Input Directory
    45deg
        camera0
        camera1
    hor
        camera0
        camera1
    ver
        camera0
        camera1
	
Where the Input Directory is specified in a file named “directory.cfg”. The subdirectories of the Input Directory are “45deg”, “hor”, and “ver”, which specify the diffraction image angles. Each of these directories should contain two subdirectories, “camera0” and “camera1”. The “.bmp” images should be stored in these subdirectories. Camera0 images are specified with the “s” designation, and camera1 images are specified with the “p” designation in the output format. 

The file names of the “.bmp” images should be given a common prefix, a differing element, and an enumerated suffix. For instance, if the “camera0” directory contained three images named:
	PicA1027.bmp, PicA1028.bmp, PicA1029.bmp
then the directory “camera1” should contain corresponding images:
	PicB1027.bmp, PicB1028.bmp, PicB1029.bmp

Two additional requirements are necessary for the input structure at this time:
1.	“45deg”, “hor”, and “ver” be the only folders or files inside of the Input Directory.
2.	“camera0” and “camera1” directories must contain the same number of .bmp images, each with a corresponding enumerated image.

“directory.cfg” is that simply contains the path of the Input Directory on the first line. For instance the path might be:
	C:\Users\John\Desktop\2012-10-19_ProsCells(8bit bmp)\PC3
where PC3 is the Input Directory.

 
The output format of the tool is a set of “.csv” (comma separated value) files which can be easily opened in any spreadsheet viewer/editor. The output will consist of 15 files. The files are prefixed with a unique date, followed by the diffraction image angle, and ending with the angle of the GLCM calculation. For instance:
	2015_7_28__1511_hor_135.csv
denotes July 28, 2015 at 3:11 P.M. whose diffraction images have a horizontal orientation, whose GLCM angle is 135 degrees.

Additional Notes:
•	Source Code is included, but is not commented yet. This is on the to-do list, as well as a write-up the proper way to use the included library for calculating GLCMs and their features.
•	The input handling is not very robust at the moment. This will be updated
•	The program currently only calculates distance 1 in all directions (as well as calculates the average of all of the directions). The library I’ve written is capable of calculating any distance that is within the bounds of the image, and the program / output model can be changed to include more distances.


dixonj13@students.ecu.edu
