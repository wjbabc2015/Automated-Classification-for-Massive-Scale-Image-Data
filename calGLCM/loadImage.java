package calGLCM;

import java.awt.*;
import java.awt.image.*;
import java.io.*;

import javax.imageio.*;

public class loadImage {	
	
	BufferedImage img;
	File imageFile;
	int[][] grayLevelMatrix;
	int[] testPixel;
	
	loadImage (File file){
		this.imageFile = file;
		imageProcessing (imageFile);
	}
	
	loadImage (){}
	
	/**
	 * Process the image file
	 * @param file
	 */
	private void imageProcessing (File file){
		try {
			img = ImageIO.read(file);
			grayLevelMatrix = imageToGray (img);
			//testPixel = testPixel(img);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Convert image to Gray-level matrix
	 * @param image
	 * @return 2-D matrix with GLs value
	 */
	private int[][] imageToGray (BufferedImage image){
		
		Raster raster = image.getData();
		
		int height = raster.getHeight();
		int width = raster.getWidth();
		
		int[][] result = new int[height][width];
		int[][] testResult = new int[image.getHeight()][image.getWidth()];
		
		for (int h =0; h < height; h++){
			for (int w = 0; w < width; w ++){
				//Color c = new Color (image.getRGB(w, h));
				
				//testResult[h][w] = c.getBlue();
				
				//int rgbValue = (c.getRed() + c.getGreen() + c.getBlue()) / 3;
						
				//result[h][w] = (int) (c.getRed() * 0.2989) + (int)(c.getGreen() * 0.5870) + (int)(c.getBlue() * 0.114);
				//result[h][w] = (int) (0.000004 * rgbValue * rgbValue * rgbValue + 0.003 * rgbValue * rgbValue - 0.033 * rgbValue + 1.8897);
				result[h][w] = raster.getSample(w, h, 0);
			}
			
			//int diff = result[h][40] - testResult[h][40];
			//System.out.println(result[h][40] + " - " + testResult[h][40] + " = " + diff);
		}
		//System.out.println(new Color(image.getRGB(20, 40)).getColorSpace().getType());
		return result;
	}
	
	private int[] testPixel (BufferedImage image){
		
		int[] result = new int[3];
		
		Color c = new Color (image.getRGB(248, 134));    //x , y 
		
		result[0] = c.getRed();
		result[1] = c.getGreen();
		result[2] = c.getBlue();
		
		return result;
	}
	
	public int[][] getGrayLevelMatrix (){
		return grayLevelMatrix;
	}
	
	public int[] getRGBvalue (){
		return testPixel;		
	}
	
	
	public static void main(String[] args) {
		
		File loadingFile = new File ("C:/Users/jiabin/Desktop/GLCM_Cal/Image/45deg/camera0/PicA1009.bmp");
		
		loadImage lI = new loadImage (loadingFile);
		
		int[][] tempResult = lI.getGrayLevelMatrix();

	}
}
