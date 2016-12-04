package calGLCM;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.IOException;
import java.util.*;

import javax.swing.*;

public class mutipleNormalFrame extends JFrame implements ActionListener, ItemListener{
	
	JPanel mainFrame;
	JPanel pmemu;
	JPanel pbutton;
	JPanel degreeBox;
	JPanel checkBox;
	JPanel distanceBox;
	JPanel printOption;
	
	JButton run;
	JButton back;
	
	JCheckBox right;
	JCheckBox upRight;
	JCheckBox up;
	JCheckBox upLeft;
	
	JCheckBox printRight;
	JCheckBox printUpRight;
	JCheckBox printUp;
	JCheckBox printUpLeft;
	
	JCheckBox symetric;
	
	JTextField distance;
	
	String path;
	
	
	boolean isSymetric = false;
	
	boolean[] angle = new boolean[4];
	boolean[] options = new boolean[4];
	
	int elementDis = 1;
	
	calculationGLCM glcm1;
	calculationGLCM glcm2;
	calculationGLCM glcm3;
	calculationGLCM glcm4;
	
	
	public mutipleNormalFrame (String path) {
		
		this.path = path;
		
		mainFrame = new JPanel(new BorderLayout ());
		pmemu = new JPanel();
		pbutton = new JPanel();
		checkBox = new JPanel();
		degreeBox = new JPanel();
		distanceBox = new JPanel();
		printOption = new JPanel();
		
		for (int i = 0; i < angle.length; i++) {
			angle[i] = false;
			options[i] = false;
		}
		
		pmemu.setLayout(new GridLayout(6, 1));
		checkBox.setLayout(new GridLayout(1, 2));
		degreeBox.setLayout(new GridLayout(1, 4));
		distanceBox.setLayout(new GridLayout(1, 2));
		printOption.setLayout(new GridLayout(1, 4));
		
		
		JLabel title = new JLabel ("Mutiple Angle process", JLabel.CENTER);
		JLabel separate = new JLabel ("Select the angle for Normalization: ", JLabel.LEFT);
		JLabel notification = new JLabel ("Select GLCM Mode: ", JLabel.LEFT);
		title.setFont(new Font("Serif", Font.ITALIC, 30));
		
		JLabel separatePrint = new JLabel ("Options for features print: ", JLabel.LEFT);
		
		JLabel distanceLab = new JLabel ("Distance: ");
		
		distance = new JTextField ();
		
		run = new JButton ("Run");
		back = new JButton ("Back");
		
		symetric = new JCheckBox ("Symmetric");
		
		right = new JCheckBox ("0");
		upRight = new JCheckBox ("45");
		up = new JCheckBox ("90");
		upLeft = new JCheckBox ("135");
		
		printRight = new JCheckBox ("0");
		printUpRight = new JCheckBox ("45");
		printUp = new JCheckBox ("90");
		printUpLeft = new JCheckBox ("135");
		
		pmemu.add(distanceBox);
		pmemu.add(separate);
		pmemu.add(degreeBox);
		pmemu.add(checkBox);
		pmemu.add(separatePrint);
		pmemu.add(printOption);
		
		
		checkBox.add(notification);
		checkBox.add(symetric);
		
		distanceBox.add(distanceLab);
		distanceBox.add(distance);
		degreeBox.add(right);
		degreeBox.add(upRight);
		degreeBox.add(up);
		degreeBox.add(upLeft);
		
		printOption.add(printRight);
		printOption.add(printUpRight);
		printOption.add(printUp);
		printOption.add(printUpLeft);
		
		pbutton.add(back);
		pbutton.add(run);
		
		mainFrame.add(title, BorderLayout.PAGE_START);
		mainFrame.add(pmemu, BorderLayout.CENTER);
		mainFrame.add(pbutton, BorderLayout.PAGE_END);
		
		this.add(mainFrame);
		
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		this.setLocation(dim.width/2-this.getSize().width/2, dim.height/2-this.getSize().height/2);
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		this.setVisible(true);
		this.pack();
		//pmemu.setSize(200, 100);
		//this.setSize(600, 300);
		
		run.addActionListener(this);
		back.addActionListener(this);
		
		symetric.addItemListener(this);
		right.addItemListener(this);
		upRight.addItemListener(this);
		up.addItemListener(this);
		upLeft.addItemListener(this);
		
		printRight.addItemListener(this);
		printUpRight.addItemListener(this);
		printUp.addItemListener(this);
		printUpLeft.addItemListener(this);
		
	}
/*	
	private double[][] mergeMatrix (double[][] mainMatrix, double[][] addMatrix){
		
		for (int a = 0; a < 256; a++){
			for (int b = 0; b < 256; b++){
				mainMatrix[a][b] += addMatrix[a][b];
			}
		}
		return mainMatrix;
		
		
	}
	
	private double[][] matrixNormalization (double[][] matrix, int cout){
		
		for (int a = 0; a < 256; a++){
			for (int b = 0; b < 256; b++){
				matrix[a][b] /= cout;
			}
		}
		return matrix;
	}
*/	
	private void runGLCMprocess (String folderName, String resultName, int dis, boolean sym, boolean[] subangle, boolean[] suboption){
		
		File loadingFile = new File (path + folderName);
		File[] allFiles = loadingFile.listFiles();
		
		Map<String, double[]> hfeatures1 = new HashMap<String, double[]>();
		Map<String, double[]> vfeatures1 = new HashMap<String, double[]>();
		Map<String, double[]> diafeatures1 = new HashMap<String, double[]>();
		Map<String, double[]> andiafeatures1 = new HashMap<String, double[]>();
		Map<String, double[]> averfeatures1 = new HashMap<String, double[]>();
		
		Map<String, double[]> hfeatures2 = new HashMap<String, double[]>();
		Map<String, double[]> vfeatures2 = new HashMap<String, double[]>();
		Map<String, double[]> diafeatures2 = new HashMap<String, double[]>();
		Map<String, double[]> andiafeatures2 = new HashMap<String, double[]>();
		Map<String, double[]> averfeatures2 = new HashMap<String, double[]>();
		
		ArrayList<String> imageName = new ArrayList<String>();
		
		String outFileName = "";
		
		for (int num = 0; num < allFiles.length; num++){
			
			double[] hresult = null;
			double[] vresult = null;
			double[] dresult = null;
			double[] aresult = null;
			
			File imageFile = allFiles[num];

			loadImage img = new loadImage (imageFile);
			
			String currentFileName = imageFile.getName().substring(0, imageFile.getName().indexOf('-')) + "-" +
									imageFile.getName().substring(imageFile.getName().indexOf('.'));
			
			int tag = Integer.parseInt(imageFile.getName().substring(imageFile.getName().indexOf('-') + 1, imageFile.getName().indexOf('-') + 2));
			
			//System.out.println(currentFileName);
			
			if ( !outFileName.equals(currentFileName)){
				
				outFileName = currentFileName;
				imageName.add(outFileName);
				//System.out.println(outFileName);
			}
			
			int[][] matrix = img.getGrayLevelMatrix();
			
			//ef.outCVS(matrix);
			
			//double[][] glcm = new double[256][256];
			int count = 0;
			
			if (subangle[0]){
				glcm1 = new calculationGLCM (matrix, "0", dis, sym, true);
				
				//glcm = mergeMatrix(glcm, glcm1.getGLCM()); //used for Thai's matlab code
				
				GLCMFeatures hgf = new GLCMFeatures (glcm1.getGLCM());
				
				hresult = hgf.getResult();
				
				if (tag == 1){
					hfeatures1.put(outFileName, hresult);
				}else {
					hfeatures2.put(outFileName, hresult);;
				}
				
				count ++;
			}
			
			if (subangle[1]){
				glcm2 = new calculationGLCM (matrix, "45", dis, sym, true);
				
				//glcm = mergeMatrix(glcm, glcm2.getGLCM());
				
				GLCMFeatures diagf = new GLCMFeatures (glcm2.getGLCM());
				
				dresult = diagf.getResult();
				
				if (tag == 1){
					diafeatures1.put(outFileName, dresult);
				}else {
					diafeatures2.put(outFileName, dresult);
				}
				
				count ++;
			}
			
			if (subangle[2]){
				glcm3 = new calculationGLCM (matrix, "90", dis, sym, true);
				
				//glcm = mergeMatrix(glcm, glcm3.getGLCM());
				
				GLCMFeatures vgf = new GLCMFeatures (glcm3.getGLCM());
				
				vresult = vgf.getResult();
				
				if (tag == 1){
					vfeatures1.put(outFileName, vresult);
				}else{
					vfeatures2.put(outFileName, vresult);
				}
				
				count ++;
			}
			
			if (subangle[3]){
				glcm4 = new calculationGLCM (matrix, "135", dis, sym, true);
				
				//glcm = mergeMatrix(glcm, glcm4.getGLCM());
				
				GLCMFeatures agf = new GLCMFeatures (glcm4.getGLCM());
				
				aresult = agf.getResult();
				
				if (tag == 1){
					andiafeatures1.put(outFileName, aresult);
				}else {
					andiafeatures2.put(outFileName, aresult);
				}
				
				count ++;
			}
			
			double[] result = new double[aresult.length];

			for (int r = 0; r < result.length; r ++) {
				result[r] = (hresult[r] + vresult[r] + dresult[r] + aresult[r]) / count;
			}
			
			if (tag == 1){
				averfeatures1.put(outFileName, result);
			}else {
				averfeatures2.put(outFileName, result);
			}
	
		}

		try {
			
			if (suboption[0]){
				exportFile efh = new exportFile (resultName + " - 0 deg", path);
				efh.initiateFile();
				
				for(String name : imageName){
					double[] sResult = new double[20];
					double[] pResult = new double[20];
					
					if (hfeatures2.containsKey(name)){
						sResult = hfeatures2.get(name);
					}
					
					if (hfeatures1.containsKey(name)){
						pResult = hfeatures1.get(name);
					}
					
					efh.fileProcessing1(sResult, name);
					efh.fileProcessing2(pResult);
				}
			}
			
			if (suboption[1]){
				exportFile efd = new exportFile (resultName + " - 45 deg", path);
				efd.initiateFile();
				
				for(String name : imageName){
					double[] sResult = new double[20];
					double[] pResult = new double[20];
					
					if (hfeatures2.containsKey(name)){
						sResult = diafeatures2.get(name);
					}
					
					if (hfeatures1.containsKey(name)){
						pResult = diafeatures1.get(name);
					}
					
					efd.fileProcessing1(sResult, name);
					efd.fileProcessing2(pResult);
				}
			}
			
			if (suboption[2]){
				exportFile efv = new exportFile (resultName + " - 90 deg", path);
				efv.initiateFile();
				
				for(String name : imageName){
					double[] sResult = new double[20];
					double[] pResult = new double[20];
					
					if (hfeatures2.containsKey(name)){
						sResult = vfeatures2.get(name);
					}
					
					if (hfeatures1.containsKey(name)){
						pResult = vfeatures1.get(name);
					}
					
					efv.fileProcessing1(sResult, name);
					efv.fileProcessing2(pResult);
				}
			}
			
			if (suboption[3]){
				exportFile efa = new exportFile (resultName + " - 135 deg", path);
				efa.initiateFile();
				
				for(String name : imageName){
					double[] sResult = new double[20];
					double[] pResult = new double[20];
					
					if (hfeatures2.containsKey(name)){
						sResult = andiafeatures2.get(name);
					}
					
					if (hfeatures1.containsKey(name)){
						pResult = andiafeatures1.get(name);
					}
					
					efa.fileProcessing1(sResult, name);
					efa.fileProcessing2(pResult);
				}
			}
			
			
			
			
			exportFile efav = new exportFile (resultName + " - Average", path);
			efav.initiateFile();
			
			for(String name : imageName){
				double[] sResult = new double[20];
				double[] pResult = new double[20];
				
				if (hfeatures2.containsKey(name)){
					sResult = averfeatures2.get(name);
				}
				
				if (hfeatures1.containsKey(name)){
					pResult = averfeatures1.get(name);
				}
				
				efav.fileProcessing1(sResult, name);
				efav.fileProcessing2(pResult);
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		
		if (e.getSource() == run){
			
			long startTime = System.currentTimeMillis();
			
			elementDis = Integer.parseInt(distance.getText());
			
			runGLCMprocess("Cell", "Cell", elementDis, isSymetric, angle, options);
			
			runGLCMprocess("Debris", "Debris", elementDis, isSymetric, angle, options);
			
			runGLCMprocess("Strip", "Strip", elementDis, isSymetric, angle, options);
			
			long endTime   = System.currentTimeMillis();
			long totalTime = endTime - startTime;
			
			JOptionPane.showMessageDialog(null, "Calculation completed, time consuming: " + totalTime + " ms!", null, JOptionPane.INFORMATION_MESSAGE);
					
		}
		
		if (e.getSource() == back){
			userInterface mf = new userInterface(path);
			this.dispose();
		}
	}

	@Override
	public void itemStateChanged(ItemEvent i) {
		
		if (i.getItemSelectable() == symetric){
			isSymetric = !isSymetric;
		}
		
		if (i.getItemSelectable() == right){
			angle[0] = !angle[0];
		}
		
		if (i.getItemSelectable() == upRight){
			angle[1] = !angle[1];
		}
		
		if (i.getItemSelectable() == up){
			angle[2] = !angle[2];
		}
		
		if (i.getItemSelectable() == upLeft){
			angle[3] = !angle[3];
		}
		
		if (i.getItemSelectable() == printRight){
			options[0] = !options[0];
		}
		
		if (i.getItemSelectable() == printUpRight){
			options[1] = !options[1];
		}
		
		if (i.getItemSelectable() == printUp){
			options[2] = !options[2];
		}
		
		if (i.getItemSelectable() == printUpLeft){
			options[3] = !options[3];
		}
		
	}

}