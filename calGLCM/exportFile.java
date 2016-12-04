package calGLCM;

import java.io.*;

public class exportFile {
	
	String fileName;
	String path;

	public exportFile (String name, String path) {
		this.fileName = name;
		this.path = path;
	}
	
	public exportFile(){}
	
	public void initiateFile () {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter (new File(path + fileName + ".csv"));
			
			StringBuilder sb = new StringBuilder ();
			
			sb.append("Pic\\Features");
			sb.append(",");
			sb.append("s-ASM");
			sb.append(',');
			sb.append("s-CON");
			sb.append(',');
			sb.append("s-COR");
			sb.append(',');
			sb.append("s-VAR");
			sb.append(',');
			sb.append("s-IDM");
			sb.append(",");
			sb.append("s-SAV");
			sb.append(',');
			sb.append("s-SEN");
			sb.append(',');
			sb.append("s-SVA");
			sb.append(',');
			sb.append("s-ENT");
			sb.append(',');
			sb.append("s-DEN");
			sb.append(",");
			sb.append("s-DVA");
			sb.append(',');
			sb.append("s-DIS");
			sb.append(',');
			sb.append("s-CLS");
			sb.append(',');
			sb.append("s-CLP");
			sb.append(',');
			sb.append("s-MIP");
			sb.append(',');
			sb.append("s-MAP");
			sb.append(',');
			sb.append("s-MEA");
			sb.append(',');
			sb.append("s-IMIN");
			sb.append(',');
			sb.append("s-IMAX");
			sb.append(',');
			sb.append("s-IMEA");
			sb.append(",");
			sb.append("p-ASM");
			sb.append(',');
			sb.append("p-CON");
			sb.append(',');
			sb.append("p-COR");
			sb.append(',');
			sb.append("p-VAR");
			sb.append(',');
			sb.append("p-IDM");
			sb.append(",");
			sb.append("p-SAV");
			sb.append(',');
			sb.append("p-SEN");
			sb.append(',');
			sb.append("p-SVA");
			sb.append(',');
			sb.append("p-ENT");
			sb.append(',');
			sb.append("p-DEN");
			sb.append(",");
			sb.append("p-DVA");
			sb.append(',');
			sb.append("p-DIS");
			sb.append(',');
			sb.append("p-CLS");
			sb.append(',');
			sb.append("p-CLP");
			sb.append(',');
			sb.append("p-MIP");
			sb.append(',');
			sb.append("p-MAP");
			sb.append(',');
			sb.append("p-MEA");
			sb.append(',');
			sb.append("P-IMIN");
			sb.append(',');
			sb.append("P-IMAX");
			sb.append(',');
			sb.append("P-IMEA");
			sb.append('\n');
			
			pw.write(sb.toString());
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			if (pw != null) pw.close();
		}
		
	}
	
	public void fileProcessing1 (double[] result, String nameTitle) throws IOException{
		
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new FileWriter(path + fileName + ".csv", true));
			
			StringBuilder sb = new StringBuilder ();
			
			sb.append(nameTitle);
			sb.append(',');
			
			sb.append(result[0]);
			sb.append(',');
			
			sb.append(result[1]);
			sb.append(',');
			
			sb.append(result[2]);
			sb.append(',');
			
			sb.append(result[3]);
			sb.append(',');
			
			sb.append(result[4]);		
			sb.append(',');
			
			sb.append(result[5]);
			sb.append(',');
			
			sb.append(result[6]);
			sb.append(',');
			
			sb.append(result[7]);
			sb.append(',');
			
			sb.append(result[8]);
			sb.append(',');
			
			sb.append(result[9]);
			sb.append(',');
			
			sb.append(result[10]);
			sb.append(',');
			
			sb.append(result[11]);
			sb.append(',');
			
			sb.append(result[12]);
			sb.append(',');
			
			sb.append(result[13]);
			sb.append(',');
			
			sb.append(result[14]);
			sb.append(',');
			
			sb.append(result[15]);
			sb.append(',');
			
			sb.append(result[16]);
			sb.append(',');
			
			sb.append(result[17]);
			sb.append(',');
			
			sb.append(result[18]);
			sb.append(',');
			
			sb.append(result[19]);
			sb.append(',');
			
			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			if (pw != null) pw.close();
		}
	}
	
public void fileProcessing2 (double[] result) throws IOException{
		
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new FileWriter(path + fileName + ".csv", true));
			
			StringBuilder sb = new StringBuilder ();
			
			sb.append(result[0]);
			sb.append(',');
			
			sb.append(result[1]);
			sb.append(',');
			
			sb.append(result[2]);
			sb.append(',');
			
			sb.append(result[3]);
			sb.append(',');
			
			sb.append(result[4]);		
			sb.append(',');
			
			sb.append(result[5]);
			sb.append(',');
			
			sb.append(result[6]);
			sb.append(',');
			
			sb.append(result[7]);
			sb.append(',');
			
			sb.append(result[8]);
			sb.append(',');
			
			sb.append(result[9]);
			sb.append(',');
			
			sb.append(result[10]);
			sb.append(',');
			
			sb.append(result[11]);
			sb.append(',');
			
			sb.append(result[12]);
			sb.append(',');
			
			sb.append(result[13]);
			sb.append(',');
			
			sb.append(result[14]);
			sb.append(',');
			
			sb.append(result[15]);
			sb.append(',');
			
			sb.append(result[16]);
			sb.append(',');
			
			sb.append(result[17]);
			sb.append(',');
			
			sb.append(result[18]);
			sb.append(',');
			
			sb.append(result[19]);
			
			sb.append('\n');
			
			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			if (pw != null) pw.close();
		}
	}
	
	public void outCVS (int[][] array){
		
		StringBuilder sb = new StringBuilder ();
		
		PrintWriter pw = null;
		
		try {
			pw =new PrintWriter (new FileWriter(path + "grayscaleMatrix.csv", true));
			
			for (int i = 0; i < array.length; i++){
				
				for (int j = 0; j < array[0].length; j++){
					sb.append(array[i][j]);
					sb.append(",");
				}
				
				sb.append("\n");
			}
			
			pw.write(sb.toString());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally{
			if (pw != null) pw.close();
		}
	}
	
public void outCVS (double[][] array){
		
		StringBuilder sb = new StringBuilder ();
		
		PrintWriter pw = null;
		
		try {
			pw =new PrintWriter (new FileWriter(path + "GLCMMatrix.csv", true));
			
			for (int a = 1; a <= 256; a ++){
				sb.append(a);
				sb.append(',');
			}
			
			sb.append('\n');
			
			for (int i = 0; i < array.length; i++){
				
				for (int j = 0; j < array[0].length; j++){
					sb.append(array[i][j]);
					sb.append(",");
				}
				
				sb.append("\n");
			}
			
			pw.write(sb.toString());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally{
			if (pw != null) pw.close();
		}
	}
}
