package calGLCM;

public class calculationGLCM {
	
	int[][] matrix;
	int dx = 0;
	int dy = 0;
	String degree;
	int distance;
	int count = 0;
	
	boolean symetic;
	boolean normalization;
	
	double[][] GLCM = new double[256][256];

	public calculationGLCM(int[][] graylevelMatrix, String degree, int distance, boolean symetic, boolean normalization){
		this.matrix = graylevelMatrix;
		this.degree = degree;
		this.distance = distance;
		this.symetic = symetic;
		this.normalization = normalization;
		
		computeXY (distance, degree);
		
		computeGLCM (symetic);
		
		if (normalization) {
			computeNormalization();
		}
		
	}
	
	private void computeXY(int distance2, String anDegree) {
		
		if (anDegree == "0"){
			dx = distance2;
		}else if (anDegree == "45"){
			dx = distance2;
			dy = distance2;
		}else if(anDegree == "90"){
			dy = distance2;
		}else if (anDegree == "135"){
			dx = -distance2;
			dy = distance2;
		}
	}

	private void computeGLCM (boolean sym){
		
		if (dx > 0 && dy == 0){
			for (int h = 0; h < matrix.length; h++){
				for (int w = 0; w < matrix[h].length - dx; w++){
					int a = matrix[h][w];
					int b = matrix[h][w + dx];
					
					GLCM[a][b] ++;
					count ++;
					
					if (sym){
						GLCM[b][a] ++;
						count ++;
					}
				}
			}
		}else if (dx > 0 && dy > 0){
			for (int h = dy; h < matrix.length; h++){
				for (int w = 0; w < matrix[h].length - dx; w++){
					int a = matrix[h][w];
					int b = matrix[h - dy][w + dx];
					
					GLCM[a][b] ++;
					count ++;
					
					if (sym){
						GLCM[b][a] ++;
						count ++;
					}
				}
			}
		}else if (dx == 0 && dy > 0){
			for (int h = dy; h < matrix.length; h++){
				for (int w = 0; w < matrix[h].length; w++){
					int a = matrix[h][w];
					int b = matrix[h - dy][w];
					
					GLCM[a][b] ++;
					count ++;
					
					if (sym){
						GLCM[b][a] ++;
						count ++;
					}
				}
			}
		}else if (dx < 0 && dy > 0){
			for (int h = dy; h < matrix.length; h++){
				for (int w = -dx; w < matrix[h].length; w++){
					int a = matrix[h][w];
					int b = matrix[h - dy][w + dx];
					
					GLCM[a][b] ++;
					count ++;
					
					if (sym){
						GLCM[b][a] ++;
						count ++;
					}
				}
			}
		}
	}
	
	private void computeNormalization (){

//System.out.println(count);
		for (int r = 0; r < GLCM.length; r++){
			for (int c = 0; c < GLCM[r].length; c++){
				GLCM[r][c] = GLCM[r][c] / (double) count;
			}
		}
	}

	public double[][] getGLCM (){
		return GLCM;
	}
	
	
}
