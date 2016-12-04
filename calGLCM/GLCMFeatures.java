package calGLCM;

public class GLCMFeatures {
	
	double[][] matrix;
	
	double[] result;
	
	double[][] mulContrast;
	double[][] mulDissimilarity;
	
	double pAverage=0;
	
	double px=0;
	double py=0;
	double stdevx=0.0;
	double stdevy=0.0;
	
	double[] pxplusy;
	double[] pxminusy;
	
	double sum = 0;
	double mean = 0;
/*	
	double asm = 0.0; ---- 0
	
	double contrast=0.0; ---- 1
	
	double correlation=0.0; ---- 2
	
	double IDM=0.0; ---- 3
	
	double entropy=0.0; ---- 4
	double suma = 0.0; ---- 5
*/	
	
 
	public GLCMFeatures (double[][] glcm){
		
		this.matrix = glcm;
		result = new double [20];
		mulContrast = new double[256][256];
		mulDissimilarity = new double[256][256];
		
		pxplusy = new double[2 * matrix.length - 1];
		pxminusy = new double[matrix.length];
		
/*					
		for (int a = 0; a < 256 ;a ++ ) {
			for (int b= 0; b< 256 ; b ++ ) {
				System.out.print(matrix[a][b] + " ");
			}
			System.out.println();
		}*/

		//doNormalization ();
		initialization();
		doCalculationFeatures();
		
	}
	
	/**
	 * No need for this version

	private void doNormalization () {

		for (double[] eleArray : matrix) {
			for (double ele : eleArray) {
				sum += ele;
			}
		}

//System.out.println(sum);	

		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				matrix[r][w] = matrix[r][w] / sum;
			}
		}
	}
	*/
	
	private void initialization (){
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				mulContrast[r][w] += (r - w) * (r - w);
				mulDissimilarity[r][w] += Math.abs(r - w);
				
				pAverage += r * matrix[r][w];
				
				
				px += (r + 1) * matrix[r][w];
				py += (w + 1) * matrix[r][w];
				
				pxplusy[r + w] += matrix[r][w];
				pxminusy[Math.abs(r - w)] += matrix[r][w];
				
			}
		}
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[r].length; w++){
				
//System.out.println(stdevx);
				stdevx=stdevx+(r+1-px)*(r+1-px)*matrix [r][w];
				stdevy=stdevy+(w+1-py)*(w+1-py)*matrix [r][w];
			}
			//System.out.println(stdevx + ":" + stdevy);
		}
		
		stdevx = Math.sqrt(stdevx);
		stdevy = Math.sqrt(stdevy);
		mean = 1 / (256 * 256);
	}
	
	/**
	 * Calculate standard deviation for array
	 * @param elements
	 * @return
	 */
	private double getStd(double[] elements){
		
		double std = 0;
		
		int size = elements.length;
		double mean = 0;
		
		for (double ele : elements) {
			mean  += ele;
		}
		
		mean /= size;
		
		double varience = 0;
		
		for (double ele: elements) {
			varience += (ele - mean) * (ele - mean);
		}
		
		std = Math.sqrt(varience / (size - 1));
		
		return std;
	}
	
	/**
	 * Revised algorithm for 17 features based on Dr. Hu's new document
	 */
	private void doCalculationFeatures(){
		
		result[14] = 255;
		result[15] = 0;
		
		for (int r = 0; r < matrix.length; r++) {
			
			for (int c = 0; c < matrix[r].length; c++){
				
				//energy or Angular Second Moment = ASM
				result[0] += matrix[r][c] * matrix[r][c];
				
				//Correlation and Mean = COR and MEA
				result[2] += (r + 1 - px) * (c + 1 - py) * matrix[r][c];
				
				result[16] = px; //MEA
				
				//Variance (Sum of Squares) = VAR
				result[3] += Math.pow((r - pAverage), 2.0) * matrix[r][c];
				
				//Entropy = ENT
				if (matrix[r][c] != 0)
					result[8] += -matrix[r][c] * Math.log(matrix[r][c]);
				
				//Dissimilarity = DIS
				result[11] += mulDissimilarity[r][c] * matrix[r][c];
				
				//Cluster shade = CLS
				result[12] += Math.pow((r + c + 2 - 2 * px), 3.0) * matrix[r][c];
				
				//Cluster prominence = CLP
				result[13] += Math.pow((r + c + 2 - 2 * px), 4.0) * matrix[r][c];
				
				//MAP
				result[15] = Math.max(result[15], matrix[r][c]);
				
				//MIP
				result[14] = Math.min(result[14], matrix[r][c]);
			}
			
		}
		
		result[2] /= (stdevx * stdevy);//COR
		
		for (int pos = 0; pos < pxminusy.length; pos ++) {
			
			//System.out.println(pxminusy[pos]);
			//Contrast = CON
			result[1] += pxminusy[pos] * pos * pos;
			
			//Inverse Difference Moment = IDM
			result[4] += pxminusy[pos] / (pos * pos + 1);
			
			//Difference Entropy = DEN
			if (pxminusy[pos] != 0)
				result[9] -= pxminusy[pos] * Math.log(pxminusy[pos]);
			
		}
		
		//Difference Variance
		result[10] = getStd(pxminusy) / (pxminusy.length - 1);
		//System.out.println(result[10]);
		
		for (int pos = 0; pos < pxplusy.length; pos ++) {
			
			//System.out.println(result[6]);
			
			//Sum Average = SAV
			result[5] += (pos + 2) * pxplusy[pos]; //needed further validation
			
			//Sum Entropy = SEN
			if (pxplusy[pos] != 0){
				result[6] -= pxplusy[pos] * Math.log(pxplusy[pos]);
			}
		}
		//System.out.println(result[6]);
		
		for (int index = 0; index < pxplusy.length; index ++){
			//Sum Variance = SVA
			result[7] += Math.pow((index - result[6] + 2), 2.0) * pxplusy[index];
			//System.out.println(result[6]);
		}
	}
	
/* The same algorithm with Thai Matlab code	
	private void doIcalculateAutocorrelation(){
//System.out.println(matrix[20][115] * (-px + 21)*(-px + 21));	
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
			
				result[0] += (r + 1) * (w + 1) * matrix[r][w];
			}
		}
	}
	
	private void doIcalculateContrast() {
		
		for (int a = 0; a < matrix.length; a ++){
			
			for (int b = 0; b<matrix[1].length; b ++){
				result[1] += mulContrast[a][b] * matrix[a][b];
			}
		}
		
	}

	private void doIcalculateCorrelation() {
//System.out.println(px + ":" + py);
//System.out.println(stdevx + ":" + stdevy);
		
		for (int e = 0; e < 256; e ++){
			for (int f = 0; f < 256; f ++){
				result[2] +=((e - px) * (f - py) * matrix[e][f]) / (stdevx * stdevy) ;
			}
		}
		//result[2] +=(result[0] - px * py) / (stdevx * stdevy) ;
//System.out.println(result[2]);
	}
	
	private void doIcalculateClusterProminence (){
//System.out.println(Math.pow((- px - py + 2), 4.0) * matrix[0][0]);
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				result[3] += Math.pow((r + w - px - py), 4.0) * matrix[r][w];
				
			}
			//System.out.println(result[3]);
		}
	}
	
	private void doIcalculateClusterShade (){
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				result[4] += Math.pow((r + w - px - py + 2), 3.0) * matrix[r][w];;
			}
		}
	}
	
	private void doIcalculateDissimilarity (){
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				result[5] += mulDissimilarity[r][w] * matrix[r][w];
			}
		}
	}
	
	private void doIcalculateEnergy (){
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				result[6] += matrix[r][w] * matrix[r][w];
			}
		}
	}
	
	private void doIcalculateHomogeneity(){
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				result[8] += matrix[r][w] / (1 + mulContrast[r][w]);
			}
		}
	}
	
	private void doIcalculateMaximumProbability (){
		
		double max = 0;
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				max = Math.max(max, matrix[r][w]);
			}
		}
		
		result[9] = max;
	}
	
	private void doIcalculateVariance (){
		
		for (int r = 0; r < matrix.length; r ++){
			for (int w = 0; w < matrix[1].length; w ++){
				
				result[10] += (r - mean + 1) * (r - mean + 1) * matrix[r][w];
			}
		}
		//System.out.println(result[10]);
	}
	
	private void doIcalculateSumAverage  (){
		
		for (int index = 0; index < pxplusy.length ; index++){
			result[11] += (index + 2) * pxplusy[index];
			//System.out.println(pxplusy[index]);
		}
		
		//System.out.println(result[11]);
	}



	private void doIcalculateIDM() {
		
		for (int a = 0; a < 256; a ++){
			
			for (int b = 0; b<256; b ++){
				result[14] += (matrix[a][b] / ( 1 + Math.abs(a - b) / 256));
			}
		}
		
		//System.out.println(result[14]);
	}



	private void doIcalculateEntropy() {
		
		for (int a = 0; a < 256; a ++){
			
			for (int b = 0; b < 256; b ++){
				if (matrix[a][b] != 0){
					result[7] += -(matrix[a][b]*(Math.log(matrix[a][b] + 2.2204e-16)));
				}
			}
		}
		
		//sum entropy
		for (int index = 0; index < pxplusy.length ; index++){
			result[12] +=  -(pxplusy[index]*(Math.log(pxplusy[index] + 2.2204e-16)));
		}
		
		//diff entropy
		for (int index = 0; index < pxminusy.length ; index++){
			result[13] +=  -(pxminusy[index]*(Math.log(pxminusy[index] + 2.2204e-16)));
		}
	}
*/
	public double[] getResult(){
		return result;
	}
	
	public static void main(String[] arg[]) {
		
	}
}
