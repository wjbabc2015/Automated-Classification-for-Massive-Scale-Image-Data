package calGLCM;

import static org.junit.Assert.*;

import org.junit.Test;

public class calculationGLCMTest {

	@Test
	public void test() {
		int[][] imageMatrix = {{1 , 6 , 2 , 3 , 9 , 3 , 0 , 6 , 2 , 8},
								{6 , 5 , 4 , 4 , 8 , 6 , 5 , 3 , 3 , 8},
								{0 , 9 , 7 , 0 , 0 , 6 , 8 , 2 , 6 , 7},
								{1 , 6 , 7 , 9 , 7 , 5 , 6 , 4 , 2 , 2},
								{5 , 7 , 1 , 2 , 2 , 6 , 2 , 4 , 7 , 5},
								{1 , 4 , 4 , 1 , 4 , 6 , 3 , 1 , 9 , 0},
								{7 , 4 , 5 , 3 , 5 , 2 , 4 , 5 , 7 , 4},
								{7 , 7 , 4 , 2 , 8 , 1 , 9 , 2 , 3 , 3},
								{7 , 1 , 6 , 4 , 4 , 9 , 1 , 3 , 5 , 1},
								{1 , 1 , 6 , 3 , 9 , 2 , 8 , 5 , 1 , 2}};
		
		double[][] expectedGLCM = {{1,0,0,0,0,0,2,0,0,1},
									{0,1,2,1,2,0,4,0,0,2},
									{0,0,2,2,2,0,2,0,3,0},
									{1,1,0,2,0,2,0,0,1,2},
									{0,1,2,0,3,2,1,1,1,1},
									{0,2,1,2,1,0,1,2,0,0},
									{0,0,3,2,2,2,0,2,1,0},
									{1,2,0,0,3,2,0,1,0,1},
									{0,1,1,0,0,1,1,0,0,0},
									{1,1,2,1,0,0,0,2,0,0}};
		
		double[][] actualGLCM = null;
		calculationGLCM cG = new calculationGLCM(imageMatrix, "0", 1, false, false);
		actualGLCM = cG.getGLCM();
		boolean tag = true;
		
		for (int i = 0; i <10; i ++){
			for (int j = 0; j <10; j ++){
				if (expectedGLCM[i][j] - actualGLCM[i][j] != 0) 
					tag = false;
			}	
		}
			
		assertTrue(tag);
	}

}
