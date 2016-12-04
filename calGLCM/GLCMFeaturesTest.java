package calGLCM;

import static org.junit.Assert.*;

import org.junit.Test;

public class GLCMFeaturesTest {

	@Test
	public void test() {
		double[][] GLCMinput = {{1,0,0,0,0,0,2,0,0,1},
								{0,1,2,1,2,0,4,0,0,2},
								{0,0,2,2,2,0,2,0,3,0},
								{1,1,0,2,0,2,0,0,1,2},
								{0,1,2,0,3,2,1,1,1,1},
								{0,2,1,2,1,0,1,2,0,0},
								{0,0,3,2,2,2,0,2,1,0},
								{1,2,0,0,3,2,0,1,0,1},
								{0,1,1,0,0,1,1,0,0,0},
								{1,1,2,1,0,0,0,2,0,0}};
		
		double[] expectedResult = {174.0, 1432.0, 0.9999651291899748, 22.749198853416924, 290.0};
		double[] actualResult = null;
		GLCMFeatures gf = new GLCMFeatures(GLCMinput);
		actualResult = gf.getResult();
		
		assertTrue(actualResult[0] == expectedResult[0] && 
				actualResult[1] == expectedResult[1] &&
				actualResult[2] == expectedResult[2]&&
				actualResult[4] == expectedResult[3]&&
				actualResult[11] == expectedResult[4]);
		
	}		
}
