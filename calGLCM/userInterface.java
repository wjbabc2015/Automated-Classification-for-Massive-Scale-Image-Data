package calGLCM;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class userInterface extends JFrame implements ActionListener {
	
	 JButton choice1;
	 JButton choice2;
	 
	 JTextField path;
	 
	 JLabel title;
	 JLabel pathTitle;
	 
	 JPanel mainUI;
	 
	 String pathName;
	 
	
	public userInterface (String name){
		
		this.pathName = name;
		
		mainUI = new JPanel ();
		
		this.add(mainUI);
		
		this.setSize(300, 200);
		
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		this.setLocation(dim.width/2-this.getSize().width/2, dim.height/2-this.getSize().height/2);
		
		title =new JLabel ("GLCM Texture Features: ");
		title.setFont(new Font("Serif", Font.ITALIC, 20));
		pathTitle = new JLabel ("PATH:");
		pathTitle.setFont(new Font("Serif", Font.BOLD, 10));
		choice1 = new JButton ("Single Angle Normalization");
		choice2 = new JButton ("GLCM and features calculation");
		choice1.setPreferredSize(new Dimension(250, 30));
		choice2.setPreferredSize(new Dimension(250, 30));
		
		path = new JTextField ();
		path.setPreferredSize(new Dimension(150, 30));
		path.setText(pathName);
		
		mainUI.add(title);
		//mainUI.add(choice1);
		mainUI.add(choice2);
		mainUI.add(pathTitle);
		mainUI.add(path);
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		//this.pack();
		this.setVisible(true);
		
		choice1.addActionListener(this);
		choice2.addActionListener(this);
		
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
/*		
		if (e.getSource() == choice1){
			singleNormalFrame sf = new singleNormalFrame(path.getText());
			this.dispose();
		}
*/		
		if (e.getSource() == choice2){
			mutipleNormalFrame mf = new mutipleNormalFrame(path.getText());
			this.dispose();
		}
	}
	
	public static void main(String[] args) {
		
		userInterface ui = new userInterface("C:/Users/wangj15/Desktop/Diffraction_Image/");
	}

}
