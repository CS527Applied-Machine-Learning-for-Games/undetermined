package ab.demo;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ab.demo.other.ActionRobot;
import ab.demo.other.Shot;
import ab.planner.TrajectoryPlanner;
import ab.utils.StateUtil;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.Vision;

import javax.imageio.ImageIO;
import java.util.Random;

public class DataCollectionAgent implements Runnable {
    private ActionRobot aRobot;
    //private Random randomGenerator;
    public int currentLevel = 6;
    String pathname = "";
    TrajectoryPlanner tp;
    boolean initial = false;
    public DataCollectionAgent() {
        this.aRobot = new ActionRobot();
        this.tp = new TrajectoryPlanner();
    }

    @Override
    public void run() {
        aRobot.loadLevel(currentLevel);
        shoot(35, 0);
        shoot(24, 0);
//        pathname = System.getProperty("user.dir")+"\\data\\"+currentLevel + "best";
//        System.out.println("Working Directory = " + System.getProperty("user.dir"));
//        boolean b= new File(pathname).mkdirs();
//        System.out.println("create directory " + b);
//        File data = new File(pathname + "\\" + "data.txt");
//        try {
//            FileWriter myWriter = new FileWriter(data);
//            for (double angle = 15; angle <= 90; angle++) {
//
////                for (int taptime = 60; taptime <= 90; taptime+=10) {
////                    try {
////                        int score = collect(angle, taptime);
////                        myWriter.append(angle+" "+score+"\n");
////                    } catch (IOException e) {
////                        e.printStackTrace();
////                    }
////                }
//
//                try {
//                    shoot(35, 0);
//                    int score = collect(angle, 0);
//                    myWriter.append(angle+" "+score+"\n");
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//                aRobot.restartLevel();
//            }
//            myWriter.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        System.out.println("finished");
    }

    private int collect(double angle, int taptime) throws IOException {

        // capture Image
        aRobot.click();
        ActionRobot.fullyZoomIn();
        ActionRobot.fullyZoomOut();

        BufferedImage screenshot = ActionRobot.doScreenShot();
        if (!initial) {
            File outputfile = new File(pathname + "\\" + "initial.png");
            ImageIO.write(screenshot, "png", outputfile);
        }

        // process image
        Vision vision = new Vision(screenshot);
        // find the slingshot
        Rectangle sling = vision.findSlingshotMBR();
        // confirm the slingshot
        while (sling == null && aRobot.getState() == GameState.PLAYING) {
            System.out
                    .println("No slingshot detected. Please remove pop up or zoom out");

            ActionRobot.fullyZoomOut();
            screenshot = ActionRobot.doScreenShot();
            vision = new Vision(screenshot);
            sling = vision.findSlingshotMBR();
        }
        int dx = 0,dy;
        Shot shot = new Shot();
        boolean tap = false;
        if (sling!=null) {
            Point releasePoint = tp.findReleasePoint(sling, Math.toRadians(angle));
            // Get the reference point
            Point refPoint = tp.getReferencePoint(sling);
            System.out.println("Release Point: " + releasePoint);
            System.out.println("Release Angle: "
                    + angle);

            dx = (int)releasePoint.getX() - refPoint.x;
            dy = (int)releasePoint.getY() - refPoint.y;
            if (aRobot.getBirdTypeOnSling()!= ABType.RedBird) {
                shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, taptime);
                tap = true;
            }else {
                shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, 0);
            }


        }
        ActionRobot.fullyZoomOut();
        screenshot = ActionRobot.doScreenShot();
        vision = new Vision(screenshot);
        Rectangle _sling = vision.findSlingshotMBR();
        if(_sling != null)
        {
            double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
            if(scale_diff < 25)
            {
                if(dx < 0)
                {

                    aRobot.cshoot(shot);
                    screenshot = ActionRobot.doScreenShot();
                }else{
                    System.out.println("cant shoot");
                }
            }
            else
                System.out.println("Scale is changed, can not execute the shot");
        }
        try {
            Thread.sleep(8000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        int score = StateUtil.getScore(ActionRobot.proxy);
        System.out.println("a shot with angle " + angle + " scored : " + score);
        File outputfile;
        if (tap) {
            outputfile = new File(pathname + "\\" + angle + "_" + score +"_"+taptime+ ".png");
        }else {
            outputfile = new File(pathname + "\\" + angle + "_" + score + ".png");
        }
        ImageIO.write(screenshot, "png", outputfile);
        return score;
    }

    private void shoot(double angle, int tapTime) {
        ActionRobot.fullyZoomOut();

        BufferedImage screenshot = ActionRobot.doScreenShot();
        // process image
        Vision vision = new Vision(screenshot);
        // find the slingshot
        Rectangle sling = vision.findSlingshotMBR();
        // confirm the slingshot
        while (sling == null && aRobot.getState() == GameState.PLAYING) {
            System.out
                    .println("No slingshot detected. Please remove pop up or zoom out");
            aRobot.click();
            ActionRobot.fullyZoomIn();
            ActionRobot.fullyZoomOut();
            aRobot.click();
            screenshot = ActionRobot.doScreenShot();
            vision = new Vision(screenshot);
            sling = vision.findSlingshotMBR();
        }
        int dx = 0,dy;
        Shot shot = new Shot();
        boolean tap = false;
        if (sling!=null) {
            Point releasePoint = tp.findReleasePoint(sling, Math.toRadians(angle));
            // Get the reference point
            Point refPoint = tp.getReferencePoint(sling);
            System.out.println("Release Point: " + releasePoint);
            System.out.println("Release Angle: "
                    + angle);

            dx = (int)releasePoint.getX() - refPoint.x;
            dy = (int)releasePoint.getY() - refPoint.y;
            if (aRobot.getBirdTypeOnSling()!= ABType.RedBird) {
                shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, tapTime);
                tap = true;
            }else {
                shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, 0);
            }


        }
        ActionRobot.fullyZoomOut();
        screenshot = ActionRobot.doScreenShot();
        vision = new Vision(screenshot);
        Rectangle _sling = vision.findSlingshotMBR();
        if(_sling != null)
        {
            double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
            if(scale_diff < 25)
            {
                if(dx < 0)
                {

                    aRobot.cshoot(shot);
                }else{
                    System.out.println("cant shoot");
                }
            }
            else
                System.out.println("Scale is changed, can not execute the shot");
        }
    }
}
