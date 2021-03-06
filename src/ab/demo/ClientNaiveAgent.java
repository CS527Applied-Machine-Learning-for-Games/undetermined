/*****************************************************************************
 ** ANGRYBIRDS AI AGENT FRAMEWORK
 ** Copyright (c) 2014, XiaoYu (Gary) Ge, Stephen Gould, Jochen Renz
 **  Sahan Abeyasinghe,Jim Keys,  Andrew Wang, Peng Zhang
 ** All rights reserved.
 **This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 **To view a copy of this license, visit http://www.gnu.org/licenses/
 *****************************************************************************/
package ab.demo;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileWriter;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;
import java.util.Random;


import javax.imageio.ImageIO;

import ab.demo.other.ClientActionRobot;
import ab.demo.other.ClientActionRobotJava;

import ab.planner.TrajectoryPlanner;

import ab.vision.GameStateExtractor.GameState;
import ab.vision.GameStateExtractor;
import ab.vision.Vision;
import ab.vision.VisionMBR;
//Naive agent (server/client version)

public class ClientNaiveAgent implements Runnable {


    //Wrapper of the communicating messages
    private ClientActionRobotJava ar;
    public byte currentLevel = 19; /**-1**/
    public int failedCounter = 0;
    public int[] solved;
    TrajectoryPlanner tp;
    private int id = 28887;
    public int idx = 1;
    private Random randomGenerator;
    int angleToShot;

    /**
     * Constructor using the default IP
     */
    public ClientNaiveAgent() {
        // the default ip is the localhost
        ar = new ClientActionRobotJava("127.0.0.1");
        tp = new TrajectoryPlanner();
        randomGenerator = new Random();
    }

    /**
     * Constructor with a specified IP
     */
    public ClientNaiveAgent(String ip) {
        ar = new ClientActionRobotJava(ip);
        tp = new TrajectoryPlanner();
        randomGenerator = new Random();

    }

    public ClientNaiveAgent(String ip, int id) {
        ar = new ClientActionRobotJava(ip);
        tp = new TrajectoryPlanner();
        randomGenerator = new Random();

        this.id = id;
    }

    public int getNextLevel() {
        int level = 0;
        boolean unsolved = false;
        //all the level have been solved, then get the first unsolved level
        for (int i = 0; i < solved.length; i++) {
            if (solved[i] == 0) {
                unsolved = true;
                level = i + 1;
                if (level <= currentLevel && currentLevel < solved.length)
                    continue;
                else
                    return level;
            }
        }
        if (unsolved)
            return level;
        level = (currentLevel + 1) % solved.length;
        if (level == 0)
            level = solved.length;
        return level;
    }

    /*
     * Run the Client (Naive Agent)
     */
    private void checkMyScore() {

        int[] scores = ar.checkMyScore();
        System.out.println(" My score: ");
        int level = 1;
        for (int i : scores) {
            System.out.println(" level " + level + "  " + i);
            if (i > 0)
                solved[level - 1] = 1;
            level++;
        }
    }

    public int[] savedaction = new int[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    public int ts = 0; // index for savedaction;
    public boolean Flag = false; // Flag is true when the action is not take and another one is resent
    public int Birdnums = 0; // This use to identify numbers of pigs in the screen
    public int Saveaction = 0;
    public int[] birdsnum = new int[]{3};
    public int stateidx = 0;

    public void run() {
        System.out.println("Client Starts Running");
        byte[] info = ar.configure(ClientActionRobot.intToByteArray(id));
        solved = new int[info[2]];
        checkMyScore();
        // shadow to fix level
        //currentLevel = (byte) getNextLevel();
        ar.loadLevel(currentLevel);
        GameState state;
        // capture Image
        BufferedImage screenshot = ar.doScreenShot();
        Birdnums = birdsnum[0];// Should Hard code number of birds to reduce the zoom times
        System.out.println("Number of Birds:" + Birdnums);
        int currbird = 0;

        while (true) {
            // Create New trojectory folder to save files
            String traj = "./Saves/traj" + idx;
            // Also create a txt file to save actions in this trajectory
            String ac = "./Saves/traj" + idx + "/actions";

            File directory = new File(traj);
            File acs = new File(ac);
           
            if (! directory.exists()){
                directory.mkdir();
                try {
                    acs.createNewFile();
                } catch (IOException e) {

                }
            }
            // Execute the action
            state = solve();
            currbird = ar.getNumberofBird();
            if (currbird != Birdnums){
                savedaction[ts] =  Saveaction;
                ts = ts + 1;
                Birdnums = Birdnums - 1;
            }

            // check numbers of birds

            if (state == GameState.WON) {

                System.out.println();
                //shadow to fix
                //currentLevel = (byte) getNextLevel();

                try {
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                screenshot = ar.doScreenShot();
                // Get reward
                int rds;
                GameStateExtractor st = new GameStateExtractor();

                rds = st.getScoreEndGame(screenshot);

                // Save the screen shot for the previous action and current state before replacing it with new one
                try {
                    File outImage = new File("./Temps/" + "idp+" + 777 + "_r+" + rds + "_Won.png");
                    File saveImage = new File("./Saves/traj" + idx + "/" + stateidx + "_" + rds + "_Won.png");
                    stateidx = 0;
                    ImageIO.write(screenshot, "png", saveImage);
                    ImageIO.write(screenshot, "png", outImage);
                    try {

                        FileWriter writer = new FileWriter("./Saves/traj" + idx + "/actions", true);

                        for (int i = 0; i < 10; i++) {
                            if (savedaction[i] == -1){
                                break;
                            }
                            writer.write(String.valueOf(savedaction[i]));
                            writer.write("\n");   // write new line
                        }
                        writer.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }


                } catch (IOException e) {

                }

                idx = idx + 1;

                // make a new trajectory planner whenever a new level is entered
                tp = new TrajectoryPlanner();

                angleToShot = -1;

                ar.loadLevel(currentLevel);
                Birdnums = birdsnum[0];
                savedaction = new int[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                ts = 0;
                // make a new trajectory planner whenever a new level is entered
                tp = new TrajectoryPlanner();


            } else
                //If lost, then restart the level
                if (state == GameState.LOST) {
                    screenshot = ar.doScreenShot();
                    int rds;
                    GameStateExtractor st = new GameStateExtractor();
                    rds = st.getScoreInGame(screenshot);
                    System.out.print(rds);
                    // Save the screen shot for the previous action and current state before replacing it with new one
                    try {
                        File outImage = new File("./Temps/" + "idp+" + 777 + "_r+" + rds + "_Lost.png");
                        File saveImage = new File("./Saves/traj" + idx + "/" + stateidx + "_" + rds + "_Lost.png");
                        stateidx = 0;
                        ImageIO.write(screenshot, "png", saveImage);
                        ImageIO.write(screenshot, "png", outImage);
                        try {

                            FileWriter writer = new FileWriter("./Saves/traj" + idx + "/actions", true);

                            for (int i = 0; i < 10; i++) {
                                if (savedaction[i] == -1){
                                    break;
                                }
                                writer.write(String.valueOf(savedaction[i]));
                                writer.write("\n");   // write new line
                            }
                            writer.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }


                    } catch (IOException e) {

                    }
                    idx = idx + 1;
                    tp = new TrajectoryPlanner();
                    angleToShot = -1;
                    //shadow to fix level
                    //currentLevel = (byte) getNextLevel();
                    ar.loadLevel(currentLevel);
                    Birdnums = birdsnum[0];
                    savedaction = new int[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    ts = 0;

                } else if (state == GameState.LEVEL_SELECTION) {
                    System.out.println("unexpected level selection page, go to the last current level : "
                            + currentLevel);
                    ar.loadLevel(currentLevel);
                    idx = idx + 1;
                    savedaction = new int[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    Birdnums = birdsnum[0];
                    ts = 0;
                } else if (state == GameState.MAIN_MENU) {
                    System.out
                            .println("unexpected main menu page, reload the level : "
                                    + currentLevel);
                    ar.loadLevel(currentLevel);
                    idx = idx + 1;
                    savedaction = new int[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    Birdnums = birdsnum[0];
                    ts = 0;
                } else if (state == GameState.EPISODE_MENU) {
                    System.out.println("unexpected episode menu page, reload the level: "
                            + currentLevel);
                    ar.loadLevel(currentLevel);
                    idx = idx + 1;
                    savedaction = new int[]{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
                    Birdnums = birdsnum[0];
                    ts = 0;
                }

        }
    }

    public GameState solve() {

        // capture Image
        BufferedImage screenshot = ar.doScreenShot();

        // process image
        Vision vision = new Vision(screenshot);

        // find the slingshot
        Rectangle sling = vision.findSlingshotMBR();

        // confirm the slingshot
        while (sling == null && ar.checkState() == GameState.PLAYING) {
            System.out.println("No slingshot detected. Please remove pop up or zoom out");
            ar.fullyZoomIn();
            screenshot = ar.doScreenShot();
            vision = new Vision(screenshot);
            sling = vision.findSlingshotMBR();
        }

        GameState gameState = ar.checkState();

        // if there is a sling, then play, otherwise just skip.
        if (sling != null) {
            Point releasePoint = null;
            int dx, dy;

            File foldercheck = new File("./Temps/");
            if (!foldercheck.exists()) {
                foldercheck.mkdir();
            }
            // Delete file if it exists after the shot was executed and before we store the current state
            try {
                Files.deleteIfExists(Paths.get("./Temps/action.txt"));

            } catch (NoSuchFileException e) {
                System.out.println("No such file/directory exists");
            } catch (DirectoryNotEmptyException e) {
                System.out.println("Directory is not empty.");
            } catch (IOException e) {
                System.out.println("Invalid permissions.");
            }

            // Store current state for DQN
            try {
                int rds;
                GameStateExtractor st = new GameStateExtractor();
                rds = st.getScoreInGame(screenshot);
                System.out.print(rds);
                File outImage = new File("./Temps/" + "idp+" + 777 + "_r+" + rds + ".png");
                File saveImage = new File("./Saves/traj" + idx + "/"+ stateidx + "_" + rds + ".png");
                stateidx += 1;
                ImageIO.write(screenshot, "png", saveImage);
                ImageIO.write(screenshot, "png", outImage);
            } catch (IOException e) {

            }

            // Decide on the next action (x y range to pick object from)
            angleToShot = -1;

            // Wait on python agent to decide on the action
            System.out.println("Waiting for action" + "\n");
            int safetyCheck = 0;
            while (angleToShot == -1) {
                File folder = new File("./Temps/");

                if (safetyCheck > 90 && folder.listFiles().length == 0) {
                    //python probably failed and deleted the state, place one more image

                    try {
                        int rds;
                        GameStateExtractor st = new GameStateExtractor();
                        rds = st.getScoreInGame(screenshot);
                        System.out.print(rds);
                        File outImage = new File("./Temps/" + "idp+" + 777 + "_r+" + rds + ".png");
                        ImageIO.write(screenshot, "png", outImage);
                    } catch (IOException e) {

                    }
                }
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                safetyCheck++;
                try (BufferedReader br = new BufferedReader(new FileReader("./Temps/action.txt"))) {

                    String line = br.readLine();
                    if (line != null) {
                        String[] params = line.split(" ");
                        int angle = Integer.parseInt(params[0]);
                        angleToShot = angle;
                    }

                } catch (IOException e) {

                } catch (NumberFormatException e) {

                }
            }
            System.out.println("Done Waiting for action: " + angleToShot);
            Saveaction = angleToShot;
            // Delete file
            try {
                Files.deleteIfExists(Paths.get("./Temps/action.txt"));

            } catch (NoSuchFileException e) {
                System.out.println("No such file/directory exists");
            } catch (DirectoryNotEmptyException e) {
                System.out.println("Directory is not empty.");
            } catch (IOException e) {
                System.out.println("Invalid permissions.");
            }

            releasePoint = tp.findReleasePoint(sling, Math.toRadians(angleToShot));
            // Get the reference point
            Point refPoint = tp.getReferencePoint(sling);


            //Calculate the tapping time according the bird type
            if (releasePoint != null) {
                int tapInterval = 0;
                switch (ar.getBirdTypeOnSling()) {

                    case RedBird:
                        tapInterval = 5;
                        break;               // start of trajectory
                    case YellowBird:
                        tapInterval = 65;
                        break; // 75% of the way
                    case WhiteBird:
                        tapInterval = 120;
                        break; // 80% of the way
                    case BlackBird:
                        tapInterval = 220;
                        break; // 70-90% of the way
                    case BlueBird:
                        tapInterval = 55;
                        break; // 65 of the way
                    default:
                        tapInterval = 60;
                }

                int tapTime = 20;
                dx = (int) releasePoint.getX() - refPoint.x;
                dy = (int) releasePoint.getY() - refPoint.y;
                ar.shoot(refPoint.x, refPoint.y, dx, dy, 0, tapTime * tapInterval, false);
            } else {
                System.err.println("No Release Point Found");
                return gameState;
            }


            gameState = ar.checkState();
        }
        return gameState;
    }

    public static void main(String args[]) {

        ClientNaiveAgent na;
        if (args.length > 0)
            na = new ClientNaiveAgent(args[0]);
        else
            na = new ClientNaiveAgent();
        na.run();

    }
}


