package ab.demo;



import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import ab.planner.TrajectoryPlanner;
import ab.demo.other.ActionRobot;
import ab.demo.other.Shot;
import ab.vision.*;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.real.shape.Circle;

public class KnowledgeBasedAgentMain {
    private static ActionRobot ar;
    private static TrajectoryPlanner tp;

    public static void main(String[] args) {
        int level = Integer.parseInt(args[0]);
        ar = new ActionRobot();
        tp = new TrajectoryPlanner();
        ar.loadLevel(level);
        ar.click();

        while (ar.getState() == GameState.PLAYING){
            System.out.println("solve");
            GameState gs = solve();
        }
    }
    public static GameState solve(){
        // capture Image

        BufferedImage screenshot = ar.doScreenShot();

        // process image
        Vision vision = new Vision(screenshot);
        VisionRealShape vrs = new VisionRealShape(screenshot);
        // find the slingshot
        Rectangle sling = vision.findSlingshotMBR();

        // confirm the slingshot
        while (sling == null && ar.getState() == GameState.PLAYING) {
            System.out
                    .println("No slingshot detected. Retrying, please wait.");
            ActionRobot.fullyZoomIn();
            ActionRobot.fullyZoomOut();
            screenshot = ActionRobot.doScreenShot();
            vision = new Vision(screenshot);
            sling = vrs.findSling();

        }
        GameState state = ar.getState();


        if (sling != null) {
            Point refPoint = tp.getReferencePoint(sling);
            final List<ABObject> pigs = vision.findPigsRealShape();
            final List<ABObject> blocks = vision.findBlocksRealShape();
            final List<ABObject> tnts = vision.findTNTs();
            Point targetRock = findrocks(blocks, pigs);
            if(tnts.size()>0){
                //shoot tnt
                Point target = tnts.get(0).getCenter();
                for (ABObject tnt: tnts) {
                    Point p = tnt.getCenter();
                    System.out.println("x: "+p.x +", y: "+p.y);
                    if (p.y<target.y){
                        System.out.println("better");
                        System.out.println("x: "+p.x +", y: "+p.y);
                        target = p;
                    }

                }
                shoot(target, refPoint, sling);

            }else if(targetRock!=null){
                //shoot round rocks
                shoot(targetRock, refPoint, sling);
            }else{
                //shoot pigs
                shoot(pigs.get(0).getCenter(), refPoint, sling);
            }

        }
        return state;
    }
    public static Point findrocks(List<ABObject> blocks, List<ABObject> pigs){
        ArrayList<ABObject> roundRocks = new ArrayList<>();
        double maxR = 0;
        for (ABObject block: blocks) {
            if (block.shape == ABShape.Circle && block.type == ABType.Stone){
                System.out.println(((Circle)block).r);
//                if ( ((Circle)block).r> 5){
//
//                }
                if(maxR<((Circle)block).r){
                    maxR = ((Circle)block).r;
                }
                for (ABObject pig: pigs) {
                    if (isBelow(pig, block)){
                        roundRocks.add(block);
                        break;
                    }
                }

            }
        }
        if (roundRocks.isEmpty()){
            return null;
        }
        Point target = roundRocks.get(0).getCenter();
        for (ABObject roundRock: roundRocks) {
            if (target.y<roundRock.getCenter().y && ((Circle)roundRock).r == maxR){
                target = roundRock.getCenter();
            }
        }
        return target;
    }
    public static boolean isBelow(ABObject object1, ABObject object2) {
        if (object1.x == object2.x && object1.y == object2.y
                && object1.width == object2.width
                && object1.height == object2.height)
            return false;

        int o2down = object1.y + object1.height;

        if (o2down - 2 > object2.y)
            return false;

        return true;
    }
    public static void shoot(Point target,Point refPoint,Rectangle sling){
        ArrayList<Point> pts = tp.estimateLaunchPoint(sling, target);
        Point releasePoint = pts.get(0);
        int dx = (int)releasePoint.getX() - refPoint.x;
        int dy = (int)releasePoint.getY() - refPoint.y;
        int tapTime = getTapTime(sling, releasePoint, target);
        Shot shot = new Shot(refPoint.x, refPoint.y, dx, dy, 0, tapTime);
        ar.cshoot(shot);
        try {
            Thread.sleep(7000);
        } catch (InterruptedException e) {

            e.printStackTrace();
        }
    }
    public static int getTapTime(Rectangle sling, Point releasePoint, Point target){
        int tapInterval = 0;
        switch (ar.getBirdTypeOnSling())
        {

            case RedBird:
                tapInterval = 0;
                break;
            case YellowBird:
                tapInterval = 65;
                break;
            case WhiteBird:
                tapInterval =  70;
                break;
            case BlackBird:
                tapInterval =  70;
                break;
            case BlueBird:
                tapInterval =  65;
                break;
            default:
                tapInterval =  65;
        }
        int tapTime = tp.getTapTime(sling, releasePoint, target, tapInterval);
        return tapTime;
    }
}
