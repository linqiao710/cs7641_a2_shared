package opt;

import shared.Instance;

/**
 * A randomized hill climbing algorithm
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class RandomizedHillClimbing extends OptimizationAlgorithm {
    
    /**
     * The current optimization data
     */
    private Instance cur;
    
    /**
     * The current value of the data
     */
    private double curVal;

    private int restart=20;
    
    /**
     * Make a new randomized hill climbing
     */
    public RandomizedHillClimbing(HillClimbingProblem hcp, int restart) {
        super(hcp);
        cur = hcp.random();
        curVal = hcp.value(cur);
        this.restart = restart;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();
        Instance neigh = hcp.neighbor(cur);
        double neighVal = hcp.value(neigh);
        if (neighVal > curVal) {
            curVal = neighVal;
            cur = neigh;
        }else{
            while(this.restart>0){
                Instance tempCur = hcp.random();
                double tempCurVal = hcp.value(tempCur);
                this.restart-=1;

                if(tempCurVal>curVal){
                    curVal = tempCurVal;
                    cur = tempCur;
                    break;
                }

            }
        }
        return curVal;
    }

    /**
     * @see opt.OptimizationAlgorithm#getOptimalData()
     */
    public Instance getOptimal() {
        return cur;
    }

}
