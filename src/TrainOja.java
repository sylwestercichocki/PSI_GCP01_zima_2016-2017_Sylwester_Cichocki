import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.MLMethod;
import org.encog.ml.TrainingImplementationType;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.LearningRate;
import org.encog.neural.networks.training.propagation.TrainingContinuation;

import java.util.Iterator;

/**
 * Created by Sylwek on 2017-02-01.
 */
public class TrainOja extends BasicTraining implements LearningRate {
    private final BasicNetwork network;
    private final MLDataSet training;
    private double learningRate;

    public TrainOja(BasicNetwork network, MLDataSet training, double learningRate){
        super(TrainingImplementationType.Iterative);
        if(network.getLayerCount()>2){
            throw new NeuralNetworkError("to many layers");
        }else{
            this.network=network;
            this.training=training;
            this.learningRate=learningRate;
        }
    }


    @Override
    public boolean canContinue() {
        return false;
    }
    @Override
    public double getLearningRate() {
        return this.learningRate;
    }
    @Override
    public MLMethod getMethod() {
        return this.network;
    }


    @Override
    public void iteration() {
        ErrorCalculation errorCalculation = new ErrorCalculation();
        Iterator i$ = this.training.iterator();

        while(i$.hasNext()) {
            MLDataPair pair = (MLDataPair) i$.next();
            MLData output = this.network.compute(pair.getInput());
            double ideal = pair.getIdeal().getData(0);
//            System.out.println(output);
            for(int currentHebbian = 0; currentHebbian < output.size(); ++currentHebbian) {
                // double diff = pair.getIdeal().getData(currentHebbian) - output.getData(currentHebbian);

                for(int i = 0; i <= this.network.getInputCount(); ++i) {
                    double input;
                    if(i == this.network.getInputCount()) {
                        input = 1.0D;
                    } else {
                        input = pair.getInput().getData(i);
                    }
                    this.network.setWeight(0, i, currentHebbian, this.learningRate * output.getData(0) *( input -output.getData(0)* network.getWeight(0, i, currentHebbian)));
//                    this.network.addWeight(0, i, currentHebbian, (this.learningRate * ideal * input - network.getWeight(0, i, currentHebbian)*forgotRate));
                }
            }



            errorCalculation.updateError(output.getData(), pair.getIdeal().getData(), pair.getSignificance());
        }
        this.setError(errorCalculation.calculate());
    }


    @Override
    public TrainingContinuation pause() {
        return null;
    }

    @Override
    public void resume(TrainingContinuation trainingContinuation) {

    }

    @Override
    public void setLearningRate(double rate) {
        this.learningRate=rate;
    }

}
