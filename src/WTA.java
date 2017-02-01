import org.encog.engine.network.activation.ActivationFunction;
import org.encog.mathutil.rbf.RBFEnum;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodFunction;
import org.encog.util.csv.CSVFormat;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Map;

/**
 * Created by Sylwek on 2016-12-19.
 */
public class WTA {

    public static void main(final String args[]) throws FileNotFoundException{
        File file = new File("Data.txt");
        File test_File = new File("Data_test.txt");
        System.out.println("Loading data");
        DataSet dataSet = new DataSet(file, test_File);
        System.out.println("Dataset loaded");

//        NeighborhoodFunction neighborhoodFunction;
//        SOM network = new SOM(26,26);

        EncogKohonenNetwork encogKohonenNetwork = new EncogKohonenNetwork(
                NeighborhoodFunctionType.BUBLLE, RBFEnum.InverseMultiquadric,
                2, 0.7, 0.01, false, 0);

        encogKohonenNetwork.buildAndTrainNetwork(dataSet);
        System.out.println(encogKohonenNetwork.getLayerLayout());

        double[] q;

        q = encogKohonenNetwork.evaluate(dataSet.getInput()[0],100);

//        for (double z:q ) {
//            System.out.println(z);
//        }

        Map<Integer, Double[]> out;
        out = encogKohonenNetwork.getOutput();

        Double[] d;
        for(int i=0; i<26;i++){
            d = out.get(i);
            for(int j=0; j<26;j++){
                System.out.print(d[i] + " ");
            }
            System.out.println();
        }

       // BasicTrainSOM train = new BasicTrainSOM(network,0.01, data);

    }
}
