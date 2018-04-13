import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoidDerivative;

import java.io.IOException;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class NN1 {


  public static void main(String[] args) throws IOException {
    INDArray w1 = Nd4j.rand(new int[]{28 * 28, 21});
    INDArray w2 = Nd4j.rand(new int[]{21, 21});
    INDArray w3 = Nd4j.rand(new int[]{21, 10});

    INDArray a1 = Nd4j.ones(new int[]{1, 21});
    INDArray a2 = Nd4j.ones(new int[]{1, 21});
    INDArray a3 = Nd4j.ones(new int[]{1, 10});

    DataSetIterator mnistTrain = new MnistDataSetIterator(1, true, 42);
    DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 42);
    mnistTrain.forEachRemaining(xy -> {
      INDArray x = xy.getFeatures();
      INDArray yTrue = xy.getLabels();

      forward(w1, w2, w3, x, a1, a2, a3);
      backward(w1, w2, w3, yTrue, a1, a2, a3);

      System.out.println(a2);
      INDArray loss = a3.sub(yTrue);
      System.out.println(yTrue);
      System.out.println(a3);
      System.out.println(loss.sumNumber());
    });


  }


  private static void backward(INDArray w1, INDArray w2, INDArray w3, INDArray yTrue, INDArray a1,
      INDArray a2, INDArray a3) {
    INDArray delta3 = sigmoidDerivative(a3.sub(yTrue));
    INDArray e3 = delta3.mmul(w3.transpose());

    INDArray delta2 = sigmoidDerivative(a2.sub(e3));
    INDArray e2 = delta2.mmul(w2.transpose());

    INDArray delta1 = sigmoidDerivative(a1.sub(e2));
    INDArray e1 = delta1.mmul(w1.transpose());

    Number lr = 0.001;

    w1 = a1.transpose().mmul(delta1).mul(lr);
    w2 = a2.transpose().mmul(delta2).mul(lr);
    w3 = a3.transpose().mmul(delta3).mul(lr);
  }

  private static void forward(INDArray w1, INDArray w2, INDArray w3, INDArray x,
      INDArray a1, INDArray a2, INDArray a3) {
    x = x.reshape(1, 28 * 28);
    a1 = sigmoid(x.mmul(w1));
    a2 = sigmoid(a1.mmul(w2));
    a3 = sigmoid(a2.mmul(w3));
  }

}