using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


// MonoBehaviour allows dynamic starts and updates.
// MonoBehaviour is for code that exists in the game environnment and needs to run.
// This Brain Class is the only class that needs MonoBehavior;
// The other three classes: ANN, Neuron and Layer only hold data and process data


public class Brain : MonoBehaviour
// MAIN CLASS - This class controls everything
// This class creates the neural network and trains it as well
{
    // Define an instance of the neural network class and call it "ann"
    ANN ann;
    // Sum Square Error is a statistical quantity that defines how closely
    // the model fits the data fed into the model
    double sumSquareError = 0;

    // Start is called before the first frame update
    void Start()
    {
        // Define the number of inputs, layers and neurons
        // on the hidden layers and the learning rate as follows:
        // Number of Inputs
        // Number of Outputs
        // Number of Hidden Layers
        // Number of Neurons Per Hidden Layer
        // Alpha or the learning rate (between 0 and 1, 0 is equivalent to no learning)
        // The higher the Alpha value, the quicker the network will run.
        ann = new ANN(2, 1, 1, 2, 0.7);

        // List keeping the results for every training line
        List<double> result;

        // Run the network for 200,000 ephocs for now
        for (int i = 0; i < 200000; i++)  // Maybe  use 500000
        {
            // Now define the training set for an XOR operation
            // Also calculate the Sum Square Error for each result
            // The power of 2
            // The result from the network minus the desired result
            // Add up the calculated square errors

            // TRAINING CODE - EXAMPLE 1
            // XOR OPERATION ****************************************************
            sumSquareError = 0;
            result = Train(1, 1, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0,2);
            result = Train(1, 0, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1,2);
            result = Train(0, 1, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1,2);
            result = Train(0, 0, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0,2);

            // TRAINING CODE - EXAMPLE 2
            // XNOR OPERATION ****************************************************
            /*
            sumSquareError = 0;
            result = Train(1, 1, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(1, 0, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(0, 1, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(0, 0, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            */

        }
        // We want the SSE to be quite small such as 0.0001
        // This is the accumulated final error
        Debug.Log("Final Value of Sum Square Error: " + sumSquareError);

        // NOTE:
        // This training code is not inside the Update function
        // because we do not want it to run over and over.

        // NOW, TEST THE NETWORK WITH 'FRESH' DATA:
        // ------------------------------------------------------------------------------
        Boolean RoundVals = true;

        // XOR OPERATION - EXAMPLE 1 ****************************************************
        result = Train(1, 1, 0);
        if (RoundVals == true)
            Debug.Log(" 1 1 " + Math.Round(result[0]));
        else
            Debug.Log(" 1 1 " + result[0]);

        result = Train(1, 0, 1);
        if (RoundVals == true)
            Debug.Log(" 1 0 " + Math.Round(result[0]));
        else
            Debug.Log(" 1 0 " + result[0]);

        result = Train(0, 1, 1);
        if (RoundVals == true)
            Debug.Log(" 0 1 " + Math.Round(result[0]));
        else
            Debug.Log(" 0 1 " + result[0]);

        result = Train(0, 0, 0);
        if (RoundVals == true)
            Debug.Log(" 0 0 " + Math.Round(result[0]));
        else
            Debug.Log(" 0 0 " + result[0]);

        // XNOR OPERATION - EXAMPLE 2 ****************************************************
        /*
        result = Train(1, 1, 1);
        if (RoundVals == true)
            Debug.Log(" 1 1 " + Math.Round(result[0]));
        else
            Debug.Log(" 1 1 " + result[0]);

        result = Train(1, 0, 0);
        if (RoundVals == true)
            Debug.Log(" 1 0 " + Math.Round(result[0]));
        else
            Debug.Log(" 1 0 " + result[0]);

        result = Train(0, 1, 0);
        if (RoundVals == true)
            Debug.Log(" 0 1 " + Math.Round(result[0]));
        else
            Debug.Log(" 0 1 " + result[0]);

        result = Train(0, 0, 1);
        if (RoundVals == true)
            Debug.Log(" 0 0 " + Math.Round(result[0]));
        else
            Debug.Log(" 0 0 " + result[0]);
        */
     //---------------------------------------------------------------------------------
    }

    List<double> Train(double i1, double i2, double o)
    // Train Class
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (ann.Go(inputs, outputs));
    }

    // Update is called once per frame
    void Update()
    {

    }
}
