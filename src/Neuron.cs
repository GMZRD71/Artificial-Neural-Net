using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// NEURON MAIN FUNCTION IS TO HOLD VALUES
// A PERCEPTRON...
public class Neuron 
{
    // ANN
    // ---
    // Every neuron in one layer is connected to every-other neuron in the next layer...

    public int numInputs;          // Number of Inputs.
    public double bias;            // Bias
    public double output;          // Output
    public double errorGradient;   //
    // List of weights
    public List<double> weights = new List<double>();
    // List of inputs either from outside the NN or from the previous layer
    public List<double> inputs = new List<double>();

    public Neuron(int nInputs)
    {
        // At the start, set the bias to a random number between -1.0 and 1.0
        bias = UnityEngine.Random.Range(-1.0f, 1.0f);   
        numInputs = nInputs;
        for (int i = 0; i < nInputs; i++)
            // Initialize the weights to a random number
            weights.Add(UnityEngine.Random.Range(-1.0f, 1.0f));
    }
}
