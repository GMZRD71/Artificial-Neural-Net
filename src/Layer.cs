using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    // Tell the layer how many neurons to create
    // in order to store the neuron created by the NEURON class
    public int numNeurons;
    // Store the neurons in a list of neurons and initialize
    // it at the very start of the program
    public List<Neuron> neurons = new List<Neuron>();

    // The number of Neuron Inputs is the number of Neurons in
    // the previous layer
    // 'LAYER' Class constructor
    public Layer(int nNeurons, int numNeuronInputs)
    {
        numNeurons = nNeurons;
        for (int i = 0; i < nNeurons; i++)
        {
            // Pass the number of neuron inputs to the next layer
            // these are initialized with random weights
            neurons.Add(new Neuron(numNeuronInputs));
        }
    }

}
