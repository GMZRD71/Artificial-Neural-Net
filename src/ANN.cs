using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
// This class constructs the layers with all the corresponding neurons
// This class also does the training
{
    // Number of inputs coming into the neural network at the start
    public int numInputs;
    // Number of output(s); An ANN can indeed have more than one output; for example an action to perform and 
    // how fast to perform the action
    public int numOutputs;
    // Layers between the input and output layers
    public int numHidden;
    // Number of neurons per hidden layer
    public int numNPerHidden;
    // ALPHA
    // This is a value that determines how fast the neural network is going to learn.
    // Alpha determines how much any particular training sample is going to affect the overall
    // neural network
    // This allows control over how much percentage of the training set is going to be used each time by the neural net
    // each time that it learns
    // For example if you only want 50% of the training set to influence how the weights are changed, then make alpha = 0.5
    public double alpha;
    // List of all of the layers with all of the neurons
    List<Layer> layers = new List<Layer>();

    // CONSTRUCTOR for ANN Class
    public ANN(int nI, int nO, int nH, int nPH, double a)
    {
        // Number of inputs
        numInputs = nI;
        // Number of outputs
        numOutputs = nO;
        // Number of hidden layers
        numHidden = nH;
        // Number of neurons [perceptrons] per hidden layer
        // FOR NOW, MAKE THE NUMBER OF NEURONS PER HIDDEN LAYER THE SAME FOR ALL THE LAYERS
        numNPerHidden = nPH;
        // Alpha value
        alpha = a;

        // If we have hidden layers, 
        if (numHidden > 0)
        {
            // INPUT LAYER
            layers.Add(new Layer(numNPerHidden, numInputs));

            // HIDDEN LAYER(S)
            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            // OUTPUT LAYER
            layers.Add(new Layer(numOutputs, numNPerHidden));
        }
        else
        {
            // If there are no hidden layers...
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    // 'GO' - METHOD used for training the neural network
    // This calculates the errors
    public List<double> Go(List<double> inputValues, List<double> desiredOutput)
    {
        // Lists of inputs and outputs to keep track of for each neuron as these are found
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        // Make sure we received the correct number of input values
        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        // Loop through each layer (i), each neuron (j) and each neuron's input (k)
        // to multiply the inputs by the weights

        // Get the initial input values
        inputs = new List<double>(inputValues);

        // Go through each layer
        for(int i = 0; i < numHidden + 1; i++)
        {
            // If not dealing with layer 0 (the input layer),
            // or starting with the second time around the layer loop,
            // then set the inputs for the current layer as the
            // outputs from the previous layer
            if(i > 0)
            {
                inputs = new List<double>(outputs);
            }
            // Clear the outputs so these can be re-filled as we go through the
            // next process
            outputs.Clear();

            // Go through number of neurons in the current layer
            // IMPORTANT:
            // For each neuron, calculate its weight multiplied
            // by its input for each of its inputs and each of its
            // weights; this is because we have multiple inputs and multiple weights
            for(int j = 0; j < layers[i].numNeurons; j++)
            {
                // N is the number calculated products of inputs and weights
                double N = 0;

                // Now, clear the inputs to the current neuron (from prior iterations)
                layers[i].neurons[j].inputs.Clear();

                // Fill the inputs from the current layer with the outputs from the previous layer...
                // Loop around each neuron's input
                for(int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // For each neuron's input, we are adding in the input from the previous layer (as defined above)
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    // Now finally, multiply the weights by the inputs and add them all together
                    // A DOT Product
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }

                // NOW SUBTRACT THE BIAS
                //     --------
                N -= layers[i].neurons[j].bias;

                // Determine if we are working with hidden layers or
                // the output layer.

                if (i == numHidden)
                {
                    // OUTPUT LAYER
                    layers[i].neurons[j].output = ActivationFunctionO(N);
                }
                else
                {
                    // HIDDEN LAYER(S)
                    // Set the output for the neuron to whatever N is as
                    // interpreted by an Activation Function
                    layers[i].neurons[j].output = ActivationFunction(N);
                    // Now, simply add to the output
                    // So, for the next loop iteration, this will become the
                    // input to the next layer
                }
                outputs.Add(layers[i].neurons[j].output);
            }
        }

        // Take the error and apply it to all of the weights throughout the network
        UpdateWeights(outputs, desiredOutput);

        return outputs;
    }

    // BACK PROPAGATION LOOP
    // METHOD - UpdateWeights
    // Using BACKPROPAGATION (FEEDBACK)
    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        // Create an error variable to keep track of the error
        double error;
        // Feedback loop (back propagation)
        // That is, calculate the error at the end and feed it back
        // through the layers...
        for(int i = numHidden; i >= 0; i--)
        {
            // First layer's neuron
            for(int j = 0; j < layers[i].numNeurons; j++)
            {
                // If we are done with all the hidden layers or 
                // we are at the output layer,
                // then we can calculate the error:
                if(i == numHidden)
                {
                    error = desiredOutput[j] - outputs[j];
                    // Delta Rule for error calculation of the error gradient
                    // EACH INDIVIDUAL NEURON IS RESPONSIBLE FOR ONLY PART OF THE ERROR
                    // The gradient caculation is determining "how responsible" the error
                    // is for the current neuron[j]
                    // In other words, calculate the percentage that the current neuron
                    // is responsible for the total error
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                    // errorGradient calculated with Delta Rule: en.wikipedia.org/wiki/Delta_rule
                }
                else
                {
                    // Apply the Delta Rule to the error calculation again
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    // Also add an error gradient sum -- The error in the layer above the current layer
                    double errorGradSum = 0;
                    // For any given neuron in the layer before, loop through the neurons in the subsequent layer
                    for(int p = 0; p < layers[i+1].numNeurons; p++)
                    {
                        // Add the errors
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    // Add the error to the error gradient of the current neuron
                    // The value of the error increases as we move through the layers until reaching the output layer
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }
                // Loop through each of the inputs for the current neuron
                for(int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // If we are on the last layer (output layer)
                    if(i == numHidden)
                    {
                        // Calculate the error
                        error = desiredOutput[j] - outputs[j];
                        // Update the weight for this layer
                        // Alpha is the learning rate
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        // if not the output layer...
                        // Update the weights: learning rate times input for the current neuron times the errorGradient
                        // Recall that the error gradient is simply the amount of responsibility that neuron had for the total error
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }
                // Update the bias for every neuron
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }

    // ACTIVATION FUNCTIONS

    // HIDDEN LAYER(S)
    // NOTE: A list of activation functions are given in Wikipedia
    double ActivationFunction(double value)
    {
        // Select type of activation funcion
        // return Sigmoid(value); 
        return ReLu(value);
        // return Step(value);
    }

    // OUTPUT LAYER
    double ActivationFunctionO(double value)
    // This is the activation function for the Output layer
    //                                         ------
    {
        // Select type of activation funcion
        // return Sigmoid(value);
        return Sigmoid(value);
        // return Step(value);
    }

    double Step(double value)  // (Binary Step)
    {
        // The binary function is a very clear binary cutoff for the values
        if (value < 0) return 0;
        else return 1;
    }

    double Sigmoid(double value) // (Logistic Softstep or Sigmoid)
    {
        // double k = (double) System.Math.Exp(value);
        double k = Mathf.Exp((float) value);
        // This equation is different that the typical equation 1 / (1 + EXP(-value))
        // but it gives the same result
        // This output values from 0 to +1
        return k / (1.0f + k);
    }

    double TanH(double value)
    {
        // TanH is a simply a multiple of sigmoid, so we can use the a simpler
        // programmatic formula
        // The output values for this fuction are between -1 and +1
        // SO, use this activation function if we want outputs that have negative
        // values in it.
        return (2 * (Sigmoid(2 * value)) - 1);
    }

    double ReLu(double value)
    {
        // Values between 0 and +1
        if (value > 0) return value;
        else return 0;
    }

    double LeakyReLu(double value)
    {
        // Negative values are allowed and give a very slight Gradient/slope
        // with negative inputs give a very small value for the Gradient
        if (value < 0) return 0.01 * value;
        else return value;
    }

    double SinusoidActv(double value)
    {
        // Values range between -1 and +1
        // return System.Math.Sin(value);
        return Mathf.Sin((float) value);
    }

    double ArcTanActv(double value)
    {
        // Values range between -Pi/2 and +Pi/2
        // return System.Math.Atan(value);
        return Mathf.Atan((float) value);
    }

    double SoftSign(double value)
    {
        // Values range between -1 and +1
        // return value / (1 + System.Math.Abs(value));
        return value / (1 + Mathf.Abs((float) value));
    }
}
