#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

//+------------------------------------------------------------------+
//| Activation function types - different ways neurons can "fire"   |
//+------------------------------------------------------------------+
enum ENUM_NEURON_ACTIVATION
{
    NEURON_ACTIVATION_SIGMOID,    // Classic S-curve, smooth and gradual
    NEURON_ACTIVATION_RELU,       // Simple cutoff - zero or pass through
    NEURON_ACTIVATION_TANH,       // Like sigmoid but centered at zero
    NEURON_ACTIVATION_LEAKY_RELU  // ReLU but allows tiny negative values
};

//+------------------------------------------------------------------+
//| Neuron structure - the basic building block of the network      |
//+------------------------------------------------------------------+
struct Neuron
{
    std::vector<double>    weights;      // How much each input matters
    double                 bias;         // The neuron's natural tendency
    double                 output;       // What it computed last time
    double                 delta;        // Error signal for learning
    ENUM_NEURON_ACTIVATION activation;   // Which math function to use
    
    //--- Constructor - set up a new neuron
    Neuron(ENUM_NEURON_ACTIVATION act_func = NEURON_ACTIVATION_SIGMOID)
        : activation(act_func), bias(0.0), output(0.0), delta(0.0)
    {
    }
    
    //--- Give the neuron random starting weights
    void Init(int num_inputs)
    {
        weights.resize(num_inputs);
        for(int i = 0; i < num_inputs; i++)
        {
            // Random value between -1 and 1
            weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
        bias = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    
    //--- Process inputs and calculate what this neuron outputs
    double Forward(const std::vector<double>& inputs)
    {
        // Sanity check - make sure we have the right number of inputs
        if(inputs.size() != weights.size())
        {
            std::cerr << "Error: Input size mismatch" << std::endl;
            return 0.0;
        }
        
        // Start with the bias, then add weighted inputs
        double sum = bias;
        for(size_t i = 0; i < weights.size(); i++)
        {
            sum += inputs[i] * weights[i];
        }
        
        // Run it through the activation function
        switch(activation)
        {
            case NEURON_ACTIVATION_SIGMOID:
                // Squashes everything between 0 and 1
                output = 1.0 / (1.0 + exp(-sum));
                break;
                
            case NEURON_ACTIVATION_RELU:
                // If positive keep it, otherwise zero
                output = (sum > 0) ? sum : 0;
                break;
                
            case NEURON_ACTIVATION_TANH:
                // Squashes between -1 and 1
                output = tanh(sum);
                break;
                
            case NEURON_ACTIVATION_LEAKY_RELU:
                // Like ReLU but lets 1% through when negative
                output = (sum > 0) ? sum : 0.01 * sum;
                break;
        }
        
        return output;
    }
    
    //--- Calculate how much the activation function changes at current output
    double ActivationDerivative() const
    {
        switch(activation)
        {
            case NEURON_ACTIVATION_SIGMOID:
                // Derivative has this neat property
                return output * (1.0 - output);
                
            case NEURON_ACTIVATION_RELU:
                // Either 1 or 0, no in-between
                return (output > 0) ? 1.0 : 0.0;
                
            case NEURON_ACTIVATION_TANH:
                // Another elegant derivative
                return 1.0 - output * output;
                
            case NEURON_ACTIVATION_LEAKY_RELU:
                // Mostly 1, but 0.01 for negative
                return (output > 0) ? 1.0 : 0.01;
        }
        return 0.0;
    }
};

#endif // NEURON_H
