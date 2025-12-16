export module Neuron;

import <vector>;
import <cmath>;
import <cstdlib>;
import <iostream>;

//---------------------------
// Activation function types
//---------------------------
export enum ENUM_NEURON_ACTIVATION
{
    NEURON_ACTIVATION_SIGMOID,    // Classic S-curve, smooth and gradual
    NEURON_ACTIVATION_RELU,       // Simple cutoff - zero or pass through
    NEURON_ACTIVATION_TANH,       // Like sigmoid but centered at zero
    NEURON_ACTIVATION_LEAKY_RELU  // ReLU but allows tiny negative values
};

//---------------------------
// Neuron struct
//---------------------------
export struct Neuron
{
    std::vector<double>    weights;      // How much each input matters
    double                 bias;         // The neuron's natural tendency
    double                 output;       // What it computed last time
    double                 delta;        // Error signal for learning
    ENUM_NEURON_ACTIVATION activation;   // Which math function to use

    //--- Constructor
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

    //--- Process inputs and calculate output
    double Forward(const std::vector<double>& inputs)
    {
        if(inputs.size() != weights.size())
        {
            std::cerr << "Error: Input size mismatch" << std::endl;
            return 0.0;
        }

        double sum = bias;
        for(size_t i = 0; i < weights.size(); i++)
        {
            sum += inputs[i] * weights[i];
        }

        switch(activation)
        {
            case NEURON_ACTIVATION_SIGMOID:
                output = 1.0 / (1.0 + exp(-sum));
                break;

            case NEURON_ACTIVATION_RELU:
                output = (sum > 0) ? sum : 0;
                break;

            case NEURON_ACTIVATION_TANH:
                output = tanh(sum);
                break;

            case NEURON_ACTIVATION_LEAKY_RELU:
                output = (sum > 0) ? sum : 0.01 * sum;
                break;
        }

        return output;
    }

    //--- Calculate activation function derivative
    double ActivationDerivative() const
    {
        switch(activation)
        {
            case NEURON_ACTIVATION_SIGMOID:
                return output * (1.0 - output);

            case NEURON_ACTIVATION_RELU:
                return (output > 0) ? 1.0 : 0.0;

            case NEURON_ACTIVATION_TANH:
                return 1.0 - output * output;

            case NEURON_ACTIVATION_LEAKY_RELU:
                return (output > 0) ? 1.0 : 0.01;
        }
        return 0.0;
    }
};
