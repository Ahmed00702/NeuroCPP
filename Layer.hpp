#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <vector>
#include <iostream>

//+------------------------------------------------------------------+
//| Layer class - think of it as a team of neurons working together |
//+------------------------------------------------------------------+
class Layer
{
private:
    std::vector<Neuron>     m_neurons;       // Our crew of neurons
    int                     m_num_neurons;   // How many neurons we've got
    int                     m_num_inputs;    // How many inputs each neuron expects
    std::vector<double>     m_outputs;       // What each neuron spits out
    
public:
    //--- Basic constructor - does nothing fancy
    Layer() : m_num_neurons(0), m_num_inputs(0) { }
    
    //--- Real constructor - sets everything up
    Layer(int num_neurons, int num_inputs, ENUM_NEURON_ACTIVATION activation = NEURON_ACTIVATION_SIGMOID)
        : m_num_neurons(num_neurons), m_num_inputs(num_inputs)
    {
        m_neurons.reserve(num_neurons);
        m_outputs.resize(num_neurons);
        
        // Create each neuron with the same activation function
        for(int i = 0; i < num_neurons; i++)
        {
            m_neurons.push_back(Neuron(activation));
        }
    }
    
    //--- Give all neurons their random starting weights
    void Init()
    {
        for(int i = 0; i < m_num_neurons; i++)
        {
            m_neurons[i].Init(m_num_inputs);
        }
    }
    
    //--- Run data through the layer and get results
    void Forward(const std::vector<double>& inputs, std::vector<double>& outputs)
    {
        // Make sure we got the right amount of inputs
        if(inputs.size() != m_num_inputs)
        {
            std::cerr << "Error: Input size mismatch. Expected " << m_num_inputs 
                      << ", got " << inputs.size() << std::endl;
            return;
        }
        
        // Let each neuron do its thing
        for(int i = 0; i < m_num_neurons; i++)
        {
            m_outputs[i] = m_neurons[i].Forward(inputs);
        }
        
        // Hand back the results
        outputs = m_outputs;
    }
    
    //--- Grab the outputs from last time we ran Forward
    void GetOutputs(std::vector<double>& outputs)
    {
        outputs = m_outputs;
    }
    
    //--- How many neurons are in this layer?
    int GetNeuronCount() const
    {
        return m_num_neurons;
    }
    
    //--- How many inputs does each neuron take?
    int GetInputCount() const
    {
        return m_num_inputs;
    }
    
    //--- Get a specific weight from a specific neuron
    double GetNeuronWeight(int neuron_idx, int weight_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons || 
           weight_idx < 0 || weight_idx >= m_num_inputs)
        {
            std::cerr << "Error: Invalid neuron or weight index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].weights[weight_idx];
    }
    
    //--- Change a specific weight in a specific neuron
    void SetNeuronWeight(int neuron_idx, int weight_idx, double value)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons || 
           weight_idx < 0 || weight_idx >= m_num_inputs)
        {
            std::cerr << "Error: Invalid neuron or weight index" << std::endl;
            return;
        }
        m_neurons[neuron_idx].weights[weight_idx] = value;
    }
    
    //--- What's the bias on this neuron?
    double GetNeuronBias(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].bias;
    }
    
    //--- Update a neuron's bias
    void SetNeuronBias(int neuron_idx, double value)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return;
        }
        m_neurons[neuron_idx].bias = value;
    }
    
    //--- What did this neuron output last time?
    double GetNeuronOutput(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].output;
    }
    
    //--- Get the error term for backpropagation
    double GetNeuronDelta(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].delta;
    }
    
    //--- Set the error term for backpropagation
    void SetNeuronDelta(int neuron_idx, double value)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return;
        }
        m_neurons[neuron_idx].delta = value;
    }
    
    //--- What activation function is this neuron using?
    ENUM_NEURON_ACTIVATION GetNeuronActivation(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return NEURON_ACTIVATION_SIGMOID;
        }
        return m_neurons[neuron_idx].activation;
    }
    
    //--- Change activation function for the whole layer
    void SetActivation(ENUM_NEURON_ACTIVATION activation)
    {
        for(int i = 0; i < m_num_neurons; i++)
        {
            m_neurons[i].activation = activation;
        }
    }
    
    //--- Change activation function for just one neuron
    void SetNeuronActivation(int neuron_idx, ENUM_NEURON_ACTIVATION activation)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return;
        }
        m_neurons[neuron_idx].activation = activation;
    }
};

#endif // LAYER_H
