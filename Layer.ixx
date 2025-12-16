export module Layer;

import "Neuron"; // assuming Neuron is already a module
import <vector>;
import <iostream>;

export
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
        if(inputs.size() != m_num_inputs)
        {
            std::cerr << "Error: Input size mismatch. Expected " << m_num_inputs
                      << ", got " << inputs.size() << std::endl;
            return;
        }

        for(int i = 0; i < m_num_neurons; i++)
        {
            m_outputs[i] = m_neurons[i].Forward(inputs);
        }

        outputs = m_outputs;
    }

    void GetOutputs(std::vector<double>& outputs)
    {
        outputs = m_outputs;
    }

    int GetNeuronCount() const
    {
        return m_num_neurons;
    }

    int GetInputCount() const
    {
        return m_num_inputs;
    }

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

    double GetNeuronBias(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].bias;
    }

    void SetNeuronBias(int neuron_idx, double value)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return;
        }
        m_neurons[neuron_idx].bias = value;
    }

    double GetNeuronOutput(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].output;
    }

    double GetNeuronDelta(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return 0.0;
        }
        return m_neurons[neuron_idx].delta;
    }

    void SetNeuronDelta(int neuron_idx, double value)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return;
        }
        m_neurons[neuron_idx].delta = value;
    }

    ENUM_NEURON_ACTIVATION GetNeuronActivation(int neuron_idx)
    {
        if(neuron_idx < 0 || neuron_idx >= m_num_neurons)
        {
            std::cerr << "Error: Invalid neuron index" << std::endl;
            return NEURON_ACTIVATION_SIGMOID;
        }
        return m_neurons[neuron_idx].activation;
    }

    void SetActivation(ENUM_NEURON_ACTIVATION activation)
    {
        for(int i = 0; i < m_num_neurons; i++)
        {
            m_neurons[i].activation = activation;
        }
    }

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
