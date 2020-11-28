#include <iostream>
#include <bits/stdc++.h>
#include <cstdlib>
using namespace std;

//Gamma used for weight decaying to avoid overfitting and is variable according to the number of epochs.
float Gamma;
//Beta is also used for weight decaying to avoid overfitting which is constant and used as a decay factor for calculating the error of the output neuron.
float beta = 0.05;
//Learning rate for finding deltaW
float learning_rate = 0.5;

//Global variables of form n_ (number of) i.e epochs so it stores number of epochs.
//Connection type : 0 fully connected 1 partially connected take from user
//Activation type : 0 - Sigmoid; 1 - Relu; 2 - Linear
int n_epochs,n_inputs_rows=0,n_inputs_cols=0,n_hidden,n_neurons_layer,connection_type=0,activation_type=0,n_testing;

//This vector stores the hidden layer neuron connection pattern only if user opts partially connected else it is never used.
vector<vector<int>> neuronAdjacency;
//Stores the training data.
vector<vector<float>> trainingInput;
//Stores the testing data used for prediction
vector<vector<float>> testingInput;
//Expected array stores the expected values for the training data for calculating errors in backpropagation.
vector<float> expected;

//Input class for storing the values into data set and reading the inputs from the various input files.
class Input{
public: int i=0,j=0,k=0,l=0,b=0,m=0,a=0,t=0;
    float c=0;
        void input()
        {
          ifstream fin;
          fin.open("input3.txt");
          while(fin)
          {
            string line;
            getline(fin,line);
            stringstream ss(line);
            if(i==0)
            {
              i=1;
              if(ss>>j)
                n_inputs_rows = j;
              trainingInput.resize(n_inputs_rows);
              if(ss>>j)
                n_inputs_cols = j;
              if(ss>>j)
                n_testing = j;
              testingInput.resize(n_testing);
              if(ss>>j)
                n_hidden = j;
              if(ss>>j)
                n_neurons_layer = j;
              if(ss>>j)
                connection_type = j;
              if(ss>>j)
                activation_type = j;
              if(ss>>j)
                n_epochs = j;
              if(connection_type == 1)
              {
                neuronAdjacency.resize(n_neurons_layer);
              }
            }
            else if(k == 0)
            {
              while(ss>>c){
                trainingInput[b].push_back(c);
              }
              b++;
              if(b == n_inputs_rows)
                k=1;
            }
            else if(m == 0)
            {
              m = 1;
              while(ss>>c)
                expected.push_back(c);
            }
            else if(connection_type == 1)
            {
              while(ss>>j)
                neuronAdjacency[l].push_back(j);
              l++;
            }
            else if(a == 0)
            {
              while(ss>>c)
                testingInput[t].push_back(c);
              t++;
              if(t == n_testing)
                a = 1;
            }
          }
        }
};


//Structure for the input given by the name InputNeuron for storing the input value and the wights of the edges from it to the first layer of the hidden network.
//Weights are assigned random decimal values between 0 - 1(included) on creation of their instances at the start.
class InputNeuron{
public: vector <float> input_weights;
        float value = 0;
        InputNeuron()
        {
          input_weights.resize(n_neurons_layer);
          initializeWeights();
        }
        void initializeWeights()
        {
          int i;
          for(i=0;i<n_neurons_layer;i++)
          {
            input_weights[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
          }
        }
};

//Structure for the neurons of the hidden network where each neuron consists of its value, error and the bias along with array of weights from current neuron
//to the next layer of all neurons randomly initialised to decimal values between 0 and 1 at the start
class Neuron{
public: float value;
        vector<float> weights;
        int bias;
        float error;
        Neuron()
        {
          value=0;
          weights.resize(n_neurons_layer);
          initializeWeights();
          bias = rand() % 2;
        }
        void initializeWeights()
        {
          int i;
            for(i=0;i<n_neurons_layer;i++)
            {
              weights[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
};

//As we have assumed only one output we want the last hidden layer to have only one weight element in the weights array.
void RectifyLastLayer(vector<vector<Neuron>> &Network)
{
  int i,n = n_hidden-1;
  for(i=0;i<n_neurons_layer;i++)
  {
    Network[i][n].weights.resize(1);
  }
}

//Making the weights of unconnected neurons as -1 for showing no connections and hence considering them while calculating the output.
void convertToPartial(vector<vector<Neuron>> &Network)
{
  int i,j,k;
  for(i=0;i<n_hidden-1;i++)
  {
    for(j=0;j<n_neurons_layer;j++)
      {
      for(k=0;k<n_neurons_layer;k++)
        {
          if(neuronAdjacency[j][k] == 0)
              Network[j][i].weights[k] = -1;
        }
    }
  }
}

//Activation Functions :
//Sigmoid
float SigmoidActivation(float x)
{
  float a =  (float) 1 / (float)(1 + (exp(-1 * x)));
  return a;
}

//Rectified Linear Unit
int ReluActivation(float x)
{
  if(x > 0)
    return x;
  return 0;
}

//Linear Activation Function
float linearActivation(float x)
{
  return x;
}

//Function for assigning right activation function according to the choice of the user which he/she selected.
float ActivationType(float x)
{
  if(activation_type == 0)
  {
    return SigmoidActivation(x);
  }
  else if(activation_type == 1)
  {
    return (float)ReluActivation(x);
  }
  else if(activation_type == 2)
  {
    return linearActivation(x);
  }
  return 0;
}

//The FeedForwardNetwork where each neuron gets its input from previous layer by summation of multiplication of corresponding weights and values of neurons
//to which it is connected.
float feedForwardNetwork(vector<float> &ip,vector<InputNeuron> &Inp,vector<vector<Neuron>> &Network)
{
  int i,j,k;
  for(i=0;i<Inp.size();i++)
    Inp[i].value = ip[i];

  //From the input layer to first layer
  for(i=0;i<Network.size();i++)
  {
    float sum = 0;
    for(j=0;j<Inp.size();j++)
    {
         sum += (Inp[j].input_weights[i] * Inp[j].value);
    }
    //Adding bias
    sum += (float)Network[i][0].bias;
    //Assigning the value by passing it to Activation function of choice.
    Network[i][0].value = ActivationType(sum);
  }

  //For all the hidden layers.
  for(i=1;i<n_hidden;i++)
  {
    for(k=0;k<Network.size();k++) {
      float sum = 0;
        for (j = 0; j < Network.size(); j++) {
          if(Network[j][i - 1].weights[k] != -1)
            sum += (Network[j][i - 1].value * Network[j][i - 1].weights[k]);
        }
        sum += (float)Network[k][i].bias;
        Network[k][i].value = ActivationType(sum);
    }
  }

  //For the output layer.
  float output = 0;
  for(i=0;i<Network.size();i++)
  {
    output += (Network[i][n_hidden-1].value * Network[i][n_hidden-1].weights[0]);
  }
  //Return the output by passing it to Activation Function for normalization.
  return ActivationType(output);
}

//The transfer derivative value for calculating errors depends upon the type of activation user has chosen.
//Consists of derivative of the respective activation functions.
float outputDerivative(float x)
{
  if(activation_type == 0)
  {
    return (x * (1 - x));
  }
  else if(activation_type == 1)
  {
    if(x >= 0)
      return 1;
    return 0;
  }
  else if(activation_type == 2)
  {
    return 1;
  }
  return 0;
}

//Weight decay to avoid overfitting at higher epochs.
float overfitting(Neuron &n)
{
  int i;
  float sum = 0;
  for(i=0;i<n.weights.size();i++)
  {
    sum += n.weights[i];
  }
  float res = 2 * sum * Gamma;
  return res;
}

//Backpropagation Algorithm for propagating back errors and fine tuning the weights to predict the output properly.
void backPropagation(vector<InputNeuron> &Inp,vector<vector<Neuron>> &Network,float &output,float expect)
{
    int i,j,k;
    //Output for output neuron multiplied with constant term to avoid overfitting or more increase or decrase of values.
    float deltak = outputDerivative(output) * (expect - output) * (1 - 2*beta);
    for(i=0;i<n_neurons_layer;i++)
    {
        //Neuron Error
        Network[i][n_hidden-1].error = (outputDerivative(Network[i][n_hidden-1].value) * deltak * Network[i][n_hidden-1].weights[0]);
        //Delta W
        float deltaW = learning_rate * deltak  * Network[i][n_hidden-1].value;
        //Update the weight
        Network[i][n_hidden-1].weights[0] = Network[i][n_hidden-1].weights[0] + deltaW;
    }

    //For the hidden layers
    for(i=n_hidden-2;i>=0;i--)
    {
      for(j=0;j<n_neurons_layer;j++)
      {
        float sum=0;
        //Summation of all the (weights * error) of connected neurons in the next dense layer for error calculation
        for(k=0;k<n_neurons_layer;k++)
        {
          if(Network[j][i].weights[k] != -1)
          {
            sum += (Network[k][i+1].error * Network[j][i].weights[k]);
          }
        }
        //Calculating neuron error overfitting term to control values at higher epochs.
        float deltaJ = (outputDerivative(Network[j][i].value) * sum) - overfitting(Network[j][i]) ;
        Network[j][i].error = deltaJ;
        for(k=0;k<n_neurons_layer;k++)
        {
          if(Network[j][i].weights[k] != -1)
          {
            //Calculating Delta W and updating weights
            float deltaW = learning_rate * Network[k][i+1].error * Network[j][i].value;
            Network[j][i].weights[k] += deltaW;
          }
        }
      }
    }
    //Updating weights for the input layer
    for(i=0;i<Inp.size();i++)
    {
      float sum = 0;
      for(k=0;k<n_neurons_layer;k++)
      {
        float deltaW = learning_rate * Network[k][0].error * Inp[i].value;
        Inp[i].input_weights[k] += deltaW;
      }
    }
}

//Training function for the Network to train based on the training data
void train(vector<InputNeuron> &Inp,vector<vector<Neuron>> &Network)
{
  int i,j;
  for(i=0;i<n_epochs;i++)
  {
    //Gamma increase at higher epochs to avoid overfitting and redundant increase of weights.
    if(i < n_epochs/4)
    {
      Gamma = 0.005;
    }
    else if(i < n_epochs/2)
    {
      Gamma = 0.01;
    }
    else
    {
      Gamma = 0.05;
    }
    //In each epoch for all input rows feedforward and back propagation algorithm applied.
    for(j=0;j<n_inputs_rows;j++)
    {
      float  op = feedForwardNetwork(trainingInput[j],Inp,Network);
      backPropagation(Inp,Network,op,expected[j]);
    }
  }
}

//Predict function for finding output on testing data.
void predict(vector<InputNeuron> &Inp,vector<vector<Neuron>> &Network)
{
  int i,j;
  float error = 0;
  for(i=0;i<n_testing;i++)
  {
    for(j=0;j<testingInput[i].size();j++) { cout << testingInput[i][j] << " "; }
    cout<<"Output is : "<<round(feedForwardNetwork(testingInput[i],Inp,Network))<<"  Expected is : "<<expected[i]<<endl;
    //Summing all the errors by subtracting the output from expected.
    error += abs(round(expected[i]) - round(feedForwardNetwork(testingInput[i],Inp,Network)));
  }
  //And calculating the mean absolute error.
  cout<<"Error is : "<<abs(error/(float)n_inputs_rows) * 100<<"%"<<endl;
  //Also calculate the accuracy of our model in prediction.
  cout<<"Accuracy is : "<<100 - abs(error/(float)n_inputs_rows)*100<<"%"<<endl;
}

//This function calls the input method, initializes the inputNeuron vector of number of inputs and the 2-D matrix of hidden network where each layer is a
//column of the matrix.
void startTraining()
{
  int i,j;
  Input inp;
  inp.input();
  vector<InputNeuron> Inp(n_inputs_cols);
  vector<vector<Neuron>> Network(n_neurons_layer);
  for(i=0;i<n_neurons_layer;i++)
  {
    Network[i].resize(n_hidden);
  }
  RectifyLastLayer(Network);
  if(connection_type == 1)
    convertToPartial(Network);
  train(Inp,Network);
  cout<<"Prediction : "<<endl;
  predict(Inp,Network);
}

int main() {
  startTraining();
  return 0;
}