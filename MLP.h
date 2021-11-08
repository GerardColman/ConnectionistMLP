/*
    All array values are temporary until I figure that part out
    Might used linked list instead
*/

typedef struct MLP {
    int numberInputs;
    int numberHiddenUnits;
    int numberOutputs;
    double weightLower[10][10];
    double weightUpper[10][10];
    double weightChangeLower[10][10];
    double weightChangeUpper[10][10];
    double activationsLower[10];
    double activationsUpper[10];
    double hiddenUnits[10];
    double outputUnits[10];
} MultiLayedPerceptron;

void MLP_setup(MultiLayedPerceptron *MLP);

void MLP_randomise(MultiLayedPerceptron *MLP);

void MLP_forward(double i, MultiLayedPerceptron *MLP);

void MLP_double_backwards(double t, MultiLayedPerceptron *MLP);

void MLP_updateWeights(double learningRate, MultiLayedPerceptron *MLP);