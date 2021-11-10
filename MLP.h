/*
    All array values are temporary until I figure that part out
    Might used linked list instead
*/
#define MAX_SIZE 5000

typedef struct MLP {
    int numberInputs;
    int numberHiddenUnits;
    int numberOutputs;
    double weightLower[MAX_SIZE][MAX_SIZE];
    double weightUpper[MAX_SIZE][MAX_SIZE];
    double weightChangeLower[MAX_SIZE][MAX_SIZE];
    double weightChangeUpper[MAX_SIZE][MAX_SIZE];
    double activationsLower[MAX_SIZE];
    double activationsUpper[MAX_SIZE];
    double hiddenUnits[MAX_SIZE];
    double outputUnits[MAX_SIZE];
} MultiLayedPerceptron;

void MLP_setup(MultiLayedPerceptron *MLP);

void MLP_randomise(MultiLayedPerceptron *MLP);

void MLP_forward(double i, MultiLayedPerceptron *MLP);

void MLP_double_backwards(double t, MultiLayedPerceptron *MLP);

void MLP_updateWeights(double learningRate, MultiLayedPerceptron *MLP);

void MLP_printMLP(MultiLayedPerceptron *MLP);