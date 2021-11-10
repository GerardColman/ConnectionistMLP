#include "MLP.h"
#include <stdio.h>

/*
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
*/

void MLP_setup(MultiLayedPerceptron *MLP){
    MLP->numberInputs = 10;
    MLP->numberHiddenUnits = 10;
    MLP->numberOutputs = 10;
}