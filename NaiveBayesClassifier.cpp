#include <iostream>
#include <string>
#include <vector>
#include "NaiveBayesClassifier.h"
#include <fstream>
#include <sstream>
#include <cmath>
//#include <chrono>

using namespace std;

NaiveBayesClassifier::NaiveBayesClassifier(){
}

//calls function for each feature to update its mean and std. var
void NaiveBayesClassifier::updateFeatureCDF(double age, string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade){

    //updates each individual feature based on grade
    if(grade == 1){
        updateFeatureUtil(&age_y, age);
        updateGenderUtil(&gender_y, gender);
        updateFeatureUtil(&height_y, height);
        updateFeatureUtil(&weight_y, weight);
        updateFeatureUtil(&bodyfat_y, bodyfat);
        updateFeatureUtil(&diastolic_y, diastolic);
        updateFeatureUtil(&systolic_y, systolic);
        updateFeatureUtil(&grip_force_y, grip_force);
        updateFeatureUtil(&sit_and_bend_forward_y, sit_and_bend_forward);
        updateFeatureUtil(&sit_up_count_y, sit_up_count);
        updateFeatureUtil(&broad_jump_y, broad_jump);
    }
    else{
        updateFeatureUtil(&age_n, age);
        updateGenderUtil(&gender_n, gender);
        updateFeatureUtil(&height_n, height);
        updateFeatureUtil(&weight_n, weight);
        updateFeatureUtil(&bodyfat_n, bodyfat);
        updateFeatureUtil(&diastolic_n, diastolic);
        updateFeatureUtil(&systolic_n, systolic);
        updateFeatureUtil(&grip_force_n, grip_force);
        updateFeatureUtil(&sit_and_bend_forward_n, sit_and_bend_forward);
        updateFeatureUtil(&sit_up_count_n, sit_up_count);
        updateFeatureUtil(&broad_jump_n, broad_jump);
    }
}

//sets the feature values means to be the initial values for each feature
void NaiveBayesClassifier::setFeatureCDF(double age, string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade){

    //initializes mean values for grade = 1 and 0
    if(grade == 1){
        age_y.mean = age;

        if(gender == "M"){
            gender_y.males = 1;
        }

        height_y.mean = height;
        
        weight_y.mean = weight;

        bodyfat_y.mean = bodyfat;

        diastolic_y.mean = diastolic;
        
        systolic_y.mean = systolic;
        
        grip_force_y.mean = grip_force;
        
        sit_and_bend_forward_y.mean = sit_and_bend_forward;
        
        sit_up_count_y.mean = sit_up_count;
        
        broad_jump_y.mean = broad_jump;
    }
    else{
        age_n.mean = age;
        
        if(gender == "M"){
            gender_n.males = 1;
        }

        height_n.mean = height;
        
        weight_n.mean = weight;
        
        bodyfat_n.mean = bodyfat;
        
        diastolic_n.mean = diastolic;
        
        systolic_n.mean = systolic;
        
        grip_force_n.mean = grip_force;
        
        sit_and_bend_forward_n.mean = sit_and_bend_forward;
        
        sit_up_count_n.mean = sit_up_count;
        
        broad_jump_n.mean = broad_jump;
    }

}

//uses equation to calculate updated mean and std. var
void NaiveBayesClassifier::updateFeatureUtil(features *feature, double x){
    feature->n = feature->n+1;
    feature->oldMean = feature->mean;
    //calculates a new mean from an old mean and new value
    feature->mean = ((feature->mean * (feature->n-1))+x)/feature->n;
    //calculates a new standard variation from old standard variation and mean
    feature->stdVar = sqrt( ( (feature->n-2)*feature->stdVar*feature->stdVar + (x - feature->mean)*(x - feature->oldMean) )/(feature->n-1));
}

//updates gender mean
void NaiveBayesClassifier::updateGenderUtil(genderFeature *gender, string genderValue){
    gender->n = gender->n+1.0;
    if(genderValue == "M"){
        gender->males = gender->males+1.0;
    }
}

//prints a features mean, number of athletes, and standard variation
void NaiveBayesClassifier::printFeatureAttributes(features feature){
    cout<<"Mean: "<<feature.mean<<"\n";
    cout<<"N: "<<feature.n<<"\n";
    cout<<"Standard Deviation: "<<feature.stdVar<<"\n";
}

features NaiveBayesClassifier::getFeature(){
    return height_n;
}

//calculate zscore from mean and std. var
double NaiveBayesClassifier::zScoreCalc(features feature, double x){
    double zscore = (x - feature.mean)/feature.stdVar;

    return 0.5 * erfc(-zscore * M_SQRT1_2);
}

//calculates probability of being within the range
double NaiveBayesClassifier::rangeProb(features feature, double x, double ceoff){
    //can change range here to test accuracy
    //2.3 seems to be best
    return zScoreCalc(feature, x + (ceoff*feature.stdVar)) - zScoreCalc(feature, x - (ceoff*feature.stdVar));
}

//calculates probability of gender
double NaiveBayesClassifier::genderProb(genderFeature feature, string gender){
    
    if(gender == "M"){
        return feature.males/feature.n;
    }
    else{
        return (feature.n - feature.males)/feature.n;
    }
}


int NaiveBayesClassifier::makeGuess(double age, string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade){
    
    //calculated weights from multiple tests
    double ageInt = 3.63; 
    double heightInt = 2.53; 
    double weightInt = 4.6;  
    double fatInt = 2.7; 
    double diaInt = 2.6; 
    double sysInt = 3; 
    double gripInt = 2.76; 
    double sitbendInt = 2.38; 
    double situpInt = 2.29; 
    double broadInt = 2.8; 

    //calculates probability for each feature
    double age_y_prob = rangeProb(age_y, age, ageInt);
    double gender_y_prob = genderProb(gender_y, gender);
    double height_y_prob = rangeProb(height_y, height, heightInt);
    double weight_y_prob = rangeProb(weight_y, weight, weightInt);
    double bodyfat_y_prob = rangeProb(bodyfat_y, bodyfat, fatInt);
    double diastolic_y_prob = rangeProb(diastolic_y, diastolic, diaInt);
    double systolic_y_prob = rangeProb(systolic_y, systolic, sysInt);
    double grip_force_y_prob = rangeProb(grip_force_y, grip_force, gripInt);
    double sit_and_bend_forward_y_prob = rangeProb(sit_and_bend_forward_y, sit_and_bend_forward, sitbendInt);
    double sit_up_count_y_prob = rangeProb(sit_up_count_y, sit_up_count, situpInt);
    double broad_jump_y_prob = rangeProb(broad_jump_y, broad_jump, broadInt);

    double age_n_prob = rangeProb(age_n, age, ageInt);
    double gender_n_prob = genderProb(gender_n, gender);
    double height_n_prob = rangeProb(height_n, height, heightInt);
    double weight_n_prob = rangeProb(weight_n, weight, weightInt);
    double bodyfat_n_prob = rangeProb(bodyfat_n, bodyfat, fatInt);
    double diastolic_n_prob = rangeProb(diastolic_n, diastolic, diaInt);
    double systolic_n_prob = rangeProb(systolic_n, systolic, sysInt);
    double grip_force_n_prob = rangeProb(grip_force_n, grip_force, gripInt);
    double sit_and_bend_forward_n_prob = rangeProb(sit_and_bend_forward_n, sit_and_bend_forward, sitbendInt);
    double sit_up_count_n_prob = rangeProb(sit_up_count_n, sit_up_count, situpInt);
    double broad_jump_n_prob = rangeProb(broad_jump_n, broad_jump, broadInt);

    //weights for each value when calculating grade probability
    //calculated by setting each = tester and finding best accuracy
    double num1 = 0;
    double num2 = 0.7;
    double num3 = 0.08;
    double num4 = 1.73;
    double num5 = 0.98;
    double num6 = 0.31;
    double num7 = 1.09;
    double num8 = 1.12;
    double num9 = 1;
    double num10 = 1;
    double num11 = .93;

    double num12 = 1.6;
    double num13 = .76;
    double num14 = .23;
    double num15 = 0;
    double num16 = 1.02;
    double num17 = .94;
    double num18 = .97;
    double num19 = 0.96;
    double num20 = 1;
    double num21 = 1;
    double num22 = 1.05;

    //calculates probability of grade = 1 with weights calculated from repeated testing
    double prob_y = num1*log2(age_y_prob) + num2*log2(gender_y_prob) + num3*log2(height_y_prob) + num4*log2(weight_y_prob) + num5*log2(bodyfat_y_prob) + num6*log2(diastolic_y_prob) + num7*log2(systolic_y_prob) + num8*log2(grip_force_y_prob) + num9*log2(sit_and_bend_forward_y_prob) + num10*log2(sit_up_count_y_prob) + num11*log2(broad_jump_y_prob);
    //calculates probability of grade = 0 with weights calculated from repeated testing
    double prob_n = num12*log2(age_n_prob) + num13*log2(gender_n_prob) + num14*log2(height_n_prob) + num15*log2(weight_n_prob) + num16*log2(bodyfat_n_prob) + num17*log2(diastolic_n_prob) + num18*log2(systolic_n_prob) + num19*log2(grip_force_n_prob) + num20*log2(sit_and_bend_forward_n_prob) + num21*log2(sit_up_count_n_prob) + num22*log2(broad_jump_n_prob);
    
    int guess = 0;

    
    if(prob_y > prob_n){
        guess = 1;
    }

    if(guess == grade){
        correctGuesses = correctGuesses+1;
    }
    totalGuessed = totalGuessed+1;

    return guess;
}

double NaiveBayesClassifier::getAccuracy(){

    return correctGuesses/totalGuessed*100.0;
}

void NaiveBayesClassifier::setTester(double num){
    tester = num;
}

double NaiveBayesClassifier::getTester(){
    return tester;
}

void NaiveBayesClassifier::resetAccuracy(){
    correctGuesses = 0.0;
    totalGuessed = 0.0;
}

int main(int argc,char* argv[])
{

    //auto start = chrono::steady_clock::now();

    NaiveBayesClassifier NBC;

    ifstream inFile1;
    inFile1.open( argv[1] );

    string line;
    
    string num1;
    string gender;
    string num2; 
    string num3; 
    string num4; 
    string num5;
    string num6;
    string num7; 
    string num8; 
    string num9;
    string num10;
    string num11;

    int grade_0_flag = 1;
    int grade_1_flag = 1;
    
    //read in lines from training file
    while ( getline(inFile1,line) )
    {
        istringstream linestream(line);

        getline(linestream, num1, ',');
        getline(linestream, gender, ',');
        getline(linestream, num2, ',');
        getline(linestream, num3, ',');
        getline(linestream, num4, ',');
        getline(linestream, num5, ',');
        getline(linestream, num6, ',');
        getline(linestream, num7, ',');
        getline(linestream, num8, ',');
        getline(linestream, num9, ',');
        getline(linestream, num10, ',');
        getline(linestream, num11, ',');

        double age = stod(num1);
        double height = stod(num2);
        double weight = stod(num3);
        double bodyfat = stod(num4);
        double diastolic = stod(num5);
        double systolic = stod(num6);
        double grip_force= stod(num7);
        double sit_and_bend_forward = stod(num8);
        double sit_up_count = stod(num9);
        double broad_jump = stod(num10);
        int grade = stoi(num11);

        
        //inititializes CDF for when athlete grade = 0 and 1
        if(grade == 0 && grade_0_flag == 1){
            grade_0_flag = 0;
            NBC.setFeatureCDF(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
        }
        else if(grade == 1 && grade_1_flag == 1){
            grade_1_flag = 0;
            NBC.setFeatureCDF(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
        }
        else{//updates CDF after already created
            NBC.updateFeatureCDF(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
        }
        
    }
    inFile1.close();

    //auto end = chrono::steady_clock::now();

    /* cout << "Elapsed time for training in milliseconds: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms\n"; */

    //start = chrono::steady_clock::now();

    ifstream inFile2;
    inFile2.open( argv[2] );


    //change to true if testing
    bool isTest = false;
    if(!isTest){
        //reads every line of testing file
        while ( getline(inFile2,line) )
        {
            istringstream linestream(line);

            getline(linestream, num1, ',');
            getline(linestream, gender, ',');
            getline(linestream, num2, ',');
            getline(linestream, num3, ',');
            getline(linestream, num4, ',');
            getline(linestream, num5, ',');
            getline(linestream, num6, ',');
            getline(linestream, num7, ',');
            getline(linestream, num8, ',');
            getline(linestream, num9, ',');
            getline(linestream, num10, ',');
            getline(linestream, num11, ',');

            double age = stod(num1);
            double height = stod(num2);
            double weight = stod(num3);
            double bodyfat = stod(num4);
            double diastolic = stod(num5);
            double systolic = stod(num6);
            double grip_force= stod(num7);
            double sit_and_bend_forward = stod(num8);
            double sit_up_count = stod(num9);
            double broad_jump = stod(num10);
            int grade = stoi(num11);
            
            int guess = NBC.makeGuess(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
            cout<<guess<<"\n";
        
        }
        inFile2.close();
    }
    //runs test repeatedly to test how each weight affects the accuracy and finds the best weights
    //must be run multiple times with tester variable inputed into a single weight to test the best weight
    else{
        double acc = 0.0;
        double testerNum = 0.00;

        for(double i = 0.0; i <= 10.00; i=i+.01){

            //sets tester value for weight
            NBC.setTester(i);
        
            ifstream inFile2;
            inFile2.open( argv[2] );

            while ( getline(inFile2,line) )
            {
                istringstream linestream(line);

                getline(linestream, num1, ',');
                getline(linestream, gender, ',');
                getline(linestream, num2, ',');
                getline(linestream, num3, ',');
                getline(linestream, num4, ',');
                getline(linestream, num5, ',');
                getline(linestream, num6, ',');
                getline(linestream, num7, ',');
                getline(linestream, num8, ',');
                getline(linestream, num9, ',');
                getline(linestream, num10, ',');
                getline(linestream, num11, ',');

                double age = stod(num1);
                double height = stod(num2);
                double weight = stod(num3);
                double bodyfat = stod(num4);
                double diastolic = stod(num5);
                double systolic = stod(num6);
                double grip_force= stod(num7);
                double sit_and_bend_forward = stod(num8);
                double sit_up_count = stod(num9);
                double broad_jump = stod(num10);
                int grade = stoi(num11);

                int guess = NBC.makeGuess(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
        
            }
            inFile2.close();

            //cout<<NBC.getTester()<<" "<<NBC.getAccuracy()<<"\n";

            if(NBC.getAccuracy() > acc){
                testerNum = i;
                acc = NBC.getAccuracy();
            }

            NBC.resetAccuracy();

        }
        cout<<"Tester: "<<testerNum<<" Accuracy: "<<acc<<"\n";
        
    }

    //end = chrono::steady_clock::now();

    /* cout << "Elapsed time for testing in milliseconds: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms\n";

    cout<<"Accuracy: "<<NBC.getAccuracy()<<"\n"; */

    return 0;
}

