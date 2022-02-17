#include <iostream>
#include <string>
#include <vector>
#include "NaiveBayesClassifier.h"
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

NaiveBayesClassifier::NaiveBayesClassifier(){
    
}

void NaiveBayesClassifier::updateFeatureCDF(double age, string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade){

    

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

void NaiveBayesClassifier::setFeatureCDF(double age, string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade){

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

void NaiveBayesClassifier::updateFeatureUtil(features *feature, double x){
    
    feature->n = feature->n+1;
    
    feature->oldMean = feature->mean;

    feature->mean = ((feature->mean * (feature->n-1))+x)/feature->n;

    feature->stdVar = sqrt( ( (feature->n-2)*feature->stdVar*feature->stdVar + (x - feature->mean)*(x - feature->oldMean) )/(feature->n-1));
}

void NaiveBayesClassifier::updateGenderUtil(genderFeature *gender, string genderValue){
    gender->n = gender->n+1.0;
    if(genderValue == "M"){
        gender->males = gender->males+1.0;
    }
}

void NaiveBayesClassifier::printFeatureAttributes(features feature){
    cout<<"Mean: "<<feature.mean<<"\n";
    cout<<"N: "<<feature.n<<"\n";
    cout<<"Standard Deviation: "<<feature.stdVar<<"\n";
}

features NaiveBayesClassifier::getFeature(){
    return height_n;
}

double NaiveBayesClassifier::zScoreCalc(features feature, double x){
    double zscore = (x - feature.mean)/feature.stdVar;

    return 0.5 * erfc(-zscore * M_SQRT1_2);

}

double NaiveBayesClassifier::rangeProb(features feature, double x){
    //can change range here to test accuracy
    return zScoreCalc(feature, x + feature.stdVar/2) - zScoreCalc(feature, x - feature.stdVar/2);
}

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
    
    double age_y_prob = rangeProb(age_y, age);
    //cout<<"age: "<<age<<" age 1 prob: "<<age_y_prob<<" ";
    double gender_y_prob = genderProb(gender_y, gender);
    //cout<<"Gender: "<<gender<<" Gender 1 prob: "<<gender_y_prob<<" ";
    double height_y_prob = rangeProb(height_y, height);
    //cout<<height_y_prob<<" ";
    double weight_y_prob = rangeProb(weight_y, weight);
    //cout<<weight_y_prob<<" ";
    double bodyfat_y_prob = rangeProb(bodyfat_y, bodyfat);
    //cout<<bodyfat_y_prob<<" ";
    double diastolic_y_prob = rangeProb(diastolic_y, diastolic);
    //cout<<diastolic_y_prob<<" ";
    double systolic_y_prob = rangeProb(systolic_y, systolic);
    //cout<<systolic_y_prob<<" ";
    double grip_force_y_prob = rangeProb(grip_force_y, grip_force);
    //cout<<grip_force_y_prob<<" ";
    double sit_and_bend_forward_y_prob = rangeProb(sit_and_bend_forward_y, sit_and_bend_forward);
    //cout<<sit_and_bend_forward_y_prob<<" ";
    double sit_up_count_y_prob = rangeProb(sit_up_count_y, sit_up_count);
    //cout<<sit_up_count_y_prob<<" ";
    double broad_jump_y_prob = rangeProb(broad_jump_y, broad_jump);
    //cout<<broad_jump_y_prob<<" ";

    double prob_y = log2(age_y_prob) + log2(gender_y_prob) + log2(height_y_prob) + log2(weight_y_prob) + log2(bodyfat_y_prob) + log2(diastolic_y_prob) + log2(systolic_y_prob) + log2(grip_force_y_prob) + log2(sit_and_bend_forward_y_prob) + log2(sit_up_count_y_prob) + log2(broad_jump_y_prob);
    
    double age_n_prob = rangeProb(age_n, age);
    //cout<<"age: "<<age<<" age 1 prob: "<<age_n_prob<<" ";
    double gender_n_prob = genderProb(gender_n, gender);
    //cout<<"Gender: "<<gender<<" Gender 0 prob: "<<gender_n_prob<<" ";
    double height_n_prob = rangeProb(height_n, height);
    //cout<<height_n_prob<<" ";
    double weight_n_prob = rangeProb(weight_n, weight);
    //cout<<weight_n_prob<<" ";
    double bodyfat_n_prob = rangeProb(bodyfat_n, bodyfat);
    //cout<<bodyfat_n_prob<<" ";
    double diastolic_n_prob = rangeProb(diastolic_n, diastolic);
    //cout<<diastolic_n_prob<<" ";
    double systolic_n_prob = rangeProb(systolic_n, systolic);
    //cout<<systolic_n_prob<<" ";
    double grip_force_n_prob = rangeProb(grip_force_n, grip_force);
    //cout<<grip_force_n_prob<<" ";
    double sit_and_bend_forward_n_prob = rangeProb(sit_and_bend_forward_n, sit_and_bend_forward);
    //cout<<sit_and_bend_forward_n_prob<<" ";
    double sit_up_count_n_prob = rangeProb(sit_up_count_n, sit_up_count);
    //cout<<sit_up_count_n_prob<<" ";
    double broad_jump_n_prob = rangeProb(broad_jump_n, broad_jump);
    //cout<<broad_jump_n_prob<<" ";

    //cout<<"\n";
    double prob_n = log2(age_n_prob) + log2(gender_n_prob) + log2(height_n_prob) + log2(weight_n_prob) + log2(bodyfat_n_prob) + log2(diastolic_n_prob) + log2(systolic_n_prob) + log2(grip_force_n_prob) + log2(sit_and_bend_forward_n_prob) + log2(sit_up_count_n_prob) + log2(broad_jump_n_prob);

    int guess = 0;

    /* cout<<" age 1 prob: "<<height_y_prob<<" ";
    cout<<"\n";
    cout<<" age 2 prob: "<<height_n_prob<<" ";
    cout<<"\n\n"; */
    //cout<<prob_y<<" "<<prob_n<<"\n";
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
    //cout<<gender_y.males<<" "<<gender_y.n<<"\n"<<gender_n.males<<" "<<gender_n.n<<"\n";
    return correctGuesses/totalGuessed;
}

int main(int argc,char* argv[])//int argc,char* argv[]
{

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

        

        if(grade == 0 && grade_0_flag == 1){
            grade_0_flag = 0;
            NBC.setFeatureCDF(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);

        }
        else if(grade == 1 && grade_1_flag == 1){
            grade_1_flag = 0;
            NBC.setFeatureCDF(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
        }
        else{
            NBC.updateFeatureCDF(age, gender, height, weight, bodyfat, diastolic, systolic, grip_force, sit_and_bend_forward, sit_up_count, broad_jump, grade);
        }
        
    }
    inFile1.close();


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
        //cout<<guess<<"\n";
        
    }
    inFile2.close();

    /* features test = NBC.getFeature();
    NBC.printFeatureAttributes(test);

    cout<<"z score: "<<((166.0 - test.mean)/test.stdVar)<<"\n";
    cout<<NBC.zScoreCalc(test, 166.0)<<"\n"; */

    cout<<NBC.getAccuracy();
    
    return 0;
}
