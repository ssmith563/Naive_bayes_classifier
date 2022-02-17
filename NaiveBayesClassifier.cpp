#include <iostream>
#include <string>
#include <vector>
#include "NaiveBayesClassifier.h"
#include <fstream>
#include <sstream>

using namespace std;

NaiveBayesClassifier::NaiveBayesClassifier(){
    
}

int main(int argc,char* argv[])//int argc,char* argv[]
{
    ifstream inFile1;
    inFile1.open( argv[1] );

    string line;
    
    string age;
    string gender;
    string height; 
    string weight; 
    string bodyfat; 
    string diastolic;
    string systolic;
    string grip_force; 
    string sit_and_bend_forward; 
    string sit_up_count;
    string broad_jump;
    string grade;

    
    while ( getline(inFile1,line) )
    {
        istringstream linestream(line);
        //while(linestream.good()){
        getline(linestream, age, ',');
        getline(linestream, gender, ',');
        getline(linestream, height, ',');
        getline(linestream, weight, ',');
        getline(linestream, bodyfat, ',');
        getline(linestream, diastolic, ',');
        getline(linestream, systolic, ',');
        getline(linestream, grip_force, ',');
        getline(linestream, sit_and_bend_forward, ',');
        getline(linestream, sit_up_count, ',');
        getline(linestream, broad_jump, ',');
        getline(linestream, grade, ',');

        double number1 = stod(age);
        double number2 = stod(height);
        double number3 = stod(weight);
        double number4 = stod(bodyfat);
        double number5 = stod(diastolic);
        double number6 = stod(systolic);
        double number7= stod(grip_force);
        double number8 = stod(sit_and_bend_forward);
        double number9 = stod(sit_up_count);
        double number10 = stod(broad_jump);
        int number11 = stoi(grade);
        
        cout<<number2<<" ";
        
    }
    inFile1.close();

    
    
    return 0;
}
