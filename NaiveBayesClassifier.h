#ifndef NAIVEBAYESCLASSIFIER_H
#define NAIVEBAYESCLASSIFIER_H

#include <string>
#include <vector>


struct features{
    double mean=0;
    double oldMean=0;
    double n=1;
    double stdVar=0;
};

struct genderFeature{
    double n=1;
    double males=1;
};

class NaiveBayesClassifier
{
    
    private:

    features age_y;
    features age_n;

    genderFeature gender_y;
    genderFeature gender_n;

    features height_y;
    features height_n;

    features weight_y;
    features weight_n;

    features bodyfat_y;
    features bodyfat_n;

    features diastolic_y;
    features diastolic_n;

    features systolic_y;
    features systolic_n;

    features grip_force_y;
    features grip_force_n;

    features sit_and_bend_forward_y;
    features sit_and_bend_forward_n;

    features sit_up_count_y;
    features sit_up_count_n;

    features broad_jump_y;
    features broad_jump_n;

    public:

    NaiveBayesClassifier();

    void updateFeatureCDF(double age, std::string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade);

    void setFeatureCDF(double age, std::string gender, double height, 
    double weight, double bodyfat, double diastolic, double systolic, 
    double grip_force, double sit_and_bend_forward, double sit_up_count, 
    double broad_jump, int grade);

    void updateFeatureUtil(features *feature, double x);

    void updateGenderUtil(genderFeature *gender, std::string genderValue);

    void printFeatureAttributes(features feature);

    features getFeature();

    
};

#endif

