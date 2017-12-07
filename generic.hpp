#ifndef GENERIC_H
#define GENERIC_H

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <assert.h>
#include "UdV.hpp"

#include <armadillo>

namespace generic {

// -------- Definitions --------

    std::vector<double> arange(double start, double stop, double step);

    std::vector<double> linspace(double start, double stop, int steps);


    std::string num2str(std::complex<double> d);
    std::string num2str(double d);
    std::string num2str(double d, int precision);
    std::string num2str(int d);
    std::string num2str(uint d);
    double str2double(std::string s);
    int str2int(std::string s);

    arma::Row<double> polar2cartesian(arma::Row<double> polarvec);

    arma::cx_mat blockmat(arma::cx_cube& matrices);
    arma::subview<std::complex<double>> block(arma::cx_mat& mat, int MatrixSizeFactor, int row, int col);
    arma::SpSubview<std::complex<double>> block(arma::sp_cx_mat& mat, int MatrixSizeFactor, int row, int col);
    arma::cx_mat blockdiagmat(arma::cx_cube& matrices);

    bool fileExists(std::string filename);

    inline void print(std::string text){
        std::cout << text << std::endl;
    };

    template<typename T>
    inline void print(std::string text, T obj){
        if (typeid(T) == typeid(double) || typeid(T) == typeid(int) || typeid(T) == typeid(cpx) || typeid(T) == typeid(arma::uword))
            std::cout << text << obj << std::endl;
        else
            std::cout << text << std::endl << obj << std::endl;
    };

    template<typename T>
    inline void print(std::string text, T obj, int precision){
        std::cout.precision(precision);
        if (typeid(T) == typeid(double) || typeid(T) == typeid(int) || typeid(T) == typeid(cpx))
            std::cout << text << std::fixed << obj << std::endl;
        else
            std::cout << text << std::endl << std::fixed << obj << std::endl;
    };


    // Defines what we consider a relative difference
    double relDiff(double a, double b);
    double relDiff(cpx a, cpx b);
    MatReal relDiff(Mat a, Mat b);
    MatReal relDiff(MatReal a, MatReal b);

    // Comparison of complex matrices in usual an UdV representation

    template<typename T>
    inline double compare_Mat_rel_max(T a, T b, double tolerance){
    // Compares two complex or real matrices relatively up to given tolerance,
    // i.e. every relative difference smaller than tolerance is considered as zero,
    // and returns the maximal relative difference of matrix elements.
        MatReal reldiff = relDiff(a,b);
        double max_reldiff = 0;
        reldiff.for_each([&](arma::mat::elem_type& val) { if(val>tolerance&&val>max_reldiff) max_reldiff = val; });
        return max_reldiff;
    };

    template<typename T>
    inline double compare_Mat_rel_mean(T a, T b, double tolerance){
    // Compares two complex or real matrices relatively up to given tolerance,
    // i.e. every relative difference smaller than tolerance is considered as zero,
    // and returns the mean relative difference of matrix elements.
        MatReal reldiff = relDiff(a, b);
        reldiff.for_each([&](arma::mat::elem_type& val) { if(val<tolerance) val = 0; });
        return arma::mean(arma::mean(reldiff));
    };

    template<typename T>
    inline double compare_Mat_abs_max(T a, T b, double tolerance){
    // Compares two complex or real matrices absolutely up to given tolerance,
    // i.e. every absolute difference smaller than tolerance is considered as zero,
    // and returns the maximal absolute difference of matrix elements.
        MatReal absdiff = arma::abs(a-b);
        double max_absdiff = 0;
        absdiff.for_each([&](arma::mat::elem_type& val) { if(val>tolerance&&val>max_absdiff) max_absdiff = val; });
        return max_absdiff;
    };

    template<typename T>
    inline double compare_Mat_abs_mean(T a, T b, double tolerance){
    // Compares two complex or real matrices absolutely up to given tolerance,
    // i.e. every absolute difference smaller than tolerance is considered as zero,
    // and returns the mean absolute difference of matrix elements.
        MatReal absdiff = arma::abs(a-b);
        absdiff.for_each([&](arma::mat::elem_type& val) { if(val<tolerance) val = 0; });
        return arma::mean(arma::mean(absdiff));
    };

    double compare_UdV_abs_max(UdV a, UdV b, double tolerance);
    double compare_UdV_abs_mean(UdV a, UdV b, double tolerance);
    double compare_UdV_rel_max(UdV a, UdV b, double tolerance);
    double compare_UdV_rel_mean(UdV a, UdV b, double tolerance);



    // Compare and print


    template<typename T>
    inline bool print_compare(T a, std::string aStr, T b, std::string bStr, double reltol){
        // TODO
        return false;
    };

    template<typename T>
    inline bool print_compare_Mat_rel(T a, std::string aStr, T b, std::string bStr, double reltol){
        // TODO
        return false;
    };

    template<typename T>
    inline bool print_compare_Mat_abs(T a, std::string aStr, T b, std::string bStr, double abstol){
        // TODO
        return false;
    };

    bool print_compare_UdV_abs(UdV a, std::string aStr, UdV b, std::string bStr, double abstol);
    bool print_compare_UdV_rel(UdV a, std::string aStr, UdV b, std::string bStr, double abstol);

    template<typename T>
    inline void print_Mat_info(T mat){
        std::cout << "Size: " << mat.n_rows << "x" << mat.n_cols << std::endl;
        std::cout << "Maximum [abs]: " << mat.max() << " [" << std::abs(mat.max()) << "]" << std::endl;
        std::cout << "Mean [abs]: " << arma::mean(arma::mean(mat)) << " [" << std::abs(arma::mean(arma::mean(mat))) << "]" << std::endl;
        std::cout << "Minimum [abs]: " << mat.min() << " [" << std::abs(mat.min()) << "]" << std::endl;
    };


    void replace(std::string& str, const std::string& oldStr, const std::string& newStr);

};

 #endif