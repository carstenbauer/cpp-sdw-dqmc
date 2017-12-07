#ifndef GENERIC_CPP
#define GENERIC_CPP

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <assert.h>

#include <armadillo>

#include "generic.hpp"

// -------- Implementation --------

    std::vector<double> generic::arange(double start, double stop, double step) {
        std::vector<double> array;
        while(start <= (stop+step/2.0) ) {  // step/2.0 to ensure, that last element (stop) is included
            array.push_back(start);
            start += step;         // could recode to better handle rounding errors
        }
        return array;
    };

    std::vector<double> generic::linspace(double start, double stop, int steps) {
        std::vector<double> array;
        double step = (stop-start) / (steps-1);

        while(start <= stop) {
            array.push_back(start);
            start += step;           // could recode to better handle rounding errors
        }
        return array;
    };


    std::string generic::num2str(std::complex<double> d) {
        std::ostringstream oss;
        //oss << std::fixed << std::setprecision(4);
        //    oss.setf(std::ostringstream::showpoint);
        oss << d;
        return oss.str();
    };
    std::string generic::num2str(double d) {
        std::ostringstream oss;
        //oss << std::fixed << std::setprecision(4);
        //    oss.setf(std::ostringstream::showpoint);
        oss << d;
        return oss.str();
    };
    std::string generic::num2str(double d, int precision) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision);
        //    oss.setf(std::ostringstream::showpoint);
        oss << d;
        return oss.str();
    };
    std::string generic::num2str(int d) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0);
        //    oss.setf(std::ostringstream::showpoint);
        oss << d;
        return oss.str();
    };
    std::string generic::num2str(uint d) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0);
        //    oss.setf(std::ostringstream::showpoint);
        oss << d;
        return oss.str();
    };
    double generic::str2double(std::string s) {
        std::istringstream i(s);
        double x;
        if (!(i >> x))
        return 0;
        return x;
    };
    int generic::str2int(std::string s) {
        std::istringstream i(s);
        int x;
        if (!(i >> x))
        return 0;
        return x;
    };

    arma::Row<double> generic::polar2cartesian(arma::Row<double> polarvec){
        return arma::Row<double>({polarvec(0)*cos(polarvec(1)),polarvec(0)*sin(polarvec(1))});
    };

    arma::cx_mat generic::blockmat(arma::cx_cube& matrices){
    // Constructs a block matrix out of x^2 equally sized square matrices
    // Short: "blockmat" = {A,B;C,D} with A,B,C,D matrices (here x = 2)

        int blockdim = matrices.slice(0).n_cols;
        int nBlocks = matrices.n_slices;
        assert(std::floor(sqrt(nBlocks)) == sqrt(nBlocks));
        int blockmatdim = sqrt(nBlocks);
        arma::cx_mat blockmat = arma::zeros<arma::cx_mat>(blockmatdim*blockdim,blockmatdim*blockdim);

        int next = 0;
        for (int row=0; row<blockmatdim; ++row) {
            for (int col=0; col<blockmatdim; ++col) {
                blockmat.submat(row*blockdim,col*blockdim,(row+1)*blockdim-1,(col+1)*blockdim-1) = matrices.slice(next) ;
                ++next;
            }   
        }

        return blockmat;
    };

    arma::subview<std::complex<double>> generic::block(arma::cx_mat& mat, int MatrixSizeFactor, int row, int col){
    // Helper function to access spin and flavor 4x4 subblocks of MatrixSizeFactor*N x MatrixSizeFactor*N matrix
        int N = mat.n_cols/MatrixSizeFactor;
        return mat.submat(row*N,col*N,(row+1)*N-1,(col+1)*N-1) ;
    };    

    arma::SpSubview<std::complex<double>> generic::block(arma::sp_cx_mat& mat, int MatrixSizeFactor, int row, int col){
    // Helper function to access spin and flavor 4x4 subblocks of MatrixSizeFactor*N x MatrixSizeFactor*N matrix
        int N = mat.n_cols/MatrixSizeFactor;
        return mat.submat(row*N,col*N,(row+1)*N-1,(col+1)*N-1) ;
    };  

    arma::cx_mat generic::blockdiagmat(arma::cx_cube& matrices){
    // Analogon to arma::diagmat but for constructing a block diagonal matrix out of cube of matrices
    // Each slice of the cube is one block matrix.
    // Supports only equally sized blocks and square matrices

        assert(matrices.slice(0).n_cols == matrices.slice(0).n_rows); // assert square matrices
        int blockdim = matrices.slice(0).n_cols;
        int nBlocks = matrices.n_slices;
        arma::cx_mat blockmat = arma::zeros<arma::cx_mat>(nBlocks*blockdim,nBlocks*blockdim);

        // Insert blocks on diagonal
        for (int i = 0; i<nBlocks; ++i)
            blockmat.submat(i*blockdim,i*blockdim,(i+1)*blockdim-1,(i+1)*blockdim-1) = matrices.slice(i);

        return blockmat;
    };


    bool generic::fileExists(std::string filename){
    // Checks and returns true/false wether a file with the given filename exists
        std::ifstream tfile(filename);
        bool x = tfile.is_open();
        if (x) tfile.close();
        return x;
    };

    double generic::relDiff(double a, double b){
        using std::abs;
        return 2.0*abs(a-b)/abs(a+b);
    };
    double generic::relDiff(cpx a, cpx b){
        using std::abs;
        return 2.0*abs(a-b)/abs(a+b);
    };
    MatReal generic::relDiff(Mat a, Mat b){
        using arma::abs;
        return 2.0*abs(a-b)/abs(a+b);
    };
    MatReal generic::relDiff(MatReal a, MatReal b){
        using arma::abs;
        return 2.0*abs(a-b)/abs(a+b);
    };

    void generic::replace(std::string& str,
               const std::string& oldStr,
               const std::string& newStr)
    {
        std::string::size_type pos = 0u;
        while((pos = str.find(oldStr, pos)) != std::string::npos){
            str.replace(pos, oldStr.length(), newStr);
            pos += newStr.length();
        }
    };


 #endif