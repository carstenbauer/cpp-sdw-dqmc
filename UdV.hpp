#ifndef UDV_H
#define UDV_H

#include "globals.hpp"
#include <iostream>
#include <armadillo>

//matrices used in the computation of B-matrices decomposed into
//(U,d,V) = (unitary matrix, real diagonal matrix elements >= 0, unitary matrix)
//We store U, d, V_t. V_t is the conjugate-transpose of V. The Lapack-Routines
//to compute svd(M) for a matrix M return U, d, V_t with M = U*d*V = U*d*(V_t)^(dagger).

struct UdV {
    Mat U;
    VecReal d;  // Eigenvalues of diagonal matrix D (singular values of original matrix)
    Mat V_t;			//conjugate-transpose of V
    //default constructor: leaves everything empty
    UdV() : U(), d(), V_t() {}
    //specify matrix size: initialize to identity
    UdV(int size) :
        U(arma::eye<Mat>(size,size)), d(arma::ones<VecReal>(size)), V_t(arma::eye<Mat>(size,size))
    { }

    void clear(int size);

    Mat product();

};

void UdVDecompose(const Mat& X, Mat& U, VecReal& d, Mat& V_t);

void UdVDecompose(const Mat& X, UdV& udv);

UdV UdVDecompose(const Mat& X);




#endif
