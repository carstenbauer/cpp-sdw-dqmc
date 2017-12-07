#ifndef UDV_CPP
#define UDV_CPP

#include <iostream>
#include <armadillo>

#include "UdV.hpp"

void UdV::clear(int size){
    U = arma::eye<Mat>(size,size);
    d = arma::ones<VecReal>(size);
    V_t = arma::eye<Mat>(size,size);
};

Mat UdV::product(){
    return U*arma::diagmat(d)*V_t.t();
};

void UdVDecompose(const Mat& X, Mat& U, VecReal& d, Mat& V_t){
// Performs a UdV decomposition (aka singular value decomposition) of the Matrix X
// and stores the resulting unitary matrix U, the singular value vector d and the
// conjugate transpose of the unitary matrix V (which we call V_t) in the respective refrence arguments.
// The matrix X can be restored by the formula: X = U*diagmat(d)*V.t()
    bool success = arma::svd(U,d,V_t,X,"std"); // TODO: "dc" or "std"?
    if(!success)
        std::cout << "UdV decomposition failed!" << std::endl;
}

void UdVDecompose(const Mat& X, UdV& udv){
    UdVDecompose(X,udv.U,udv.d,udv.V_t);
}

UdV UdVDecompose(const Mat& X){
    UdV udv;
    UdVDecompose(X,udv.U,udv.d,udv.V_t);
    return udv;
}


#endif
