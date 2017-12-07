#ifndef GLOBALS_H
#define GLOBALS_H

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>

#include <boost/math/constants/constants.hpp>

#include <armadillo>

// Definition of datatypes

typedef unsigned int uint;
typedef std::complex<double> cpx;
typedef arma::Mat<cpx> Mat;
typedef arma::SpMat<cpx> SpMat;
typedef arma::Col<cpx> Vec;
typedef arma::Row<cpx> RowVec;
typedef arma::Col<double> VecReal;
typedef arma::Row<double> RowVecReal;
typedef arma::Row<uint> uRowVecInt;
typedef arma::Col<int> VecInt;
typedef arma::Col<uint> uVecInt;
typedef arma::Cube<cpx> Cube;
typedef arma::Cube<double> CubeReal;
typedef arma::Mat<double> MatReal;
//typedef arma::umat uMatInt;
typedef arma::Mat<uint> uMatInt;
typedef arma::imat MatInt;

static constexpr uint32_t MatrixSizeFactor = 2;
// This factor specifies the number of irreducible non-spatial dimensions.
// We have 2 flavors and 2 spins. Hence, matrices (e.g. Green's function) typically have size
// 4*Nx4*N. However, symmetry does identify two 2*Nx2*N subspaces which are equivalent up to complex 
// conjugation. Hence, the number of irreducible non-spatial dimensions is 2.

//typedef typename arma::Mat<cpx>::template fixed<MatrixSizeFactor,MatrixSizeFactor> MatSmall;
typedef Mat::fixed<MatrixSizeFactor,MatrixSizeFactor> MatSmall;
typedef MatReal::fixed<MatrixSizeFactor,MatrixSizeFactor> MatRealSmall;

// Constant helper matrices (complex)
static const MatSmall EyeSmall = arma::eye<Mat>(MatrixSizeFactor,MatrixSizeFactor);
static const MatSmall OneSmall = arma::ones<Mat>(MatrixSizeFactor,MatrixSizeFactor);
static const MatSmall ZeroSmall = arma::ones<Mat>(MatrixSizeFactor,MatrixSizeFactor);

// Global constans
namespace constants {
    const double kB = 1.0;           // Boltzmann constant
    const double c = 1.0;            // speed of light
    const double hbar = 1.0;         // reduced Planck constant
    const double pi =  boost::math::double_constants::pi;
};

#endif