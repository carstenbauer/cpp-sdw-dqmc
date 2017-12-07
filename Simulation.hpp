#ifndef SIMULATION_H
#define SIMULATION_H

#include <fstream>
#include <iostream>
#include <exception>
#include <cmath>            // e.g. sin(), cos(), sqrt()
#include <algorithm>        // e.g. min(), max()
#include <tuple>
#include <assert.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/timer/timer.hpp>
// #include <alps/alea.h>
#include <armadillo>

#include "globals.hpp"
#include "UdV.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"

#define print_compare(a,b,reltol) print_compare(a,#a,b,#b,reltol)
#define print_compare_Mat_abs(a,b,abstol) print_compare_Mat_abs(a,#a,b,#b,abstol)
#define print_compare_UdV_abs(a,b,abstol) print_compare_UdV_abs(a,#a,b,#b,abstol)
#define print_var(x) print(std::string(#x)+": ",x)
#define print_var_prec(x,prec) print(std::string(#x)+": ",x,prec)

namespace MonteCarlo
{
    enum Stage // We devide the whole MC sweeping into two general classes: Thermalization and Equilibrium
    {
        OFFLINE,
        THERMALIZATION, // We are on our way to equilibrium
        EQUILIBRIUM    // We (hopfully) achieved equilibrium
    };

    enum SubStages
    {
        NONE,   // this means we are doing nothing special indep of wether we are
                // in Stage::THERMALIZATION, Stage::EQUILIBRIUM or Stage::OFFLINE
        // Thermal stage
        ADAPTION, 
        // Equilibrium stage
        MEASUREMENTS
    };

    enum SweepDirection { DOWNSWEEP=false, UPSWEEP=true }; // During a sweep, are we traversing the imaginary time dimension up (0->beta) or down (beta->0) 
}

class Simulation {
    private:

    public:

        // -------- Attributes --------

        bool tests = false; // Should Simulation::performTests() should be run?

        SimulationParameters simParams;
        ModelParameters modelParams;
        Logger logger;

        uMatInt TimeNeighborTable;
        uMatInt NeighborTable;
        Mat kineticExponential; // Should be MatReal if there is no B-field
        
        // 3D cube for storing bosonic order parameter
        // 1st dim (rows): lattice sites (linearized 2D lattice) 
        // 2nd dim (cols): order parameter components (x,y)
        // 3nd dim (slices): imaginary time (slice)
        CubeReal phiField;


        // Storing computed quantities of current configuration
        Mat g; // Green's function (2*Nx2*N) (Which timeslice? current in update? Or cube to store everything?)
        double S_phi; // Bosonic action

        // Storing computed quantities within local update
        MatSmall Delta_i;
        MatSmall M;
        MatSmall Minverse; 
        cpx Mdet;

        // Storing computed quantities within sweep
        std::vector<UdV> UdVStorage;
        MonteCarlo::SweepDirection sweepDirection;

        uint tempStateCounter;

        const size_t seed = 1123581321;
        boost::random::mt19937_64 rng;

        bool resume;

        MonteCarlo::Stage stage;
        MonteCarlo::SubStages substage;

        Mat Eye;
        Mat Zeros;
        Mat& Zero;
        Mat& One;
        Mat Ones;


        // Observables
        bool measuring = false; // Indicates wether we are currently in a "measurement sweep"
                                // (in contrast to in a thermal or waiting between measurements sweep)

        double occupationNumber;

        RowVecReal meanPhi;
        double normMeanPhi;
        double meanNormPhi;
        double SDWSuscept;

        std::vector<RowVecReal> meanPhiTimeSeries;
        std::vector<double> normMeanPhiTimeSeries;
        std::vector<double> meanNormPhiTimeSeries;
        std::vector<double> SDWSusceptTimeSeries;

        std::vector<int> times; // sweep numbers at which the measurements have been taken

        // -------- Methods --------

        // ## Initializion

            Simulation():One(Eye),Zero(Zeros),tempStateCounter(0),resume(false)
            ,stage(MonteCarlo::OFFLINE)
            ,substage(MonteCarlo::NONE)
            ,S_phi(0)
            ,sweepDirection(MonteCarlo::DOWNSWEEP)
            ,rng(boost::random::mt19937_64(this->seed))
            ,phiField(arma::zeros<CubeReal>(simParams.N,2,simParams.tausteps.size()))
            ,g(arma::zeros<Mat>(2*simParams.N,2*simParams.N)){
                // this->init();
            };

            void performTests();

            void commonInit();
            // Initialization indep. of wether the Simulations has been
            // initialized from scratch (Simulation::init()) or resumed (Simulation::initResume())

            void init();
            // Initializes a new simulation from scratch
            void initResume();
            // Initializes a new simulation from loaded settings
            void initGreensFunction();
            void initGreensFunctionEmpty();
            void initPhiField();

            void initRNG(); // Initializes the random number generator

        // ## Generic & Helper

            static double dataReal(const cpx& value) { return value.real(); } // could just use std::real
            static double dataReal(const double& value) { return value; }        //      -- " --
            static void setReal(cpx& value, double realPart) { value.real(realPart); }
            static void setReal(double& value, double realPart) { value = realPart; }
            static void setImag(cpx& value, double imagPart) { value.imag(imagPart); }
            static void setImag(double& value, double imagPart) { value = imagPart; }

            //need these to have the same code handle real/complex subviews
            //subviews do not have members set_real or set_imag.
            //For real subviews: just discard the imagPart.
            template<class Matrix1, class Matrix2>
            static void setRealImag(arma::subview<double> subv,
                                    const Matrix1& realPart, const Matrix2& imagPart) {
                (void) imagPart;
                subv = realPart;
            }
            template<class Matrix1, class Matrix2>
            static void setRealImag(arma::subview<cpx> subv,
                                    const Matrix1& realPart, const Matrix2& imagPart) {
                subv = MatCpx(realPart, imagPart);
            }

            RowVecReal getPhi(CubeReal& phiField, int site, int timeslice){
                assert(site >= 0);
                assert(site <  phiField.n_rows);
                assert(timeslice >= 0); // MAX has 1 here. Why?!
                assert(timeslice < phiField.n_slices);
                RowVecReal phivec(2);
                phivec(0) = phiField(site,0,timeslice);
                phivec(1) = phiField(site,1,timeslice);
                return phivec;
            };
            RowVecReal getPhi(int site, int timeslice){
                return getPhi(this->phiField,site,timeslice);
            };

            void setPhi(CubeReal& phiField, int site, int timeslice, RowVecReal phi){
                assert(site >= 0);
                assert(site <  phiField.n_rows);
                assert(timeslice >= 0); // MAX has 1 here. Why?!
                assert(timeslice < phiField.n_slices);
                phiField(site,0,timeslice) = phi(0);
                phiField(site,1,timeslice) = phi(1);
            };
            void setPhi(int site, int timeslice, RowVecReal phi){
                setPhi(this->phiField,site,timeslice,phi);
            };
            void setPhi(std::tuple<RowVecReal,int,int>& newlocalphi){
                setPhi(this->phiField,newlocalphi);
            };
            void setPhi(CubeReal& phiField, std::tuple<RowVecReal,int,int>& newlocalphi){
                RowVecReal newphi;
                int site;
                int timeslice;
                std::tie(newphi,site,timeslice) = newlocalphi;
                setPhi(phiField,site,timeslice,newphi);
            };

        // ## Monte Carlo simulation

            void run();
            // Main method, runs the Markov Chain Monte Carlo simulation

            // ## Sweeps
            double sweep(); // Perform full space-time lattice sweep, where each point of the order parameter field cube is visited once.
            double sweepNoUDVStorage();
            double performSpatialLatticeSweep(int timeslice); // Sweep over spatial lattice for given timeslice

            // ## Update proposals
            std::tuple<RowVecReal,int,int> proposeLocalUpdateAtSite(int site, int tau);
            std::tuple<RowVecReal,int,int> proposeLocalUpdate();

            // ## Calculations
                //## Bosonic sector
                double calculateBosonicAction(CubeReal& conf);
                double calculateBosonicAction();
                double calculateBosonicActionChange(std::tuple<RowVecReal,int,int>& newlocalphi);
            
                // ## Fermions (e.g. Green's function)
                std::tuple<double,double> calculateCoshSinhTerm(int site, int timeslice, CubeReal& phiField, bool sign);
                std::tuple<double,double> calculateCoshSinhTerm(RowVecReal phivec, bool sign);
                
                Mat calculatePotentialExponential(CubeReal& phiField, int timeslice, bool sign);
                Mat calculatePotentialExponential(int timeslice, bool sign);
                void calculateKineticExponential(); // Needs to be calculated once!               
                Mat calculateBmat(int l); // Calculate single B(l) matrix
                Mat calculateBmatNoUdV(int l2, int l1);
                void calculateBmat(int l2, int l1, UdV& udv_res, int udvRate);
                Mat calculateGFfromScratchNaive(Mat& g, int l);
                
                void calculateGFfromScratch(Mat& g, int l, int udvRate=1);
                void advanceGFusingStorage(Mat& g, int n_tau, MonteCarlo::SweepDirection sd);
                void greenFromUdV(Mat& green_out, const UdV& UdV_l, const UdV& UdV_r) const;
                void greenFromUdV(Mat& green_out, const UdV& UdV_r) const;                
                void updateGFIterativelyInSlice(std::tuple<RowVecReal,int,int>& newlocalphi);
                void wrapGFToNextTimeslice(Mat& g, int timeslice, MonteCarlo::SweepDirection sd);

                bool consistencyCheckGF(Mat& g_1, Mat& g_2, double tol, std::string error_msg); // TODO

                // Checkerboard decomposition
                Mat leftMultiplyKineticExponentialCheckerboard(Mat& A);

                // UdV Storage
                void calculateUdVofBmat(int n_tau);
                void setupUdVstorage();
                
                // ## Acceptance probability
                Mat calculateDeltaNaive(int l,CubeReal& newphiField);
                void calculateDelta_i(std::tuple<RowVecReal,int,int>& newlocalphi);
                void calculateM(std::tuple<RowVecReal,int,int>& newlocalphi);
                void calculateMInverseAndDet();

                cpx calculateDetRatioForLocalUpdateNaive(CubeReal& newphiField,int site, int timeslice);
                cpx calculateDetRatioForLocalUpdateMax(std::tuple<RowVecReal,int,int>& newlocalphi);
                cpx calculateDetRatioForLocalUpdate(std::tuple<RowVecReal,int,int>& newlocalphi);


            // ## Metropolis acceptance check
            bool acceptMetropolisLocal(std::tuple<RowVecReal,int,int>& newlocalphi);

            // ## Measurements
            void initMeasurements();
            void measure(int timeslice);
            void finishMeasurements();
            void measureMeanPhi(int timeslice);
            void measureMeanNormPhi(int timeslice);
            void measureSDWSuscept(int timeslice);
            void measureOccupationNumber();
            void finishMeanPhi();
            void finishOccupationNumber();

        // ## Lattice logic

            Cube hoppingCube();
            uVecInt siteIndex2Coords(int site);
            int coords2SiteIndex(uVecInt pos);
            int coords2SiteIndex(uint x, uint y);
            uVecInt calculateNeighbors(int site);
            uVecInt calculateTimeNeighbors(int timeslice);

            uVecInt getNeighbors(int site);
            uVecInt getTimeNeighbors(int timeslice);
            uVecInt getTopAndRightNeighbors(int site);
            void createNeighborTable();
            void createTimeNeighborTable();
            Mat calculateKineticExponentialCheckerboard(int sign, double factor);


        // ## Random number generation

            double randu(double min, double max);
            // Generates a random number drawn from uniform_real_distribution in range [min,max)
            double randu01();
            // Defines randu01(), which delivers a random number chosen from uniform_real_distribution in
            // range [0,1)
            int randint(int min, int max);
            // Generates a random number drawn from uniform_int_distribution in range [min,max]
            bool randbool(double prob_true);
            // Generates a boolean which is true with probability prob_true
            RowVecReal getRandom2DVectorPolar(double max_length);
            // Generates a random two-dimensional row vector, given a max length constraint.
            // The resulting vector has the form vector = {radius, angle}, i.e. polar coordinates.
            RowVecReal getRandom2DVector(double max_length);
            // Generates a random two-dimensional row vector, given a max length constraint.
            // The resulting vector has the form vector = {x, y}, i.e. cartesian coordinates.
            RowVecReal getRandom2DVectorBox(double box_length);
            // Generates a random two-dimensional row vector within a quadratic box of length box_length.
            // The resulting vector has the form vector = {x, y}, i.e. cartesian coordinates.


        // ## Storage

            class StorageException: public std::exception {
                private:
                    std::string message_;
                public:
                    //explicit StorageException(const std::string& message);
                    virtual const char* what() const throw() {
                        return message_.c_str();
                    }

                    StorageException(const std::string& message) : message_(message) {

                    }  
            };

            void writeConfigurationToFile(std::string filename);
            void writeStateToFile(std::string filename);
            void loadConfigurationFromFile(std::string filename);
            void loadStateFromFile(std::string filename);

            // Available w/o instantiation
            static void writeConfigurationToFile(CubeReal& phiField, std::string filename);
            static void writeStateToFile(Simulation& sim, std::string filename);
            static void loadConfigurationFromFile(std::string filename, CubeReal& phiField);
            static void loadStateFromFile(std::string filename, Simulation& sim);

            void timeSeriesToFile(std::vector<int> times, std::vector<RowVecReal> timeSeries, std::string filename);
            void timeSeriesToFile(std::vector<int> times, std::vector<double> timeSeries, std::string filename);

};


#endif
