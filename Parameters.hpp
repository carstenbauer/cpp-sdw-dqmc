#ifndef PARAMS_H
#define PARAMS_H

#include "globals.hpp"
#include "generic.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include <boost/lexical_cast.hpp>

struct SimulationParameters {
    enum ProposeMethod {
        BOXPROPOSE,
        CIRCLEPROPOSE
    };

    // Simulation parameters
    double T = 0.1;           // temperature
    int L = 2;              // linear dimension of square lattice, e.g. L=2 -> 2x2 lattice
    
    double dtau = 0.01;      // imaginary time step size
    int s = 10;              // Maximal number of timesteps until GF is calculated from scratch
                            // instead of iteratively (error accumulation) (Assaad tau1 = s*dtau)
                            // Also specifies that direct chain products of B matrices may only have length s-1 
                            // Negative value of s means "never use udv" in chain product

    int thermalSweeps = 500; // # # of thermalization sweeps (choose an even number! up down sweep)
    int sweeps = 500;        // # # of sweeps after thermalization (choose an even number! up down sweep)

    int saveRate = 20;      // Save state temporarily to disk after every 'saveRate' DOWNsweep.
    int printRate = 50;      // Print notice every 'printRate' sweeps.
    bool keepTempStates = false; // Store temporary state to separate files and keep them

    int currsweep = 1;      // current sweep, i.e. running from 1 to totalSweeps
                            // (inclusive, that is final state will be currsweep=totalSweeps+1)
    
    bool initRandomPhiField = true; // Otherwise we initialize phi field with zeros.

    double initMaxPhiLength = 1; // max length (norm) of order parameter vector at a site during initialization

    ProposeMethod proposeMethod = BOXPROPOSE;   //For local updates, new local op is randomly chosen within a
                                                // circle or box around old op vector
    double DeltaPhiLength = .1;  // Specifies the length (radius) of the quadratic box (circle)

    bool noFermions = false; // When set to true, we simulate a purely bosonic theory

    // Adaption Mode
    int adaptRounds = 50;           // number of times DeltaPhiLength is modified to achieve targetAcceptRate
                                    // The adaption is part of thermalSweeps!
                                    // Hence, we must have/enforce adaptRounds*sweepsPerAdaptRound<=thermalSweeps

    int sweepsPerAdaptRound = 10;   // specifies over how many sweeps lastAcceptRate is averaged
                                    // Choose even number to incorporate full set of up down sweeps.

    double targetAcceptRate = .5;   // DeltaPhiLength, that is the box or circle size is being
                                    // adaptively adjusted to achieve an acceptance rate of
                                    // local updates of targetAcceptRate (e.g. 50%=0.5)

    double adaptPercentage = 0.1;   // Increase or Decrease DeltaPhiLength by adaptPercentage percent
                                    // to reach targetAcceptRate

    double currAcceptRateSum = 0.0;

    // NEW Measurements
    int measureRate = 20; // Measure every measureRate sweep during Equilibrium (Measurements) stage


    

    // initialized via constructor
    double beta;
    std::vector<double> tausteps;
    int N;                  //Total number of sites
    int totalSweeps;        // thermalSweeps + sweeps
    int m;         // Defined by beta = m*dtau. Number of timesteps (in the integral discretization sense)
    int n;         // Total number of timesteps where GF is calculated from scratch = ceil(m/s)
    int dn_tau;    // GF is calculated from scratch every dn_tau timestep l ("commensurable version of parameter s")
    int M;         // number of measurements of observables (data points)

    SimulationParameters(){
        this->init();
    };

    void init() {
        assert(T>0);
        assert(L>0);
        assert(adaptRounds>=0);
        if(adaptRounds>0)
            assert(sweepsPerAdaptRound>0);
        assert(targetAcceptRate>=0.0&&targetAcceptRate<=1.0);

        this->beta = constants::kB/T; // calculate beta from T 
        this->tausteps = generic::arange(0,beta,dtau); // tau goes from 0 to beta (inclusive)

        this->N = this->L*this->L;
        this->totalSweeps = this->thermalSweeps + this->sweeps;
        this->m = int(tausteps.size()-1); // = beta/dtau
        this->n = this->s>0?std::ceil(m/s):0;    
        this->dn_tau = this->n==0?0:m/n;

        // Decrease adaptRounds, in case we exceed # of thermalSweeps
        while(adaptRounds*sweepsPerAdaptRound>thermalSweeps){
            --adaptRounds;
            generic::print("Decreasing simParams.adaptRounds to achieve desired # of thermalSweeps...");
        }

    }

    int n_tau2l(int n_tau){
    // Discrete index n_tau specifies a timeslice where GF is calculated from scratch but DOES NOT
    // correspond to the timeslice index l of that particular timeslice! This function translates
    // the given enumeration index n_tau into timeslice index l.
        return n_tau*dn_tau;
    };

    std::string toString(){
        using generic::num2str;

        int decimalPrecision = 20;
        std::string s = "SimulationParameters(";
        s += num2str(this->T,decimalPrecision)+",";
        s += num2str(this->L)+",";
        s += num2str(this->dtau,decimalPrecision)+",";
        s += num2str(this->s)+",";
        s += num2str(this->thermalSweeps)+",";
        s += num2str(this->sweeps)+",";
        s += num2str(this->currsweep)+",";
        s += num2str(this->saveRate)+",";
        s += num2str(this->printRate)+",";
        s += num2str(this->keepTempStates)+",";
        s += num2str(this->DeltaPhiLength,decimalPrecision)+",";
        s += num2str(this->adaptRounds)+",";
        s += num2str(this->sweepsPerAdaptRound)+",";
        s += num2str(this->adaptPercentage)+",";
        s += num2str(this->currAcceptRateSum)+",";
        s += (proposeMethod==BOXPROPOSE?std::string("BOX"):std::string("CIRCLE"))+",";
        s += ")";
        return s;
    };
    void save(std::ostream &strm) {
        strm << this->toString();
    };

};

struct HoppingParameters {
    double xh = 1;
    double xv = 0.5;
    double yh = 0.5;
    double yv = 1;
};
struct ModelParameters {
    // Model parameters
    double r = 8;         // tuning parameter; i.e. "mass" of bosonic field
    double lambda = 3;      // Yukawa coupling, i.e. boson-fermion-fermion three-point vertex
    double u = 1;           // coupling of quartic bosonic interaction
    double mu = 0.5;      // chemical potential
    HoppingParameters t;        // hoppings


    std::string toString(){
        using generic::num2str;

        std::string s = "ModelParameters(";
        s += num2str(this->r)+",";
        s += num2str(this->lambda)+",";
        s += num2str(this->u)+",";
        s += num2str(this->mu)+",";
        s += num2str(this->t.xh)+",";
        s += num2str(this->t.xv)+",";
        s += num2str(this->t.yh)+",";
        s += num2str(this->t.yv)+",";
        s += ")";
        return s;
    };

    void save(std::ostream &strm) {
        strm << this->toString();
    };
};

// Define stream output for structs above
std::ostream& operator<<(std::ostream &strm, ModelParameters &obj);

std::ostream& operator<<(std::ostream &strm, SimulationParameters &obj);

#endif
