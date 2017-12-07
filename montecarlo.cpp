#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <armadillo>

#include <boost/chrono.hpp>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/array.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"

#include "globals.hpp"
#include "generic.hpp"
#include "Parameters.hpp"
#include "Logger.hpp"
#include "Simulation.hpp"

namespace po = boost::program_options;


bool handleCMDOptions(int ac, char* av[], Simulation& sim) {
    // ## Cmd options handling
    try {

        po::command_line_style::style_t style = po::command_line_style::style_t(
        po::command_line_style::unix_style | 
        po::command_line_style::case_insensitive | 
        po::command_line_style::allow_long_disguise );

        // Add options
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("linearSize", po::value<int>(), "set linear system size, i.e. linear dimension  of square lattice")
            ("thermalSweeps", po::value<int>(), "set # of thermal sweeps")
            ("sweeps", po::value<int>(), "set # of non-thermal sweeps, e.g. for measurements")
            ("printRate", po::value<int>(), "print notification every 'printRate' sweeps")
            ("saveRate", po::value<int>(), "save 'temp0.state' every 'saveRate' sweeps")
            ("temperature", po::value<double>(), "system temperature")
            ("dtau", po::value<double>(), "imaginary time step")
            ("r", po::value<double>(), "model parameter r, i.e. 'mass' of boson field")
            ("lambda", po::value<double>(), "model parameter lambda, i.e. 'Yukawa' coupling")
            ("u", po::value<double>(), "model parameter u, i.e. quartic coupling")
            ("mu", po::value<double>(), "chemical potential mu")
            ("s", po::value<int>(), "Maximal number of timesteps until GF is calculated from scratch")
            ("adaptRounds", po::value<int>(), "Number of times DeltaPhiLength is modified to achieve targetAcceptRate")
            ("targetAcceptRate", po::value<int>(), "Target accept rate")
            ("adaptPercentage", po::value<int>(), "Increase or Decrease DeltaPhiLength by adaptPercentage percent to reach targetAcceptRate")
            ("sweepsPerAdaptRound", po::value<int>(), "Specifies over how many sweeps lastAcceptRate is averaged. Choose even number to incorporate full set of up down sweeps.")
            ("keep-temp-states", "save system state to separate tempX.state files and keep them")
            ("tests", "Run Simulation::performTests() method")
            ("noFermions", "Simulate pure OP field")
            ("initZeroPhiField", "Uses zero OP field as initial configuration")
        ;

        // Parse options
        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc, style),  vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("keep-temp-states")) {
            sim.simParams.keepTempStates = true;
        }

        if (vm.count("tests")) {
            sim.tests = true;
        }

        if (vm.count("noFermions")) {
            std::cout << "(CMD) Running in 'noFermions' mode."<< ".\n";
            sim.simParams.noFermions = true;
        }

        if (vm.count("initZeroPhiField")) {
            sim.simParams.initRandomPhiField = false;
        }

        if (vm.count("linearSize")) {
            sim.simParams.L=vm["linearSize"].as<int>();
            std::cout << "(CMD) Linear system size was set to " << sim.simParams.L << ".\n";
        }
        if (vm.count("s")) {
            sim.simParams.s=vm["s"].as<int>();
            std::cout << "(CMD) Paramter s was set to " << sim.simParams.s << ".\n";
        }
        if (vm.count("sweepsPerAdaptRound")) {
            sim.simParams.sweepsPerAdaptRound=vm["sweepsPerAdaptRound"].as<int>();
            std::cout << "(CMD) sweepsPerAdaptRound was set to " << sim.simParams.sweepsPerAdaptRound << ".\n";
        }
        if (vm.count("adaptRounds")) {
            sim.simParams.adaptRounds=vm["adaptRounds"].as<int>();
            std::cout << "(CMD) adaptRounds was set to " << sim.simParams.adaptRounds << ".\n";
        }
        if (vm.count("targetAcceptRate")) {
            sim.simParams.targetAcceptRate=vm["targetAcceptRate"].as<int>();
            std::cout << "(CMD) targetAcceptRate was set to " << sim.simParams.targetAcceptRate << ".\n";
        }
        if (vm.count("adaptPercentage")) {
            sim.simParams.adaptPercentage=vm["adaptPercentage"].as<int>();
            std::cout << "(CMD) adaptPercentage was set to " << sim.simParams.adaptPercentage << ".\n";
        }
        if (vm.count("thermalSweeps")) {
            sim.simParams.thermalSweeps=vm["thermalSweeps"].as<int>();
            std::cout << "(CMD) thermalSweeps set to " << sim.simParams.thermalSweeps << ".\n";
        }
        if (vm.count("sweeps")) {
            sim.simParams.sweeps=vm["sweeps"].as<int>();
            std::cout << "(CMD) Non-thermal sweeps set to " << sim.simParams.sweeps << ".\n";
        }
        if (vm.count("printRate")) {
            sim.simParams.printRate=vm["printRate"].as<int>();
            std::cout << "(CMD) printRate set to " << sim.simParams.printRate << ".\n";
        }
        if (vm.count("saveRate")) {
            sim.simParams.saveRate=vm["saveRate"].as<int>();
            std::cout << "(CMD) saveRate set to " << sim.simParams.saveRate << ".\n";
        }
        if (vm.count("temperature")) {
            sim.simParams.T=vm["temperature"].as<double>();
            std::cout << "(CMD) temperature set to " << sim.simParams.T << ".\n";
        }
        if (vm.count("dtau")) {
            sim.simParams.dtau=vm["dtau"].as<double>();
            std::cout << "(CMD) dtau set to " << sim.simParams.dtau << ".\n";
        }

        if (vm.count("r")) {
            sim.modelParams.r=vm["r"].as<double>();
            std::cout << "(CMD) r set to " << sim.modelParams.r << ".\n";
        }
        if (vm.count("lambda")) {
            sim.modelParams.lambda=vm["lambda"].as<double>();
            std::cout << "(CMD) lambda set to " << sim.modelParams.lambda << ".\n";
        }
        if (vm.count("u")) {
            sim.modelParams.u=vm["u"].as<double>();
            std::cout << "(CMD) u set to " << sim.modelParams.u << ".\n";
        }    
        if (vm.count("mu")) {
            sim.modelParams.mu=vm["mu"].as<double>();
            std::cout << "(CMD) mu set to " << sim.modelParams.mu << ".\n";
        }            

    }
    catch(std::exception& e) {
        std::cerr << "CMD handling: " << e.what() << "\n";
        return 0;
    }

    return true;
};

void tryResumeSimulation(Simulation& sim){
    // ## New simulation or resuming old one?
    std::string filename = "";
    for(int i=0; i<1000000; ++i){
        if (!generic::fileExists("temp" + generic::num2str(i) + ".state"))
            break;
        filename = "temp" + generic::num2str(i) + ".state";
        sim.tempStateCounter = (i+1);
        sim.resume = true;
    }

    if (sim.resume){
        std::cout << "Old state file " << filename << " found. Should I resume? [y (default)]" << std::endl;

        // Ask if simulation should be resumed
        char shouldResume;
        std::string input;
        std::getline( std::cin, input );
        if ( !input.empty() ) {
            std::istringstream stream( input );
            stream >> shouldResume;
            if(shouldResume!='y'){
                sim.resume=false;
                generic::print(std::string("=> Starting new simulation in file 'temp")+generic::num2str(sim.tempStateCounter)+std::string(".state' ...\n"));
                return;
            }
        }
        generic::print(std::string("=> Resuming simulation from state '")+filename+std::string("' ...\n"));

        try {
            sim.loadStateFromFile(filename);
        }
        catch(Simulation::StorageException& se) {
            std::cerr << "StorageException: " << se.what() << "\n";
            sim.resume = false;
        }
        catch(std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            sim.resume = false;
        }

        if (!sim.resume)
            std::cout << "Failed to resume simulation => Initializing new simulation..." << std::endl;
    }    
};

void printCurrentTime(){
    namespace pt = boost::posix_time;
    pt::ptime now = pt::second_clock::local_time();
    std::stringstream ss;
    ss << now.time_of_day() << " on " << now.date().day() << "/" << static_cast<int>(now.date().month())
        << "/" << now.date().year();
    std::cout << ss.str() << std::endl;
};

int main(int ac, char* av[]){
    std::cout << "Job started at " ;
    printCurrentTime();

    Simulation sim;

    tryResumeSimulation(sim);

    if(!sim.resume){
        if (!handleCMDOptions(ac, av, sim)) // Ignore cmd arguments when resuming a simulation from file
            return 0;
        
        sim.simParams.init();

        // ## Output Simulation Details
        std::cout << std::endl << "Simulation & Model Parameters" << std::endl;
        std::cout << "\t r=" << sim.modelParams.r << ", ";
        std::cout << "T=" << sim.simParams.T << " (beta=" << sim.simParams.beta <<", m=" << sim.simParams.m <<", dtau=" << sim.simParams.dtau <<"), ";
        std::cout << "L=" << sim.simParams.L << " (" << sim.simParams.N << " sites)" << std::endl;
        //std::cout << "\t #timeslices=" << sim.simParams.m << ", #fromscratchtimeslices=" << sim.simParams.n <<", dn_tau=" << sim.simParams.dn_tau << std::endl << std::endl;
        std::cout << "\t Thermalization sweeps: " << sim.simParams.thermalSweeps << std::endl;
        std::cout << "\t Sweeps: " << sim.simParams.sweeps << std::endl << std::endl;
        // Details
        //std::cout << "dtau=" << sim.simParams.dtau << ", ";
        //std::cout << "lambda=" << sim.modelParams.lambda << ", u=" << sim.modelParams.u <<", mu=" << sim.modelParams.mu << std::endl;

        // ## Init System
        sim.init();
    }     

    // Start timer
    boost::timer::auto_cpu_timer timer(3);
    timer.start();

    // Run Monte Carlo simulation
    sim.run();

    // Stop timer and print report
    timer.stop();
    std::cout << std::endl << "Timing report" << std::endl;
    std::cout << "\t ";
    timer.report();

    std::cout << std::endl << "Logging report" << std::endl;
    sim.logger.report();
    sim.logger.report("final.log");

    sim.writeStateToFile("final.state");
    if (!sim.simParams.keepTempStates){
        std::remove("temp0.gf");
        std::remove("temp0.state");
    }

    std::cout << std::endl << "Job ended at " ;
    printCurrentTime();

    return 0;
}
