#ifndef LOGGER_CPP
#define LOGGER_CPP

#include "generic.hpp"

#include <string>
#include <iostream>

#include "Logger.hpp"

void Logger::report(std::ostream& stream) {
    stream << "\t " << "Accepted " << this->acceptedProposalsCount << " of " << this->proposalCount << " move proposals in total." << std::endl;
    stream << "\t " << "Acceptance rate total: " << generic::num2str((double)acceptedProposalsCount/(double)proposalCount*100,2) << "%" << std::endl;
    stream << "\t " << "Acceptance rate after adaption: " << generic::num2str((double)acceptedProposalsCountDuringAdaption/(double)proposalCountDuringAdaption*100,2) << "%" << std::endl; 
};

void Logger::report() {
    this->report(std::cout); 
};

void Logger::report(std::string filename) {
    std::ofstream file;
    file.open(filename);
    this->report(file);
    file.close();
};

std::string Logger::toString(){
    std::string s = "Logger(";
    s += generic::num2str(this->proposalCount)+",";
    s += generic::num2str(this->acceptedProposalsCount)+",";
    s += generic::num2str(this->proposalCountDuringAdaption)+",";
    s += generic::num2str(this->acceptedProposalsCountDuringAdaption)+",";
    s += ")";
    return s;
};
void Logger::save(std::ostream &strm) {
    strm << this->toString();
};

// -------- Overload stream operators --------

std::ostream& operator<<(std::ostream &strm, Logger &obj) {
    return strm << obj.toString();
}

#endif