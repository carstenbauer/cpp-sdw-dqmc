#ifndef LOGGER_H
#define LOGGER_H

#include "generic.hpp"

#include <string>
#include <iostream>

class Logger {
    public:
    // -------- Attributes --------

    int acceptedProposalsCount = 0;
    int proposalCount = 0;

    int acceptedProposalsCountDuringAdaption = 0;
    int proposalCountDuringAdaption = 0;

    // -------- Methods --------

    Logger(){

    };

    void report(std::ostream& stream);
    void report();

    void report(std::string filename);

    std::string toString();
    void save(std::ostream &strm);

};

// -------- Overload stream operators --------

std::ostream& operator<<(std::ostream &strm, Logger &obj);

#endif