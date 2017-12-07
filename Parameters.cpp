#ifndef PARAMS_CPP
#define PARAMS_CPP

#include "Parameters.hpp"

// Define stream output for structs above
std::ostream& operator<<(std::ostream &strm, ModelParameters &obj) {
    return strm << obj.toString();
}

std::ostream& operator<<(std::ostream &strm, SimulationParameters &obj) {
    return strm << obj.toString();
}

#endif