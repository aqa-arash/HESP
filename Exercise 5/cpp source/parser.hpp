//import libraries in c++ for a parser
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <fstream>

//parse vtk file
void parseVTKFile(const std::string& filename, std::vector<double>& positions,
               std::vector<double>& velocities, std::vector<double>& masses) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::string line;
    bool readingPoints = false, readingMasses = false, readingVelocities = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }

        if (line.find("POINTS") != std::string::npos) {
            readingPoints = true;
            readingMasses = false;
            readingVelocities = false;
            continue;
        } else if (line.find("SCALARS m double") != std::string::npos) {
            readingPoints = false;
            readingMasses = true;
            readingVelocities = false;
            // Skip the LOOKUP_TABLE line
            std::getline(file, line);
            continue;
        } else if (line.find("VECTORS v double") != std::string::npos) {
            readingPoints = false;
            readingMasses = false;
            readingVelocities = true;
            continue;
        }

        if (readingPoints) {
            std::istringstream iss(line);
            double x, y, z;
            if (iss >> x >> y >> z) {
                positions.push_back(x);
                positions.push_back(y);
                positions.push_back(z);
            }
        } else if (readingMasses) {
            std::istringstream iss(line);
            double mass;
            if (iss >> mass) {
                masses.push_back(mass);
            }
        } else if (readingVelocities) {
            std::istringstream iss(line);
            double vx, vy, vz;
            if (iss >> vx >> vy >> vz) {
                velocities.push_back(vx);
                velocities.push_back(vy);
                velocities.push_back(vz);
            }
        }
    }
    // Sanity check
    if (positions.empty() || velocities.empty() || masses.empty()) {
        std::cerr << "Error: Missing data in VTK file: " << filename << std::endl;
        return;
    }
    if (positions.size() / 3 != velocities.size() / 3 || positions.size() / 3 != masses.size()) {
        std::cerr << "Error: Mismatched data sizes in VTK file: " << filename << std::endl;
        return;
    }
    //std::cout << "Parsed data from VTK file: " << filename << std::endl;
    file.close();
}

void parseConfigFile(const std::string& filename, std::vector<double>& positions,
               std::vector<double>& velocities, std::vector<double>& masses, double & boxSize,
               double& timeStepLength, double & timeStepCount, double & sigma, double & epsilon , int & printInterval) 
               {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
     // Declare variables outside the loop
    std::string vtkFileName;
    std::smatch match;
    std::string line;
    // skip comment lines
    while (std::getline(file, line)) {
        
        if (line.empty() || line[0] == '#') {continue; // Skip empty lines and comments   
               }

        // get the vtk file name for position, velocity, acceleration, and mass
        if (std::regex_search(line, match, std::regex("initialization file:\\s*(\\S+)"))) {
            vtkFileName = match[1];
            //std::cout << "VTK file name: " << vtkFileName << std::endl;
            // Read the VTK file
            parseVTKFile(vtkFileName, positions, velocities, masses);
        }

        if (std::regex_search(line, match, std::regex("time step length:\\s*(\\S+)"))) {
            timeStepLength = std::stod(match[1]);
           // std::cout << "Time step length: " << timeStepLength << std::endl;
        }

        if (std::regex_search(line, match, std::regex("time step count:\\s*(\\S+)"))) {
            timeStepCount = std::stod(match[1]);
          ////  std::cout << "Time step count: " << timeStepCount << std::endl;
        }

        if (std::regex_search(line, match, std::regex("sigma:\\s*(\\S+)"))) {
            sigma = std::stod(match[1]);
          //  std::cout << "Sigma: " << sigma << std::endl;
        }

        if (std::regex_search(line, match, std::regex("epsilon:\\s*(\\S+)"))) {
            epsilon = std::stod(match[1]);
          //  std::cout << "Epsilon: " << epsilon << std::endl;
        }
        if (std::regex_search(line, match, std::regex("print interval:\\s*(\\S+)"))) {
            printInterval = std::stod(match[1]);
           // std::cout << "Print interval: " << printInterval << std::endl;
        }
        if (std::regex_search(line, match, std::regex("box size:\\s*(\\S+)"))) {
            boxSize = std::stod(match[1]);
          //  std::cout << "Box size: " << boxSize << std::endl;
        }
    }}

//write vtk file
void writeVTKFile(const std::string& filename, const std::vector<double>& positions,
                  const std::vector<double>& velocities, const std::vector<double>& masses) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file << "# vtk DataFile Version 4.0" << std::endl;
    file << "hesp visualization file" << std::endl;
    file << "ASCII" << std::endl;
    file << "DATASET UNSTRUCTURED_GRID" << std::endl;
    file << "POINTS " << positions.size()/3 << " double" << std::endl;

    for (size_t i = 0; i < positions.size()/3; ++i) {
        file << positions[i*3] << " " << positions[i*3+1] << " " << positions[i*3+2] << std::endl;
    }
    file<<"CELLS 0 0"<<std::endl;
    file<<"CELL_TYPES 0"<<std::endl;
    file << "POINT_DATA " << positions.size()/3 << std::endl;
    file << "SCALARS m double" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (size_t i = 0; i < masses.size(); ++i) {
        file << masses[i] << std::endl;
    }
    file << "VECTORS v double" << std::endl;
    for (size_t i = 0; i < velocities.size()/3; ++i) {
        file << velocities[i*3] << " " << velocities[i*3+1] << " " << velocities[i*3+2] << std::endl;
    }
    file.close();
}

