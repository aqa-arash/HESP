//import libraries in c++ for a parser
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

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
    std::string vtkFileName;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "initialization" && line.find("file:") != std::string::npos) {
            vtkFileName = line.substr(line.find("file:") + 5);
            vtkFileName.erase(0, vtkFileName.find_first_not_of(" \t"));
            parseVTKFile(vtkFileName, positions, velocities, masses);
        } else if (key == "time" && line.find("step") != std::string::npos && line.find("length:") != std::string::npos) {
            timeStepLength = std::stod(line.substr(line.find("length:") + 7));
        } else if (key == "time" && line.find("step") != std::string::npos && line.find("count:") != std::string::npos) {
            timeStepCount = std::stod(line.substr(line.find("count:") + 6));
        } else if (key == "sigma:") {
            sigma = std::stod(line.substr(line.find("sigma:") + 6));
        } else if (key == "epsilon:") {
            epsilon = std::stod(line.substr(line.find("epsilon:") + 8));
        } else if (key == "print" && line.find("interval:") != std::string::npos) {
            printInterval = std::stoi(line.substr(line.find("interval:") + 9));
        } else if (key == "box" && line.find("size:") != std::string::npos) {
            boxSize = std::stod(line.substr(line.find("size:") + 5));
        }
    }
}

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
    file << "CELLS 0 0" << std::endl;
    file << "CELL_TYPES 0" << std::endl;
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
