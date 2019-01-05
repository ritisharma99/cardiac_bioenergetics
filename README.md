cardiac_bioenergetics V 2.0
===========

A set of codes to simulate mitochondrial OXPHOS and energy metabolism in cardiac cells using FE methods. 
The code uses the OpenCMISS libraries (www.opencmiss.org) to set up the simulation framework. The meshes used in these simulations were generated from
2D cross sectional images of cardiomyocytes. 

**required Applications/Packages**
----------------------------------
openCMISS and associated libraries - www.opencmiss.org

**BRANCH INFORMAITON**
----------------------
V 2.0:
 - contains the version 2.0 codes for simulating mitochondrial OXPHOS and subsequent metabolite exchange between mitochondria and myofibrils. 
The simulation uses reaction diffusion equations to model the diffusion of metabolites over a realistic FE mesh dervied from electron 
microsocpy images. 

**What's NEW ?**
----------------------
1. Simplified Mesh with only two mesh regions (mitochondria and myofibrls)
2. Updated description of mt-CK enzyme reaction in mitochondria

**FOLDER INFORMATION**
----------------------
 src/:
 - directory containing the fortran 90 program routine that uses opencmiss libraries to simulate mitochondrial OXHPOS and diffusion of metabolites in a cross section of a cell.

 MESH/:
 - filenames containing 2COMP.1.node/ele/face : trinagle generated mesh files of a 2D cross section from TEM derived rat ventricular myocyte
        
 **FLIE INFORMATION**
----------------------
 The parent directory containins all the necessary inputs (other than mesh files) that are required for the simulations
 1. inputs.txt contains the diffusvity values and intial micromolar concentrations of the metabolites simulated in the model.
 2. Mitochondria_control.cellml (for control mitochondria) and Mitochondria_diabetes.cellml (diabetic mitochondria) 
    contains the ODEs describing mitochondrial OXPHOS reactions (solved alongwith the PDEs using strang splitting). 
 3. Myofibril.cellml contains the ODEs describing myofibrilar reactions (solved alongwith the PDEs using strang splitting). 


RUNNING THE PROGRAM
-------------------
1. Interested users need to install compiled versions of the opencmiss libraries. Instructions and support are available at www.opencmiss.org. 
2. Once installed, generate a Makefile in the cardiac_bioenergetics directory using using CMake: "cmake -DOpenCMISSLibs_DIR=YOUR_OPENCMISS_INSTALL_DIR_HERE ."
3. Build the Fortran source file with the generated Makefile: "make"
4. File 'inputs.txt' contains default settings to run a simulation. Comments (prefixed by #) explain the different input variables.
5. Run the executable: "./src/fortran/cardiac_bioenergetics"
