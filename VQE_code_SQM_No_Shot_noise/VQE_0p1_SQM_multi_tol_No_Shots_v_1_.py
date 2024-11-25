
#==============================================================================================
# This file is called VQE_0p1_SQM_multi_tol_No_Shots_v_1_.ipynb
# It is available at https://github.com/emanuele-mendicelli/0p1_Supersymmetric_Quantum_Mechanics
#
# The following code utilizes Qiskit's Variational Quantum Eigensolver (VQE) to find the ground state energy of a 0+1 dimensional supersymmetric quantum system, 
# employing a statevector simulator, without shot noise.
#
# The code was used to generate the results presented in the proceeding: 
# Towards quantum simulation of lower-dimensional supersymmetric lattice models.
#                                 E. Mendicelli and D. Schaich
#                                 https://arxiv.org/abs/2411.15083 [hep-lat] (2024)
# If you find this code (or parts of it) useful, please cite the preprint as well as the code.
#
# emanuelemendicelli@hotmail.it
#==============================================================================================

#================================================================================================================================================
#================================================================================================================================================

# Essential information about the structure of the code for each block "approximately"
"""
Code structure
0) User choices block
1) Load all libraries modules
2) Calculate the Hamiltonian
3) Convert the Hamiltonian into pauli strings
4) Find the ground state energy
5) Functions needed to plot and save data into files
6) Ansatz definition block
7) VQE loops over optimizer tolerance and VQE runs
8) Print on screen some VQE run info and execute all function for plots and save data

"""
#================================================================================================================================================
#================================================================================================================================================

## Define user choice

# Parameters for the construction of the Hamiltonian
superpotential = "DW"                   # Chose a superpontentials "HO" Harmonic Oscillator; "DW" Double Well; "AHO" Anharmonic Oscillator 
N_bosonic_modes = 16                     # Chose the number of bosonic modes allowed (\Lambda)
m = 1                                   # Chose the bosonic and fermionic mass
g = 1                                   # Chose the g interaction strenght present in AHO and DW
mu = 1                                  # Chose the mu interaction strenght present in DW

H_full_or_block = "Block"                # Chose the "Full" for the full Hamiltonian or "Block" for the lower block


# Parameter for the VQE
vqe_steps = 100                         # Chose the number of VQE runs

# Parameters for the optimizer, for now only (COYBLA)
max_iterations = 10000                  # Chose the maximum number of optimizer iterations
initial_tolerance = 1                   # Chose the intial n for 10**(-n) tolerance
final_tolerance = 8                     # Chose the final n for 10**(-n) tolerance

#Parameter for constructing the Ansatz
RealAmplitudes_or_NLocal = "RA"         # Chose "RA" for the ansatz structure RealAmplitudes, or "NL" for NLocal
repetition = 1                          # Chose the number of layer repetitions
rotation_type = "RY"                    # Chose the rotation type "RX", "RY", "RZ". For multiple rotation layer use "RY_RZ" for 2 rotations; "RX_RY_RY" 3 rotations and so on   
entanglment_type ="reverse_linear"      # Chose the entaglement strategies: "None" "full"; "linear"; "reverse_linear"; "pairwise"; "circular"; "sca"

reference_state_choice = False          # Chose True for enter a specific initial state by providing below the circuit, otherwise all qubits are set to the default state of |0⟩
reference_state_name = "h"              # Chose a string for the reference_circuit name


string_reference_state="""              # Enter here the circuit to build the reference state

reference_circuit = QuantumCircuit(N_qubits)
reference_circuit.h([i for i in range(0, N_qubits)])
reference_circuit.barrier()

"""

print_ansatz = True                     # Chose True to save the ansatz as an image file, False to not save it

#================================================================================================================================================
#================================================================================================================================================
## Section importing all needed libraries modules

import sys
#Append the code's directory to the interpreter to search for files
sys.path.append('..')

#Importing the Hamiltonian from the file (Hamiltonian_SQM_0p1.py) inside the folder (hamiltonian_SQM)
from hamiltonian_SQM.Hamiltonian_SQM_0p1 import *

# Needed to convert the Hamiltonian into pauli gates
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp

#Packages for interacting with the operating system
import os

#Packages for numerical calculation
import statistics
import numpy as np


from scipy.sparse.linalg import eigs

#For eigenvalues and eigenvectors
from numpy import linalg 


#Packages for plotting
import matplotlib.pyplot as plt


#Needed to get the time and date
import time
from datetime import datetime

#Needed to print into a file using print() command
from contextlib import redirect_stdout

## Loading modules from Qiskit
# Load quantum
from qiskit.circuit import Parameter, QuantumCircuit

# Load some gates 
from qiskit.circuit.library import CCXGate, CRZGate, RXGate, RYGate,RZGate

# Load Ansatz class
from qiskit.circuit.library import NLocal, TwoLocal, RealAmplitudes, EfficientSU2

# Load optimizers 
from qiskit_algorithms.optimizers import SPSA, COBYLA


# Load Aer Estimator for noiseless statevector simulation
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator

# Load the Variational Quantum Eigensolver
from qiskit_algorithms import VQE


## Not fully necessary modules
# Load basic sound playing machine in Windows
import winsound
#================================================================================================================================================
#================================================================================================================================================

# Calculate the Hamiltonian 
hamiltonian = Hamiltonian_SQM_0p1(superpotential, n_bosonic_modes = N_bosonic_modes, m = m, g = g, mu = mu)

#Defining the Hamiltonian name for plots and files
Hamiltonian_name = superpotential+"_"+H_full_or_block

#================================================================================================================================================
#================================================================================================================================================

#Dictionary to handle the selection of full or block Hamiltonian 
H_full_or_block_dict = {
        "Full": hamiltonian.toarray(),
        "Block": hamiltonian.toarray()[N_bosonic_modes:2*N_bosonic_modes,N_bosonic_modes:2*N_bosonic_modes]
        }

hamiltonian_array = H_full_or_block_dict.get(H_full_or_block, "Choice not present! type " + str([i for i in H_full_or_block_dict.keys()]))

#================================================================================================================================================
#================================================================================================================================================

#Convert the Hamiltonian into Pauli gates using Qiskit function SparsePauliOp
# atol (float): Optional. Absolute tolerance for checking if coefficients are zero (Default: 1e-8). 
Hamiltonian_op = SparsePauliOp.from_operator(hamiltonian_array, atol = 1e-8)

N_qubits = Hamiltonian_op.num_qubits

N_paulis = Hamiltonian_op.size

print(Hamiltonian_op)
print("Number of qubits: ", N_qubits)
print("Number of pauli strings: ", N_paulis)

#================================================================================================================================================
#================================================================================================================================================

# Calculate the eigenvalue and eigenvetor and save the min eigenvalue
eig, eigenV = np.linalg.eig(hamiltonian_array)
ref_value = min(eig).real
print(ref_value)

#================================================================================================================================================
#================================================================================================================================================

#Defining all the functions needed to run and collect the VQE the results

# Function to create a compact string version for number in scientific notation 
def SnString(number):
    # Convert a number in scientific notation
    a = '%E' % number
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]
#============================================================================================================

# Function to convert a numbers in scientific notation using latex
def latex_number_SN(number):
    a = '%.4E' % number
    return "$"+str(a.split('E')[0].rstrip('0').rstrip('.'))+ "\u005c" + "mathrm{e}"+ "{"+str(a.split('E')[1])+"}"+ "$"

#============================================================================================================


# Function to plot the history of the smallest of all the VQE energy versus the optimizer iteration steps
#All the variables needed are global variables
def plot_min_VQE_energy_history():

    plt.figure(figsize=(14,6))
    plt.plot(values, lw=1, marker = "o", label="VQE")
    #Choose log scale for the y-axis if the ground-state energy is larger or equal to zero
    if ref_value >= 0:
        plt.yscale('log')


    #Plot title and axis labels
    fontsize = 15

    plt.title(f"Smallest VQE eigenvalue's convergence history (No Shots) \n" + plot_info_title, fontsize = fontsize)
    plt.xlabel("Optimizer iterations", fontsize = fontsize)
    plt.ylabel("VQE Energy",fontsize = fontsize)
    plt.xticks(fontsize = 12) 
    plt.yticks(fontsize = 12) 

    #Plot a red line for the exact result
    plt.axhline(y=ref_value, color='r', linestyle='-', label= "Exact")

    plt.legend(loc="upper right")

    plt.savefig(os.path.join(folder_name,"Best_vqe_conv_"+Hamiltonian_name+'_.png'), bbox_inches='tight', format='png')

    plt.close()

    #plt.show()
#=======================================================================================================================


# Function to plot all VQE energies versus the VQE runs
#All the variables needed are global variables
def plot_all_VQE_energy():

    plt.figure(figsize=(14,6))
    plt.plot(vqe_eigenvalues, lw=1, marker = "o", label="VQE")
    #Choose log scale for the y-axis if the ground-state energy is larger or equal to zero
    if ref_value >= 0:
        plt.yscale('log')


    #Plot title and axis labels
    fontsize = 15

    plt.title(f"VQE eigenvalues (No Shots) \n" + plot_info_title, fontsize = fontsize)
    plt.xlabel("VQE runs", fontsize = fontsize)
    plt.ylabel("VQE Energy",fontsize = fontsize)
    plt.xticks(fontsize = 12) 
    plt.yticks(fontsize = 12) 

    #Plotting a red line for the exact result
    plt.axhline(y=ref_value, color='r', linestyle='-', label= "Exact")
    plt.axhline(vqe_eigenvalues_median, color='b', linewidth=2, label="Median")

    plt.legend(loc="upper right")

    plt.savefig(os.path.join(folder_name,"all_vqe_results_"+Hamiltonian_name+'_.png'), bbox_inches='tight', format='png')

    plt.close()

    #plt.show()

#=========================================================================================================================

# Function to plot the histogram of all VQE runs energy
#All the variables needed are global variables
def plot_histogram_VQE_energy():
    
    fontsize = 15

    plt.hist(vqe_eigenvalues, bins=vqe_steps, range=[min(ref_value, min(vqe_eigenvalues)), max(vqe_eigenvalues)], color="limegreen",label = "VQE")  # density=False would make counts

    #Line indicating the exact value
    plt.axvline(ref_value, color='k', linewidth=2, label="Exact")

    #Line indicating the median
    plt.axvline(vqe_eigenvalues_median, color='b', linewidth=2, label="Median")


    plt.title(f"VQE runs (No Shots) \n" + plot_info_title, fontsize = fontsize)
    plt.ylabel('bin count', fontsize = fontsize)
    plt.xlabel('ground state energy from VQE', fontsize = fontsize)

    #plt.xticks(np.arange(min(ref_value, min(vqe_eigenvalues)),  max(vqe_eigenvalues)))


    plt.legend(loc="upper right")

    plt.xticks(fontsize = 12) 
    plt.yticks(fontsize = 12) 

    plt.savefig(os.path.join(folder_name, "hist_"+Hamiltonian_name+'_.png'), bbox_inches='tight', format='png')

    plt.close()

    #plt.show()
#=========================================================================================================================

#Function to plot the boxplot of all the VQE energy
#All the variables needed are global variables
def  boxplot_VQE_energy():

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)


    #Line indicating the exact value
    plt.axvline(ref_value, color='k', linewidth=2, label="Exact "+r"($\Lambda$ ="+str(N_bosonic_modes)+")")



    # Creating axes instance
    bp = ax.boxplot(vqe_eigenvalues, patch_artist = True, notch =False, vert = 0)


    colors = ["limegreen"]
    #Double each color because for each box plot the two whiskers and caps are treated separately
    whisker_cap_colors = ["limegreen", "limegreen"]


    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)


    # changing color and linewidth of
    # whiskers
    for whisker, color in zip(bp['whiskers'], whisker_cap_colors):
        whisker.set(color = color,
            linewidth = 1.5,
            linestyle =":")


    # Caps color and linewidth
    for cap, color in zip(bp['caps'], whisker_cap_colors):
        cap.set(color = color,
        linewidth = 2)


    # Median color and linewidth
    for median in bp['medians']:
        median.set(color ='red',
            linewidth = 3)

    # changing style of fliers
    for flier, color in zip(bp['fliers'], colors):
        flier.set(marker ='D', color = "r", alpha = 0.5)
        flier.set_markerfacecolor(color)



    # y-axis labels
    ax.set_yticklabels("")


    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


    #Choose log scale for the y-axis if the ground-state energy is larger or equal to zero
    if ref_value >= 0:
        plt.yscale('log')

    #Line indicating the median
    #vqe_eigenvalues_median = statistics.median(tol_m_02)
    #plt.axvline(vqe_eigenvalues_median, color='b', linewidth=2, label="Median")


    fontsize = 18
    fontsize_axis = 15


    plt.title(f"VQE runs (No Shots) \n" + plot_info_title, fontsize = fontsize)
    plt.xlabel('VQE ground state energy', fontsize = fontsize)
    plt.legend(loc="upper right", fontsize=14)
    plt.yticks(fontsize = fontsize_axis) 
    plt.xticks(fontsize = fontsize_axis) 


    ax.set_ylabel("", fontsize = fontsize)

    #plt.xticks(np.arange(min(ref_value, min(vqe_eigenvalues)),  max(vqe_eigenvalues)))
    #plt.xticks(list(plt.xticks()[0])+[ref_value])
    #ax.set_xlim([1e-20, 1e-12])


    # Dealing with the right y-axis
    ax2 = ax.twinx() 
    ax2.set_ylabel("run time & average iterations ", fontsize = fontsize)
    ax2.set_ylim(ax.get_ylim())


    string_time_iterations = time.strftime("%Hh%Mm%Ss", time.gmtime(time_range)).replace('00h','').replace('00m','') +"__"+str(mean_iterations)


    ax2.set_yticklabels([string_time_iterations])
    ax2.set_yticks(ax.get_yticks())
    plt.yticks(fontsize = fontsize_axis) 


    plt.savefig(os.path.join(folder_name, "boxplot_"+Hamiltonian_name+'_.png'), bbox_inches='tight', format='png')

    plt.close()
#=========================================================================================================================
 

#=========================================================================================================================
    
#Function to print into a file the lines for the latex table
#This file is created in the same folder of the code
#All the variables needed are global variables
def latex_lines_file():
    with open(os.path.join(f"LateX_TL_bm_{N_bosonic_modes}_.txt"), 'a') as file:
                if tolerance == 10**(-1):
                    file.write(ansatz_name.replace('_',' ') +" & "+ latex_number_SN(tolerance) +" & "+ f"{len(vqe_eigenvalues)}/"+ f"{vqe_steps} " + "&" f"{mean_iterations}" + " & "+ latex_number_SN(min_vqe_eingevalue) + " & " + latex_number_SN(min_vqe_eingevalue - ref_value)+ " & "+ latex_number_SN(vqe_eigenvalues_median) + " & " + latex_number_SN(vqe_eigenvalues_median - ref_value)+ " & " + latex_number_SN(ref_value) + " & " + time.strftime("%Hh %Mm %Ss", time.gmtime(time_range)) +"\u005c"+"\u005c"+"\n")
                else:
                    file.write(ansatz_name.replace('_',' ') +" & "+ latex_number_SN(tolerance) +" & "+ f"{len(vqe_eigenvalues)}/"+ f"{vqe_steps} " + "&" f"{mean_iterations}" + " & "+ latex_number_SN(min_vqe_eingevalue) + " & " + latex_number_SN(min_vqe_eingevalue - ref_value)+ " & "+ latex_number_SN(vqe_eigenvalues_median) + " & " + latex_number_SN(vqe_eigenvalues_median - ref_value)+ " & " +"-"+ " & " + time.strftime("%Hh %Mm %Ss", time.gmtime(time_range)) +"\u005c"+"\u005c"+"\n")
    file.close()

#=========================================================================================================================

#=========================================================================================================================

#Function to save into a text file the data needed for the histogram (VQE energies)
#All the variables needed are global variables
def histogram_data():
    with open(os.path.join(folder_name, "histogram_data.txt"), mode ='w') as file:

        for element in vqe_eigenvalues:
            file.write(f"{element}\n")
    file.close()

#=========================================================================================================================

#=========================================================================================================================

#Function to save into a text file the data needed for the boxplot (tolernace, ansatz name, VQE energies)
#All the variables needed are global variables
def boxplot_data():
    with open(os.path.join(folder_name, "boxplot_data_tol_"+SnString(tolerance)+"__"+ansatz_name+"_.txt"), mode ='w') as file:
        
        file.write(ansatz_name+"\n")
        file.write(time.strftime("%Hh%Mm%Ss", time.gmtime(time_range)).replace('00h','').replace('00m','') +"__"+str(mean_iterations)+"\n")
        file.write(SnString(tolerance)+"\n")
                
        for element in vqe_eigenvalues:
            file.write(f"{element}\n")
    file.close()

#=========================================================================================================================


#Functions for saving the run info inside a file
#All the variables needed are global variables
def run_info_file():
    
    with open(os.path.join(folder_name, "run_info.txt"), mode ='w') as f:
        with redirect_stdout(f):
            print(H_full_or_block + " Hamiltonian")
            print("No Shot Noise")
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),"\n")
            print(run_name, "\n")
            print(f"Reference value: {ref_value:.10e}", "\n")

            print("Qubits: ",N_qubits)
            print("VQE runs: ",vqe_steps)
            print("Optimizer: ",optimizer_name)
            print("Max iterations: ",max_iterations)
            print("Optimizers's tolerance: ",SnString(tolerance))
            print("Ansatz: ", ansatz_name, "\n")



            print("All run converged!!!" if len(vqe_eigenvalues) == vqe_steps else f"Only {len(vqe_eigenvalues)} runs out of {vqe_steps} converged")
            print("Number of iterations for each converged VQE run:")
            print(optimizer_iterations, "\n")
            print("Average iterations of converged run:")
            print(mean_iterations,"\n")
            
            print("VQE eigenvalues:")
            print(vqe_eigenvalues,"\n")

            print("Median VQE eigenvalues:")
            print(vqe_eigenvalues_median,"\n")

            print(f"Min eigenvalue after {vqe_steps} VQE steps on Aer qasm simulator: {min_vqe_eingevalue:.5e}")
            print(f"Delta from reference energy value is {(min_vqe_eingevalue - ref_value):.5e}","\n")

            print("Optimizer history of the smallest VQE eigenvalue:")
            print(values,"\n")

            print(f"Smallest VQE eingenvalue optimizer final message:")
            print(vqe_results_info[index])

            print(full_ansatz.decompose())
            print("The number of ansatz parameters is: \n", ansatz.num_parameters)
            
            #print line for latex
            print("Line for latex table")
            print(ansatz_name.replace('_',' ') +" & "+ latex_number_SN(tolerance) +" & "+ f"{len(vqe_eigenvalues)}/"+ f"{vqe_steps} " + "&" f"{mean_iterations}" + " & "+ latex_number_SN(min_vqe_eingevalue) + " & " + latex_number_SN(min_vqe_eingevalue - ref_value)+ " & "+ latex_number_SN(vqe_eigenvalues_median) + " & " + latex_number_SN(vqe_eigenvalues_median - ref_value)+ " & " +"-"+ " & " + time.strftime("%Hh %Mm %Ss", time.gmtime(time_range)) +"\u005c"+"\u005c"+"\n")

            
    f.close()


#================================================================================================================================================
#================================================================================================================================================

## Section to construct all Ansatz related instructions

# Define the angle parameter theta for rotation gates
theta = Parameter("θ")


## Section to create the rotation gate list needed for the ansatz's rotation block
# Dictionary rotation gates
rotation_dict = {
    'RX': RXGate(theta),
    'RY': RYGate(theta),
    'RZ': RZGate(theta)
}

# Split the rotation choice into a list
rotation_list_choice = rotation_type.split("_")

# Create rotation gate list using list comprehension to replace based on dictionary choices
rotation_list = [rotation_dict.get(item) for item in rotation_list_choice]
#===========================================================================================================


# Dictionary defining the ansatzs 
ansatz_dict = {
    "RA" : RealAmplitudes(num_qubits = N_qubits, entanglement = entanglment_type, reps = repetition, insert_barriers=True),
    "NL" : NLocal(num_qubits = N_qubits, rotation_blocks = rotation_list, entanglement = entanglment_type, reps = repetition, insert_barriers=True)
}

# Select the ansatz based on user choice
ansatz = ansatz_dict.get(RealAmplitudes_or_NLocal)


# Section to generate the ansatz name

# Dictionary to convert the entanglement strategy to name for files and plots
entaglement_name_dict = {
    "None" : "noE",
    "full" : "f",
    "linear"  : "l",
    "reverse_linear" : "rl",
    "pairwise" : "p",
    "circular" : "c",
    "sca" : "s"
}

# Dictionary defining the foundamental name of the ansatz
ansatz_name_dict = {
    "RA" : "RA_r"+str(repetition)+"_"+entaglement_name_dict.get(entanglment_type),
    "NL" : "NL_r"+str(repetition)+"_"+entaglement_name_dict.get(entanglment_type)+"_"+'_'.join(rotation_list_choice),

}
#========================================================================================================================

# Create a reference circuit and add it to the ansatz if reference_state_choice is True
# otherwise create the fundamental ansatz
if reference_state_choice:
    
    # Execute the circuit introduced as a string 
    exec(string_reference_state)

    full_ansatz = reference_circuit.compose(ansatz)
    # Generate the ansatz name for plots and files
    ansatz_name = reference_state_name +"_"+ ansatz_name_dict.get(RealAmplitudes_or_NLocal)


else:
    full_ansatz = ansatz
    ansatz_name = ansatz_name_dict.get(RealAmplitudes_or_NLocal)



# Save the ansatz circuit as an image file
if print_ansatz:
    if reference_state_choice:
        # To print the full ansatz in a nice way, the ansatz is first decomposed and then composed with the reference circuit
        reference_circuit.compose(ansatz.decompose()).draw("mpl",filename="ansatz_"+ ansatz_name +"_.png")
        #print(reference_circuit.compose(ansatz.decompose()).draw())
    else:
        ansatz.decompose().draw("mpl",filename="ansatz_"+ ansatz_name +"_.png")
        #print(ansatz.decompose().draw())

#================================================================================================================================================
#================================================================================================================================================

# Define the name of the optimizer
optimizer_name = "COBYLA"

# Please keep in mind: when shots is None and approximation=True: Return an exact expectation value.
noiseless_estimator = AerEstimator(run_options={"shots": None}, approximation=True)

# Generate the tolerance list
tolerance_list = [10**(-n) for n in range(initial_tolerance, final_tolerance + 1)]


# VQE loop only changing the optimizer's tolerance
for t in tolerance_list:
    tolerance = t
    
    # Start the clock to measure each VQE runtime
    start_time = time.time()

    # Function to store intermediate data at each optimization step inside a VQE
    def store_intermediate_result(eval_count, parameters, mean, step):
        values.append(mean)
        steps.append(step)



    # Store the eigenvalue calculated at each VQE step
    vqe_eigenvalues = []

    # Store the final message of each VQE run
    vqe_results_info = []


    tot_values = []
    tot_steps = []

    for i in range(vqe_steps):

        # Store the intermediate optimizer values and steps for each VQE step
        values = []
        steps = []

        vqe = VQE(noiseless_estimator, full_ansatz, optimizer = COBYLA(maxiter=max_iterations, tol = tolerance, disp = True), callback=store_intermediate_result)
        result = vqe.compute_minimum_eigenvalue(operator=Hamiltonian_op)


        # Save only the converged runs (optimizer iteration < max allowed iteration)        
        if result.cost_function_evals < max_iterations:

        
            vqe_eigenvalues.append(result.eigenvalue.real)
            vqe_results_info.append(result)

            tot_values.append(values)
            tot_steps.append(steps)        


    min_vqe_eingevalue = min(vqe_eigenvalues)

    # Stop the clock to measure each VQE run time
    time_range = time.time() - start_time

    print(f"Run Tolerance {(tolerance):.5e} \n ")
    print("All run converged!!! \n" if len(vqe_eigenvalues) == vqe_steps else f"Only {len(vqe_eigenvalues)} runs out of {vqe_steps} converged \n")
    
    # Check for which VQE step the optimizer converged in less than the fixed max iterations
    optimizer_iterations = [data.cost_function_evals for data in vqe_results_info] 
    print(optimizer_iterations, "\n")
    
    # Calculate the mean iteration value
    mean_iterations = round(statistics.mean(optimizer_iterations))
    print("Average iterations of converged run:")
    print(mean_iterations, "\n")

    print(f"Min eigenvalue after {vqe_steps} VQE steps on Aer qasm simulator: {min_vqe_eingevalue:.5e}")
    print(f"Delta min VQE from reference energy value is {(min_vqe_eingevalue - ref_value):.5e}")


    print(f"Exact eingenvalue: {ref_value:.5e} \n")
    print(f"Smallest VQE eingenvalue: {min_vqe_eingevalue:.5e}\n")

    vqe_eigenvalues_median = statistics.median(vqe_eigenvalues)
    print("Median VQE eigenvalues:")
    print(f"{vqe_eigenvalues_median:.5e} \n")
    print(f"Delta median from reference energy value is {(vqe_eigenvalues_median - ref_value):.5e}")
#============================================================================================================================================

# Section needed to generate string names and variables 

    run_name = "VQE_"+str(vqe_steps)+"_"+optimizer_name+"_maxIter_"+str(max_iterations)+"_tol_"+SnString(tolerance)+"_"+ansatz_name

    folder_name = "NShots_"+Hamiltonian_name+"_bm_"+str(N_bosonic_modes)+"_"+run_name

    # Check whether the folder already exist
    if os.path.isdir(folder_name) == False:
        os.makedirs(folder_name)
    else:
        folder_name = folder_name+"__bis"
        os.makedirs(folder_name)



    # Define a standard title info for all plots
    plot_info_title = str(f"{Hamiltonian_name}    nBm = {N_bosonic_modes}    VQE_runs = {vqe_steps}    {optimizer_name}    tol = {SnString(tolerance)}    maxiter = {max_iterations}    {ansatz_name}")
    
    # Enter the index of VQE run to print its history. The default value is the index of the smallest VQE energy

    # Extract the index of the smallest VQE energy
    index_min_eigen = vqe_eigenvalues.index(min_vqe_eingevalue)
    index = index_min_eigen
    
    # Select the list of VQE intermediate values
    values = tot_values[index]    

    print(f"Smallest VQE eingenvalue optimizer final message:")
    print(vqe_results_info[index])


    # Execute all necessary functions to generate plots and save them to files
    plot_min_VQE_energy_history()

    plot_all_VQE_energy()

    plot_histogram_VQE_energy()

    boxplot_VQE_energy()

    boxplot_data()

    histogram_data()

    latex_lines_file()

    run_info_file()
#============================================================================================================

#================================================================================================================================================
#================================================================================================================================================

# Play a sound indicating the end of all runs
frequency = 400  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
