# this code is for preprocessing Yulei's data on 20240817 to the format with features of omnilearn
import uproot
import h5py
import numpy as np
import awkward as ak
from ROOT import TLorentzVector
import matplotlib.pyplot as plt
from optparse import OptionParser

def plot_hist(c_pt, s_pt):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,gridspec_kw={"height_ratios": (3, 1)})
    
    # Define bin edges and centers
    num_bins = 50  # Adjust this value based on your data and preference
    bins = np.linspace(-50, 50, num_bins + 1)  # Define bins from -50 to 50
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Calculate histograms for both datasets
    counts_c, _ = np.histogram(c_pt, bins=num_bins)
    counts_s, _ = np.histogram(s_pt, bins=bins)            

    # Calculate the ratio and its uncertainty
    ratio = counts_c / counts_s
    uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

    # Upper panel: Histograms of C and S
    ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label='c jets', color='blue', density=True, log=True, histtype='step')
    ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label='s jets', color='hotpink', density=True, log=True, histtype='step')
    ax1.set_title('Qibin Data Jets')
    ax1.set_xlabel('Jet Charge')
    ax1.set_ylabel('Normalized Events (log)')
    ax1.legend()
    ax1.grid(True)

    # Lower panel: Ratio plot with uncertainties
    ax2.errorbar(bin_centers, ratio, yerr=uncertainty_ratio, fmt='o', color='black', label='C/S')
    ax2.axhline(1, color='gray', linestyle='--')  # Reference line at ratio=1
    # ax2.set_xlabel('Bin')
    ax2.set_ylabel('C/S')
    ax2.legend()
    ax2.grid(True)

    plt.savefig(f"/global/homes/w/weipow/My_omnilearn_output/0819_plot_jetcharge/yulei_jetcharge.jpg", dpi=300)
    plt.close()

def preprocess(input_path="/pscratch/sd/w/weipow/YuleiData/Yulei_0818.root", output_path="/pscratch/sd/w/weipow/YuleiData/processed_data"):
    # Open the ROOT file using uproot
    file = uproot.open(input_path)

    print(file.keys())

    tree = file["qe"]

    # # print branch names
    # print("below are the branches:")
    # branch_names = tree.keys()
    # for branch in branch_names:
    #     print(branch)


    # load data 
    debug = False
    if debug:
        truth_particles_pid = tree["obj_TruthParticles_PID"].array()[0:100]
        truth_particles_charge = tree["obj_TruthParticles_Charge"].array()[0:100]
        truth_particles_mass = tree["obj_TruthParticles_Mass"].array()[0:100]
        truth_particles_energy = tree["obj_TruthParticles_E"].array()[0:100]
        truth_particles_px = tree["obj_TruthParticles_Px"].array()[0:100]
        truth_particles_py = tree["obj_TruthParticles_Py"].array()[0:100]
        truth_particles_pz = tree["obj_TruthParticles_Pz"].array()[0:100]

    else:
        truth_particles_pid = tree["obj_TruthParticles_PID"].array()
        truth_particles_charge = tree["obj_TruthParticles_Charge"].array()
        truth_particles_mass = tree["obj_TruthParticles_Mass"].array()
        truth_particles_energy = tree["obj_TruthParticles_E"].array()
        truth_particles_px = tree["obj_TruthParticles_Px"].array()
        truth_particles_py = tree["obj_TruthParticles_Py"].array()
        truth_particles_pz = tree["obj_TruthParticles_Pz"].array()



    # pflows_in_up_jet = tree["obj_pflows_in_up_jet"].array()
    # pflows_in_down_jet = tree["obj_pflows_in_down_jet"].array()
    # truth_djet_pt = tree["truth_djet_pt"].array()

    print(len(truth_particles_pid))
    print(truth_particles_pid)
    # print(truth_particles_pid[9997])
    # print(len(pflows_in_up_jet))
    # print(pflows_in_up_jet[9997])
    # print(len(pflows_in_down_jet))
    # print(pflows_in_down_jet[9997])

    # for particle in particles_pid[0]:
    #     if np.abs(particle )<= 6:
    #         print(particle)

    # for i, subarray in enumerate(pflows_in_up_jet):
    #     if len(subarray) > 0:  # Check if the subarray is not empty
    #         print(i)
    #         print(subarray)

    # find the quarks, get their data
    truth_cs_pid = truth_particles_pid[ (truth_particles_pid==3) | (truth_particles_pid==4 )]
    truth_cs_charge = truth_particles_charge[ (truth_particles_pid==3) | (truth_particles_pid==4) ]
    truth_cs_mass = truth_particles_mass[ (truth_particles_pid==3) | (truth_particles_pid==4) ]
    truth_cs_energy = truth_particles_energy[ (truth_particles_pid==3) | (truth_particles_pid==4) ]
    truth_cs_px = truth_particles_px[ (truth_particles_pid==3) | (truth_particles_pid==4) ]
    truth_cs_py = truth_particles_py[ (truth_particles_pid==3) | (truth_particles_pid==4) ]
    truth_cs_pz = truth_particles_pz[ (truth_particles_pid==3) | (truth_particles_pid==4) ]
    
    print(f"truth cs charge: {truth_cs_charge}")
    print(f"truth cs mass: {truth_cs_mass}")
    print(f"truth cs energy: {truth_cs_energy}")
    print(f"truth cs px: {truth_cs_px}")

    # truth_s_charge = truth_particles_charge[ truth_particles_pid==3 ]
    # truth_s_mass = truth_particles_mass[ truth_particles_pid==3 ]
    # truth_s_energy = truth_particles_energy[ truth_particles_pid==3 ]
    # truth_s_px = truth_particles_px[ truth_particles_pid==3 ]
    # truth_s_py = truth_particles_py[ truth_particles_pid==3 ]
    # truth_s_pz = truth_particles_pz[ truth_particles_pid==3 ]
    # print(f"truth s charge: {truth_s_charge}")
    # print(f"truth s mass: {truth_s_mass}")
    # print(f"truth s energy: {truth_s_energy}")
    # print(f"truth s px: {truth_s_px}")

    # truth_c_charge = truth_particles_charge[ truth_particles_pid==4 ]
    # truth_c_mass = truth_particles_mass[ truth_particles_pid==4 ]
    # truth_c_energy = truth_particles_energy[ truth_particles_pid==4 ]
    # truth_c_px = truth_particles_px[ truth_particles_pid==4 ]
    # truth_c_py = truth_particles_py[ truth_particles_pid==4 ]
    # truth_c_pz = truth_particles_pz[ truth_particles_pid==4 ]
    # print(f"truth c charge: {truth_c_charge}")
    # print(f"truth c mass: {truth_c_mass}")
    # print(f"truth c energy: {truth_c_energy}")
    # print(f"truth c px: {truth_c_px}")

    # calculate the delta_R to make the subjet for each quark. Construct a delta_R array.
    print("calculating Delta_R ...")

    cs_subjet_events_pid = []
    cs_subjet_events_charge = []
    cs_subjet_events_mass = []
    cs_subjet_events_energy = []
    cs_subjet_events_px = []
    cs_subjet_events_py = []
    cs_subjet_events_pz = []
    for evt in range( len(truth_cs_pid) ):
        cs_subjet_jets_pid = []
        cs_subjet_jets_charge = []
        cs_subjet_jets_mass = []
        cs_subjet_jets_energy = []
        cs_subjet_jets_px = []
        cs_subjet_jets_py = []
        cs_subjet_jets_pz = []

        for s in range(len(truth_cs_px[evt])):
            cs_subjet_particles_pid = []
            cs_subjet_particles_charge = []
            cs_subjet_particles_mass = []
            cs_subjet_particles_energy = []
            cs_subjet_particles_px = []
            cs_subjet_particles_py = []
            cs_subjet_particles_pz = []

            subjet_LorentzVec = TLorentzVector()
            subjet_LorentzVec.SetPxPyPzE(truth_cs_px[evt][s], truth_cs_py[evt][s], truth_cs_pz[evt][s], truth_cs_energy[evt][s])

            # loop over all particles in the event
            for i in range( len(truth_particles_px[evt]) ):
                particle_LorentzVec = TLorentzVector()
                particle_LorentzVec.SetPxPyPzE(truth_particles_px[evt][i], truth_particles_py[evt][i], truth_particles_pz[evt][i], truth_particles_energy[evt][i])

                Delta_R = subjet_LorentzVec.DeltaR(particle_LorentzVec)

                if Delta_R <= 0.2:
                    cs_subjet_particles_pid.append(truth_particles_pid[evt][i])
                    cs_subjet_particles_charge.append(truth_particles_charge[evt][i])
                    cs_subjet_particles_mass.append(truth_particles_mass[evt][i])
                    cs_subjet_particles_energy.append(truth_particles_energy[evt][i])
                    cs_subjet_particles_px.append(truth_particles_px[evt][i])
                    cs_subjet_particles_py.append(truth_particles_py[evt][i])
                    cs_subjet_particles_pz.append(truth_particles_pz[evt][i])

            cs_subjet_jets_pid.append(cs_subjet_particles_pid)
            cs_subjet_jets_charge.append(cs_subjet_particles_charge)
            cs_subjet_jets_mass.append(cs_subjet_particles_mass)
            cs_subjet_jets_energy.append(cs_subjet_particles_energy)
            cs_subjet_jets_px.append(cs_subjet_particles_px)
            cs_subjet_jets_py.append(cs_subjet_particles_py)
            cs_subjet_jets_pz.append(cs_subjet_particles_pz)
            
        cs_subjet_events_pid.append(cs_subjet_jets_pid)
        cs_subjet_events_charge.append(cs_subjet_jets_charge)
        cs_subjet_events_mass.append(cs_subjet_jets_mass)
        cs_subjet_events_energy.append(cs_subjet_jets_energy)
        cs_subjet_events_px.append(cs_subjet_jets_px)
        cs_subjet_events_py.append(cs_subjet_jets_py)
        cs_subjet_events_pz.append(cs_subjet_jets_pz)

    
    # print(cs_subjet_events_pid)
    # print(len(cs_subjet_events_pid))

    # flatten the array, so there is not events, just a bunch subjets
    truth_cs_pid = ak.flatten(truth_cs_pid) # what quark initiated the subjet

    cs_subjet_events_pid = ak.flatten(cs_subjet_events_pid)
    cs_subjet_events_charge = ak.flatten(cs_subjet_events_charge)
    cs_subjet_events_mass = ak.flatten(cs_subjet_events_mass)
    cs_subjet_events_energy = ak.flatten(cs_subjet_events_energy)
    cs_subjet_events_px = ak.flatten(cs_subjet_events_px)
    cs_subjet_events_py = ak.flatten(cs_subjet_events_py)
    cs_subjet_events_pz = ak.flatten(cs_subjet_events_pz)

    print(len(truth_cs_pid))
    print(len(cs_subjet_events_pid))


    # for each subjet, calculate jet charge. jet charge is the sum of particle charge weighted by pt.
    print("calculating jet charge")

    cs_subjet_events_jetCharge = []
    for subjet in range( len(truth_cs_pid) ):
        particle_pt = np.sqrt(cs_subjet_events_px[subjet]**2 + cs_subjet_events_py[subjet]**2)
        weighted_charge = cs_subjet_events_charge[subjet] * particle_pt
        jet_pt = ak.sum( particle_pt, axis=0 ) 
        if jet_pt==0:
            jet_charge = 0
        else:
            jet_charge = ak.sum(weighted_charge, axis=0) / jet_pt
        
        cs_subjet_events_jetCharge.append(jet_charge)

    cs_subjet_events_jetCharge = np.array(cs_subjet_events_jetCharge)
    print(cs_subjet_events_jetCharge)
    print(len(cs_subjet_events_jetCharge))


    # plot jet charge distribution
    s_jetcharge = cs_subjet_events_jetCharge[np.array(truth_cs_pid)==3]
    c_jetcharge = cs_subjet_events_jetCharge[np.array(truth_cs_pid)==4]

    plot_hist(c_jetcharge, s_jetcharge)
    



if __name__ == '__main__':

    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/QGCMS', help="Folder containing input files")
    (flags, args) = parser.parse_args()

    preprocess()
