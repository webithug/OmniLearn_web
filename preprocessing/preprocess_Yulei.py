# this code is for preprocessing Yulei's data on 20240817 to the format with features of omnilearn
import uproot
import h5py
import numpy as np
import awkward as ak
from ROOT import TLorentzVector
import matplotlib.pyplot as plt
from optparse import OptionParser

def plot_hist(c_pt, s_pt, c_constituents, s_constituents):
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
    ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'u jets. Neutral={c_constituents[0]}, Pos.={c_constituents[1]}, Neg.={c_constituents[2]}', color='blue', density=True, log=True, histtype='step')
    ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f'd jets. Neutral={s_constituents[0]}, Pos.={s_constituents[1]}, Neg.={s_constituents[2]}', color='hotpink', density=True, log=True, histtype='step')
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

def preprocess_truth(input_path="/pscratch/sd/w/weipow/YuleiData/output_0820.root", output_path="/pscratch/sd/w/weipow/YuleiData/processed_data"):
    # Open the ROOT file using uproot
    file = uproot.open(input_path)

    print(file.keys())

    tree = file["qe"]

    # # print branch names
    # print("below are the branches:")
    # branch_names = tree.keys()
    # for branch in branch_names:
    #     print(branch)

    # raise

    debug = True
    if debug:
        truth_particles_pid = tree["obj_TruthParticles_PID"].array()[0:100]
        truth_particles_charge = tree["obj_TruthParticles_Charge"].array()[0:100]
        truth_particles_mass = tree["obj_TruthParticles_Mass"].array()[0:100]
        truth_particles_energy = tree["obj_TruthParticles_E"].array()[0:100]
        truth_particles_px = tree["obj_TruthParticles_Px"].array()[0:100]
        truth_particles_py = tree["obj_TruthParticles_Py"].array()[0:100]
        truth_particles_pz = tree["obj_TruthParticles_Pz"].array()[0:100]
        truth_particles_in_b_had = tree["obj_TruthParticles_in_b_had"].array()[0:100]
        truth_particles_in_b_lep = tree["obj_TruthParticles_in_b_lep"].array()[0:100]
        truth_particles_in_up_jet = tree["obj_TruthParticles_in_up_jet"].array()[0:100]
        truth_particles_in_down_jet = tree["obj_TruthParticles_in_down_jet"].array()[0:100]
        truth_particles_in_top_had = tree["obj_TruthParticles_in_top_had"].array()[0:100]
        truth_particles_in_top_lep = tree["obj_TruthParticles_in_top_lep"].array()[0:100]
        truth_particles_in_W_had = tree["obj_TruthParticles_in_W_had"].array()[0:100]
        truth_particles_in_W_lep = tree["obj_TruthParticles_in_W_lep"].array()[0:100]

    else:
        truth_particles_pid = tree["obj_TruthParticles_PID"].array()
        truth_particles_charge = tree["obj_TruthParticles_Charge"].array()
        truth_particles_mass = tree["obj_TruthParticles_Mass"].array()
        truth_particles_energy = tree["obj_TruthParticles_E"].array()
        truth_particles_px = tree["obj_TruthParticles_Px"].array()
        truth_particles_py = tree["obj_TruthParticles_Py"].array()
        truth_particles_pz = tree["obj_TruthParticles_Pz"].array()
        truth_particles_in_b_had = tree["obj_TruthParticles_in_b_had"].array()
        truth_particles_in_b_lep = tree["obj_TruthParticles_in_b_lep"].array()
        truth_particles_in_up_jet = tree["obj_TruthParticles_in_up_jet"].array()
        truth_particles_in_down_jet = tree["obj_TruthParticles_in_down_jet"].array()
        truth_particles_in_top_had = tree["obj_TruthParticles_in_top_had"].array()
        truth_particles_in_top_lep = tree["obj_TruthParticles_in_top_lep"].array()
        truth_particles_in_W_had = tree["obj_TruthParticles_in_W_had"].array()
        truth_particles_in_W_lep = tree["obj_TruthParticles_in_W_lep"].array()
        


    # create array for storing number of c,s in each event
    c_count = np.abs(ak.sum( truth_particles_pid[np.abs(truth_particles_pid)==4], axis=1 )) / 4
    s_count = np.abs(ak.sum( truth_particles_pid[np.abs(truth_particles_pid)==3], axis=1 )) / 3
    u_count = np.abs(ak.sum( truth_particles_pid[np.abs(truth_particles_pid)==2], axis=1 )) / 2
    d_count = np.abs(ak.sum( truth_particles_pid[np.abs(truth_particles_pid)==1], axis=1 )) / 1
    
    print("count")
    print(len(c_count))
    print(s_count)
    print(s_count)

    # get particle of c jets, and calculate jetcharge
    c_constituents = truth_particles_in_up_jet[c_count>0] # getting events with c quark
    c_constituents_pid = (truth_particles_pid[c_count>0])[c_constituents>0]
    c_constituents_px = (truth_particles_px[c_count>0])[c_constituents>0]
    c_constituents_py = (truth_particles_py[c_count>0])[c_constituents>0]
    c_constituents_charge = (truth_particles_charge[c_count>0])[c_constituents>0]
    c_constituents_pt = np.sqrt( c_constituents_px**2 + c_constituents_py**2 )
    weighted_charge = ak.sum(c_constituents_charge * c_constituents_pt, axis=1)
    c_jet_pt = ak.sum(c_constituents_pt, axis=1)
    c_jet_charge = weighted_charge / c_jet_pt

    c_jet_neutral_count = np.count_nonzero( c_constituents_charge==0 )
    c_jet_pos_count = np.count_nonzero( c_constituents_charge==1 )
    c_jet_neg_count = np.count_nonzero( c_constituents_charge==-1 )

    # print(c_jet_charge)
    # print(len(c_jet_charge))
    # print(c_constituents_pid[0])

    # get particle of s jets
    s_constituents = truth_particles_in_down_jet[s_count>0]
    s_constituents_pid = (truth_particles_pid[s_count>0])[s_constituents>0]
    s_constituents_px = (truth_particles_px[s_count>0])[s_constituents>0]
    s_constituents_py = (truth_particles_py[s_count>0])[s_constituents>0]
    s_constituents_charge = (truth_particles_charge[s_count>0])[s_constituents>0]
    s_constituents_pt = np.sqrt( s_constituents_px**2 + s_constituents_py**2 )
    weighted_charge = ak.sum(s_constituents_charge * s_constituents_pt, axis=1)
    s_jet_pt = ak.sum(s_constituents_pt, axis=1)
    s_jet_charge = weighted_charge / s_jet_pt

    s_jet_neutral_count = np.count_nonzero( s_constituents_charge==0 )
    s_jet_pos_count = np.count_nonzero( s_constituents_charge==1 )
    s_jet_neg_count = np.count_nonzero( s_constituents_charge==-1 )

    print("constituents")
    print(len(c_constituents))
    print(len(c_constituents))
    print(s_constituents[c_constituents==1])

    # print(s_jet_charge)
    # print(len(s_jet_charge))
    # print(s_constituents_pid[0])

    # plot cs jet charge distribution
    plot_hist(c_jet_charge, s_jet_charge, [c_jet_neutral_count, c_jet_pos_count, c_jet_neg_count], [s_jet_neutral_count, s_jet_pos_count, s_jet_neg_count])

    # # get particle of u jets
    # u_constituents = truth_particles_in_up_jet[u_count>0]
    # u_constituents_pid = (truth_particles_pid[u_count>0])[u_constituents>0]
    # u_constituents_px = (truth_particles_px[u_count>0])[u_constituents>0]
    # u_constituents_py = (truth_particles_py[u_count>0])[u_constituents>0]
    # u_constituents_charge = (truth_particles_charge[u_count>0])[u_constituents>0]
    # u_constituents_pt = np.sqrt( u_constituents_px**2 + u_constituents_py**2 )
    # weighted_charge = ak.sum(u_constituents_charge * u_constituents_pt, axis=1)
    # u_jet_pt = ak.sum(u_constituents_pt, axis=1)
    # u_jet_charge = weighted_charge / u_jet_pt

    # u_jet_neutral_count = np.count_nonzero( u_constituents_charge==0 )
    # u_jet_pos_count = np.count_nonzero( u_constituents_charge==1 )
    # u_jet_neg_count = np.count_nonzero( u_constituents_charge==-1 )

    # print(u_jet_charge)
    # print(len(u_jet_charge))

    # # get particle of d jets
    # d_constituents = truth_particles_in_down_jet[d_count>0]
    # d_constituents_pid = (truth_particles_pid[d_count>0])[d_constituents>0]
    # d_constituents_px = (truth_particles_px[d_count>0])[d_constituents>0]
    # d_constituents_py = (truth_particles_py[d_count>0])[d_constituents>0]
    # d_constituents_charge = (truth_particles_charge[d_count>0])[d_constituents>0]
    # d_constituents_pt = np.sqrt( d_constituents_px**2 + d_constituents_py**2 )
    # weighted_charge = ak.sum(d_constituents_charge * d_constituents_pt, axis=1)
    # d_jet_pt = ak.sum(d_constituents_pt, axis=1)
    # d_jet_charge = weighted_charge / d_jet_pt

    # d_jet_neutral_count = np.count_nonzero( d_constituents_charge==0 )
    # d_jet_pos_count = np.count_nonzero( d_constituents_charge==1 )
    # d_jet_neg_count = np.count_nonzero( d_constituents_charge==-1 )

    # print(d_jet_charge)
    # print(len(d_jet_charge))

    # # plot ud jet charge distribution
    # plot_hist(u_jet_charge, d_jet_charge, [u_jet_neutral_count, u_jet_pos_count, u_jet_neg_count], [d_jet_neutral_count, d_jet_pos_count, d_jet_neg_count])

    



def preprocess(input_path="/pscratch/sd/w/weipow/YuleiData/output_0820.root", output_path="/pscratch/sd/w/weipow/YuleiData/processed_data"):
    # Open the ROOT file using uproot
    file = uproot.open(input_path)

    print(file.keys())

    tree = file["qe"]

    # # print branch names
    # print("below are the branches:")
    # branch_names = tree.keys()
    # for branch in branch_names:
    #     print(branch)

    # raise


    # load data 
    debug = True
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

    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 1 ))
    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 2 ))
    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 3 ))
    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 4 ))
    # print(truth_particles_pid)
    # print(ak.flatten(truth_particles_pid))


    



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

    # # print quarks
    # for particle in particles_pid[0]:
    #     if np.abs(particle )<= 6:
    #         print(particle)


    # for i, subarray in enumerate(pflows_in_up_jet):
    #     if len(subarray) > 0:  # Check if the subarray is not empty
    #         print(i)
    #         print(subarray)


    # find the quarks, get their data

    # get cs 
    mask_cs = ( np.abs(truth_particles_pid)==3) | (np.abs(truth_particles_pid)==4 )
    truth_cs_pid = truth_particles_pid[ mask_cs ]
    truth_cs_charge = truth_particles_charge[ mask_cs ]
    truth_cs_mass = truth_particles_mass[ mask_cs ]
    truth_cs_energy = truth_particles_energy[ mask_cs ]
    truth_cs_px = truth_particles_px[ mask_cs ]
    truth_cs_py = truth_particles_py[ mask_cs ]
    truth_cs_pz = truth_particles_pz[ mask_cs ]

    # # get ud
    # mask_ud = ( np.abs(truth_particles_pid)==1) | (np.abs(truth_particles_pid)==2 )
    # truth_cs_pid = truth_particles_pid[ mask_ud ]
    # truth_cs_charge = truth_particles_charge[ mask_ud ]
    # truth_cs_mass = truth_particles_mass[ mask_ud ]
    # truth_cs_energy = truth_particles_energy[ mask_ud ]
    # truth_cs_px = truth_particles_px[ mask_ud ]
    # truth_cs_py = truth_particles_py[ mask_ud ]
    # truth_cs_pz = truth_particles_pz[ mask_ud ]

    
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
    c_neutral_parts_count = 0
    c_pos_parts_count = 0
    c_neg_parts_count = 0
    s_neutral_parts_count = 0
    s_pos_parts_count = 0
    s_neg_parts_count = 0
    u_neutral_parts_count = 0
    u_pos_parts_count = 0
    u_neg_parts_count = 0
    d_neutral_parts_count = 0
    d_pos_parts_count = 0
    d_neg_parts_count = 0
    for subjet in range( len(truth_cs_pid) ):

        particle_pt = np.sqrt(cs_subjet_events_px[subjet]**2 + cs_subjet_events_py[subjet]**2)
        
        # keep only particles of charge < 50
        particle_pt = particle_pt[cs_subjet_events_charge[subjet] < 50]

        len_before = np.array(cs_subjet_events_charge[subjet]).shape[0]

        particle_charge = np.array(cs_subjet_events_charge[subjet])[cs_subjet_events_charge[subjet] < 50]
        
        len_after = len(particle_charge)
        if len_before != len_after:
            print(len_before, len_after)

        weighted_charge = particle_charge * particle_pt
        jet_pt = ak.sum( particle_pt, axis=0 ) 
        if jet_pt==0:
            # print(truth_cs_pid[subjet])
            jet_charge = 0
        else:
            jet_charge = ak.sum(weighted_charge, axis=0) / jet_pt
        
        cs_subjet_events_jetCharge.append(jet_charge)

        # calculate number of neutral, positive, negative particles
        if np.abs(truth_cs_pid[subjet]) == 3: # s
            print("s jet")
            s_neutral_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==0)
            s_pos_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==1)
            s_neg_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==-1)
        elif np.abs(truth_cs_pid[subjet]) == 4: # c
            c_neutral_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==0)
            c_pos_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==1)
            c_neg_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==-1)
        elif np.abs(truth_cs_pid[subjet]) == 1: # d
            d_neutral_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==0)
            d_pos_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==1)
            d_neg_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==-1)
        elif np.abs(truth_cs_pid[subjet]) == 2: # u
            u_neutral_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==0)
            u_pos_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==1)
            u_neg_parts_count += np.count_nonzero(cs_subjet_events_charge[subjet]==-1)



    cs_subjet_events_jetCharge = np.array(cs_subjet_events_jetCharge)
    print(cs_subjet_events_jetCharge)
    print(len(cs_subjet_events_jetCharge))


    # plot jet charge distribution

    s_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==3]
    c_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==4]
    s_constituents = [s_neutral_parts_count, s_pos_parts_count, s_neg_parts_count]
    c_constituents = [c_neutral_parts_count, c_pos_parts_count, c_neg_parts_count]
    plot_hist(c_jetcharge, s_jetcharge, c_constituents, s_constituents)
    print(s_jetcharge)

    # d_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==1]
    # u_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==2]
    # d_constituents = [d_neutral_parts_count, d_pos_parts_count, d_neg_parts_count]
    # u_constituents = [u_neutral_parts_count, u_pos_parts_count, u_neg_parts_count]
    # plot_hist(u_jetcharge, d_jetcharge, u_constituents, d_constituents)

    
    



if __name__ == '__main__':

    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/QGCMS', help="Folder containing input files")
    (flags, args) = parser.parse_args()

    # preprocess()
    preprocess_truth()
