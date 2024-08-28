# this code is for preprocessing Yulei's data on 20240817 to the format with features of omnilearn
import uproot
import h5py
import numpy as np
import awkward as ak
from ROOT import TLorentzVector
import matplotlib.pyplot as plt
from optparse import OptionParser
from sklearn.metrics import roc_curve, auc


def plot_hist(c_pt, s_pt, c_constituents, s_constituents, CSorUD, Delta_R):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,gridspec_kw={"height_ratios": (3, 1)})
    
    # Define bin edges and centers
    num_bins = 50
    bins = np.linspace(-1, 1, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Calculate histograms for both datasets
    counts_c, _ = np.histogram(c_pt, bins=bins)
    counts_s, _ = np.histogram(s_pt, bins=bins)            

    # Calculate the ratio and its uncertainty
    ratio = counts_c / counts_s
    uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

    # Upper panel: Histograms of C and S
    if CSorUD == "cs":
        # ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'c jets. Neutral={c_constituents[0]}, Pos.={c_constituents[1]}, Neg.={c_constituents[2]}', color='blue', density=True, log=True, histtype='step')
        # ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f's jets. Neutral={s_constituents[0]}, Pos.={s_constituents[1]}, Neg.={s_constituents[2]}', color='hotpink', density=True, log=True, histtype='step')
        ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'c jets', color='blue', density=True, log=True, histtype='step')
        ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f's jets', color='hotpink', density=True, log=True, histtype='step')
    elif CSorUD == "ud":
        # ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'u jets. Neutral={c_constituents[0]}, Pos.={c_constituents[1]}, Neg.={c_constituents[2]}', color='blue', density=True, log=True, histtype='step')
        # ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f'd jets. Neutral={s_constituents[0]}, Pos.={s_constituents[1]}, Neg.={s_constituents[2]}', color='hotpink', density=True, log=True, histtype='step')
        ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'u jets', color='blue', density=True, log=True, histtype='step')
        ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f'd jets', color='hotpink', density=True, log=True, histtype='step')
    elif CSorUD == "udcs":
        # ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'up-type jets. Neutral={c_constituents[0]}, Pos.={c_constituents[1]}, Neg.={c_constituents[2]}', color='blue', density=True, log=True, histtype='step')
        # ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f'down-type jets. Neutral={s_constituents[0]}, Pos.={s_constituents[1]}, Neg.={s_constituents[2]}', color='hotpink', density=True, log=True, histtype='step')
        ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label=f'up-type jets', color='blue', density=True, log=True, histtype='step')
        ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label=f'down-type jets', color='hotpink', density=True, log=True, histtype='step')
    
    ax1.set_title(f'Qibin Data Jets Delta_R={Delta_R}')
    ax1.set_xlabel('Jet Charge')
    ax1.set_ylabel('Normalized Events (log)')
    ax1.legend()
    ax1.grid(True)

    # Lower panel: Ratio plot with uncertainties
    ax2.errorbar(bin_centers, ratio, yerr=uncertainty_ratio, fmt='o', color='black', label=f'{CSorUD[0:int(len(CSorUD)/2)]}/{CSorUD[int(len(CSorUD)/2):]}')
    ax2.axhline(1, color='gray', linestyle='--')  # Reference line at ratio=1
    # ax2.set_xlabel('Bin')
    ax2.set_ylabel(f'{CSorUD[0:int(len(CSorUD)/2)]}/{CSorUD[int(len(CSorUD)/2):]}')
    ax2.legend()
    ax2.grid(True)

    plt.savefig(f"/global/homes/w/weipow/My_omnilearn_output/0819_plot_jetcharge/{Delta_R}_yulei_jetcharge_{CSorUD}.jpg", dpi=300)
    plt.close()

def pad_with_zeros(array, length, fill_value=0):
            return ak.fill_none(ak.pad_none(array, target=length, clip=True), fill_value)

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

    debug = False
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

    print(c_jet_charge)
    print(len(c_jet_charge))
    print(c_constituents_pid[0])

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

    # plot cs jet charge distribution
    plot_hist(c_jet_charge, s_jet_charge, [c_jet_neutral_count, c_jet_pos_count, c_jet_neg_count], [s_jet_neutral_count, s_jet_pos_count, s_jet_neg_count], CSorUD="cs", Delta_R="truth")

    # print("constituents")
    # print(c_constituents[0])
    # print(len(s_constituents[0]))
    # shared_num = len(ak.flatten(s_constituents[(c_constituents==1) & (s_constituents==1)], axis=1))
    # all_s_constituent = len(ak.flatten(s_constituents[s_constituents==1], axis=1))
    # print(shared_num / all_s_constituent)

    # print(s_jet_charge)
    # print(len(s_jet_charge))
    # print(s_constituents_pid[0])


    # get particle of u jets
    u_constituents = truth_particles_in_up_jet[u_count>0]
    u_constituents_pid = (truth_particles_pid[u_count>0])[u_constituents>0]
    u_constituents_px = (truth_particles_px[u_count>0])[u_constituents>0]
    u_constituents_py = (truth_particles_py[u_count>0])[u_constituents>0]
    u_constituents_charge = (truth_particles_charge[u_count>0])[u_constituents>0]
    u_constituents_pt = np.sqrt( u_constituents_px**2 + u_constituents_py**2 )
    weighted_charge = ak.sum(u_constituents_charge * u_constituents_pt, axis=1)
    u_jet_pt = ak.sum(u_constituents_pt, axis=1)
    u_jet_charge = weighted_charge / u_jet_pt

    u_jet_neutral_count = np.count_nonzero( u_constituents_charge==0 )
    u_jet_pos_count = np.count_nonzero( u_constituents_charge==1 )
    u_jet_neg_count = np.count_nonzero( u_constituents_charge==-1 )

    print(u_jet_charge)
    print(len(u_jet_charge))

    # get particle of d jets
    d_constituents = truth_particles_in_down_jet[d_count>0]
    d_constituents_pid = (truth_particles_pid[d_count>0])[d_constituents>0]
    d_constituents_px = (truth_particles_px[d_count>0])[d_constituents>0]
    d_constituents_py = (truth_particles_py[d_count>0])[d_constituents>0]
    d_constituents_charge = (truth_particles_charge[d_count>0])[d_constituents>0]
    d_constituents_pt = np.sqrt( d_constituents_px**2 + d_constituents_py**2 )
    weighted_charge = ak.sum(d_constituents_charge * d_constituents_pt, axis=1)
    d_jet_pt = ak.sum(d_constituents_pt, axis=1)
    d_jet_charge = weighted_charge / d_jet_pt

    d_jet_neutral_count = np.count_nonzero( d_constituents_charge==0 )
    d_jet_pos_count = np.count_nonzero( d_constituents_charge==1 )
    d_jet_neg_count = np.count_nonzero( d_constituents_charge==-1 )

    print(d_jet_charge)
    print(len(d_jet_charge))

    # plot ud jet charge distribution
    plot_hist(u_jet_charge, d_jet_charge, [u_jet_neutral_count, u_jet_pos_count, u_jet_neg_count], [d_jet_neutral_count, d_jet_pos_count, d_jet_neg_count], CSorUD="ud", Delta_R="truth")


    # compute ROC
    # print(labels)
    # Binary_labels = labels[:,0]
    # print(Binary_labels)

    # fpr, tpr, thresholds = roc_curve(Binary_labels, jet_charge)
    # roc_auc = auc(fpr, tpr)  # Calculate area under the curve

def preprocess_truth_omnilearn_features(input_path="/pscratch/sd/w/weipow/YuleiData/ntuples.boosted.v0821.2M.0.root", output_path="/pscratch/sd/w/weipow/YuleiData/processed_data"):
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

    debug = False
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

    


    def create_data(pid=4):
        # find the c jets (quarks) and their info
        c_mask = (np.abs(truth_particles_pid)==pid)
        c_jet_pid = truth_particles_pid[c_mask]
        c_jet_mass = truth_particles_mass[c_mask]
        c_jet_energy = truth_particles_energy[c_mask]
        c_jet_px = truth_particles_px[c_mask]
        c_jet_py = truth_particles_py[c_mask]
        c_jet_pz = truth_particles_pz[c_mask]
        
        c_jet_pt = np.sqrt(c_jet_px**2 + c_jet_py**2)
        c_jet_theta = np.arctan2(c_jet_pt, c_jet_pz)
        c_jet_eta = -np.log(np.tan(c_jet_theta / 2)) 
        c_jet_phi = np.arctan2(c_jet_py, c_jet_px)
        
        c_jet_pt = ak.flatten(c_jet_pt)
        c_jet_eta = ak.flatten(c_jet_eta)
        c_jet_phi = ak.flatten(c_jet_phi)
        c_jet_mass = ak.flatten(c_jet_mass)
        c_jet_energy = ak.flatten(c_jet_energy)
        c_jet_pid = ak.flatten(c_jet_pid)

        print(len(c_jet_pt))
        print(len(c_jet_eta))
        print(len(c_jet_phi))
        print(len(c_jet_mass))

        # create array for storing number of c,s in each event
        c_count = ak.sum( np.abs(truth_particles_pid[np.abs(truth_particles_pid)==pid]), axis=1 ) / pid
        
        # get particle of c jets, and calculate the jet info and particle info
        c_constituents = truth_particles_in_up_jet[c_count>0] # getting events with c quark
        c_constituents_pid = (truth_particles_pid[c_count>0])[c_constituents>0]
        c_constituents_px = (truth_particles_px[c_count>0])[c_constituents>0]
        c_constituents_py = (truth_particles_py[c_count>0])[c_constituents>0]
        c_constituents_pz = (truth_particles_pz[c_count>0])[c_constituents>0]
        c_constituents_charge = (truth_particles_charge[c_count>0])[c_constituents>0]
        c_constituents_energy = (truth_particles_energy[c_count>0])[c_constituents>0]
        c_constituents_pt = np.sqrt( c_constituents_px**2 + c_constituents_py**2 )

        

        c_min_num_constituents = ak.min(ak.num(c_constituents_pid))
        print(f"min = {c_min_num_constituents}")


        print(len(c_count))
        print(c_count)
        # print(c_constituents)
        # print(len(c_constituents))
        # print(c_constituents_pid[0])
        # print(c_max_num_constituents)

        # order particles in pt
        top_num = 40 # keep top 20 pt particles
        sorted_indices = ak.argsort(c_constituents_pt, ascending=False)[:, :top_num]

        c_constituents_pid = c_constituents_pid[sorted_indices]
        c_constituents_px = c_constituents_px[sorted_indices]
        c_constituents_py = c_constituents_py[sorted_indices]
        c_constituents_pz = c_constituents_pz[sorted_indices]
        c_constituents_charge = c_constituents_charge[sorted_indices]
        c_constituents_pt = c_constituents_pt[sorted_indices]
        c_constituents_energy = c_constituents_energy[sorted_indices]

        c_constituents_pid = pad_with_zeros(c_constituents_pid, top_num)
        print(c_constituents_pid[9])

        c_constituents_px = pad_with_zeros(c_constituents_px, top_num)
        c_constituents_py = pad_with_zeros(c_constituents_py, top_num)
        c_constituents_pz = pad_with_zeros(c_constituents_pz, top_num)
        c_constituents_charge = pad_with_zeros(c_constituents_charge, top_num)
        c_constituents_pt = pad_with_zeros(c_constituents_pt, top_num)
        c_constituents_energy = pad_with_zeros(c_constituents_energy, top_num)

        c_constituents_theta = np.arctan2(c_constituents_pt, c_constituents_pz)
        c_constituents_eta = -np.log(np.tan(c_constituents_theta / 2)) 
        c_constituents_phi = np.arctan2(c_constituents_py, c_constituents_px)

        print(len(c_constituents_pt))
        print(len(c_constituents_pid))

        # create the features as omnilearn_cms
        delta_eta = c_constituents_eta - c_jet_eta[:, None]
        delta_phi = c_constituents_phi - c_jet_phi[:, None]
        mask = delta_phi > np.pi
        delta_phi = delta_phi - 2 * np.pi * mask
        mask = delta_phi <= -np.pi
        delta_phi = delta_phi + 2 * np.pi * mask
        log_partpt_jetpt = np.log(1.0 - c_constituents_pt/c_jet_pt[:, None])
        log_partpt = np.log(c_constituents_pt)
        log_parte_jete = np.log(1.0 - c_constituents_energy/c_jet_energy[:, None])
        log_parte = np.log(c_constituents_energy)
        hypot_partpt_parteta = np.sqrt(c_constituents_pt**2 + c_constituents_eta**2)

        # make features from pid
        c_constituents_charge = np.sign(c_constituents_pid) * (c_constituents_pid!=22) * (c_constituents_pid!=130)
        c_constituents_pionkaonproton = (np.abs(c_constituents_pid) == 211) | (np.abs(c_constituents_pid) == 321) | (np.abs(c_constituents_pid) == 2212)
        c_constituents_kaonneutronundefine = (np.abs(c_constituents_pid)==130) | (np.abs(c_constituents_pid) == 2112) | (c_constituents_pid == 0)
        c_constituents_photon = np.abs(c_constituents_pid)==22 
        c_constituents_electron = np.abs(c_constituents_pid)==11
        c_constituents_muon = np.abs(c_constituents_pid)==13

        print(len(ak.num(delta_eta)))
        print(ak.num(log_parte))
        print(ak.to_numpy(log_parte))

        # point[:,:,0] = c_constituents_pt
        zero_array = ak.zeros_like(c_jet_mass) # since this is truth data, c_jet_mass would make accuracy 100 percent
        jet_X = np.stack([c_jet_pt, c_jet_eta, c_jet_phi, zero_array], -1)

        c_data_X = np.concatenate([jet_X, ak.to_numpy(log_partpt), 
                                ak.to_numpy(log_parte), ak.to_numpy(delta_eta), ak.to_numpy(delta_phi),
                                  ak.to_numpy(log_partpt_jetpt), 
                                ak.to_numpy(log_parte_jete), 
                                ak.to_numpy(hypot_partpt_parteta), ak.to_numpy(c_constituents_charge),
                                ak.to_numpy(c_constituents_pionkaonproton), ak.to_numpy(c_constituents_kaonneutronundefine),
                                ak.to_numpy(c_constituents_photon), ak.to_numpy(c_constituents_electron),
                                ak.to_numpy(c_constituents_muon)], axis=-1) 
        
        # # no jet_X
        # c_data_X = np.concatenate([ak.to_numpy(delta_eta), ak.to_numpy(delta_phi), 
        #                         ak.to_numpy(log_partpt_jetpt), ak.to_numpy(log_partpt), 
        #                         ak.to_numpy(log_parte_jete), ak.to_numpy(log_parte), 
        #                         ak.to_numpy(hypot_partpt_parteta), ak.to_numpy(c_constituents_charge),
        #                         ak.to_numpy(c_constituents_pionkaonproton), ak.to_numpy(c_constituents_kaonneutronundefine),
        #                         ak.to_numpy(c_constituents_photon), ak.to_numpy(c_constituents_electron),
        #                         ak.to_numpy(c_constituents_muon)], axis=-1) 

        c_data_X_array = ak.to_numpy(c_data_X)
        c_data_Y_array = ak.to_numpy(c_jet_pid)
        print(c_data_X_array.shape)
        print(c_data_Y_array.shape)

        return c_data_X_array, c_data_Y_array

    c_X, c_Y = create_data(4)
    c_X[np.isinf(c_X)] = 0
    c_Y[:] = 1

    s_X, s_Y = create_data(3)
    s_X[np.isinf(s_X)] = 0
    s_Y[:] = 0

    u_X, u_Y = create_data(2)
    u_X[np.isinf(u_X)] = 0
    u_Y[:] = 1

    d_X, d_Y = create_data(1)
    d_X[np.isinf(d_X)] = 0
    d_Y[:] = 0

    print(d_X.shape)


    data_X = np.concatenate([c_X, s_X, u_X, d_X], 0)
    data_Y = np.concatenate([c_Y, s_Y, u_Y, d_Y], 0)

    print(data_X.shape)
    print(data_Y.shape)
    print(data_Y)

    np.save('/global/homes/w/weipow/xgboost/Yulei_data_X.npy', data_X)
    np.save('/global/homes/w/weipow/xgboost/Yulei_data_Y.npy', data_Y)






    
# try not to use for loop
def preprocess(input_path="/pscratch/sd/w/weipow/YuleiData/output_0820.root", output_path="/pscratch/sd/w/weipow/YuleiData/processed_data", CSorUD="cs", Delta_R_threshold=0.2):
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
    debug = False
    if debug:
        truth_particles_pid = tree["obj_TruthParticles_PID"].array()[0:100]
        truth_particles_charge = tree["obj_TruthParticles_Charge"].array()[0:100]
        truth_particles_mass = tree["obj_TruthParticles_Mass"].array()[0:100]
        truth_particles_energy = tree["obj_TruthParticles_E"].array()[0:100]
        truth_particles_px = tree["obj_TruthParticles_Px"].array()[0:100]
        truth_particles_py = tree["obj_TruthParticles_Py"].array()[0:100]
        truth_particles_pz = tree["obj_TruthParticles_Pz"].array()[0:100]

        truth_subjet_flavor = tree["obj_smallRjet_TruthFlavor"].array()[0:100]
        truth_subjet_pt = tree["obj_smallRjet_PT"].array()[0:100]
        truth_subjet_eta = tree["obj_smallRjet_Eta"].array()[0:100]
        truth_subjet_phi = tree["obj_smallRjet_Phi"].array()[0:100]
        truth_subjet_mass = tree["obj_smallRjet_Mass"].array()[0:100]

    else:
        truth_particles_pid = tree["obj_TruthParticles_PID"].array()
        truth_particles_charge = tree["obj_TruthParticles_Charge"].array()
        truth_particles_mass = tree["obj_TruthParticles_Mass"].array()
        truth_particles_energy = tree["obj_TruthParticles_E"].array()
        truth_particles_px = tree["obj_TruthParticles_Px"].array()
        truth_particles_py = tree["obj_TruthParticles_Py"].array()
        truth_particles_pz = tree["obj_TruthParticles_Pz"].array()
        
        truth_subjet_flavor = tree["obj_smallRjet_TruthFlavor"].array()
        truth_subjet_pt = tree["obj_smallRjet_PT"].array()
        truth_subjet_eta = tree["obj_smallRjet_Eta"].array()
        truth_subjet_phi = tree["obj_smallRjet_Phi"].array()
        truth_subjet_mass = tree["obj_smallRjet_Mass"].array()


    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 1 ))
    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 2 ))
    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 3 ))
    print(np.count_nonzero( np.abs(ak.flatten(truth_particles_pid)) == 4 ))
    # print(truth_subjet_flavor[3])
    # print(truth_subjet_pt[3])
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
    if CSorUD == "cs":
        # get cs 
        mask_cs = ( np.abs(truth_subjet_flavor)==3) | ( np.abs(truth_subjet_flavor)==4 )
        truth_cs_pid = truth_subjet_flavor[ mask_cs ]
        truth_cs_mass = truth_subjet_mass[ mask_cs ]
        truth_cs_pt = truth_subjet_pt[ mask_cs ]
        truth_cs_eta = truth_subjet_eta[ mask_cs ]
        truth_cs_phi = truth_subjet_phi[ mask_cs ]
    elif CSorUD == "ud":
        # get ud
        mask_ud = ( np.abs(truth_subjet_flavor)==1 ) | ( np.abs(truth_subjet_flavor)==2 )
        truth_cs_pid = truth_subjet_flavor[ mask_ud ]
        truth_cs_mass = truth_subjet_mass[ mask_ud ]
        truth_cs_pt = truth_subjet_pt[ mask_ud ]
        truth_cs_eta = truth_subjet_eta[ mask_ud ]
        truth_cs_phi = truth_subjet_phi[ mask_ud ]
    else:
        # plot all udcs
        mask_ud = (np.abs(truth_subjet_flavor)==1) | (np.abs(truth_subjet_flavor)==2) | (np.abs(truth_subjet_flavor)==3) | (np.abs(truth_subjet_flavor)==4)
        truth_cs_pid = truth_subjet_flavor[ mask_ud ]
        truth_cs_mass = truth_subjet_mass[ mask_ud ]
        truth_cs_pt = truth_subjet_pt[ mask_ud ]
        truth_cs_eta = truth_subjet_eta[ mask_ud ]
        truth_cs_phi = truth_subjet_phi[ mask_ud ]

    
    print(f"truth cs mass: {truth_cs_mass}")

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
    # loop over all events
    for evt in range( len(truth_cs_pid) ):
        cs_subjet_jets_pid = []
        cs_subjet_jets_charge = []
        cs_subjet_jets_mass = []
        cs_subjet_jets_energy = []
        cs_subjet_jets_px = []
        cs_subjet_jets_py = []
        cs_subjet_jets_pz = []

        # loop over all cs as subjet
        for s in range(len(truth_cs_pid[evt])):
            cs_subjet_particles_pid = []
            cs_subjet_particles_charge = []
            cs_subjet_particles_mass = []
            cs_subjet_particles_energy = []
            cs_subjet_particles_px = []
            cs_subjet_particles_py = []
            cs_subjet_particles_pz = []

            subjet_LorentzVec = TLorentzVector()
            subjet_LorentzVec.SetPtEtaPhiM(truth_cs_pt[evt][s], truth_cs_eta[evt][s], truth_cs_phi[evt][s], truth_cs_mass[evt][s])

            # loop over all particles in the event
            for i in range( len(truth_particles_px[evt]) ):
                particle_LorentzVec = TLorentzVector()
                particle_LorentzVec.SetPxPyPzE(truth_particles_px[evt][i], truth_particles_py[evt][i], truth_particles_pz[evt][i], truth_particles_energy[evt][i])

                Delta_R = subjet_LorentzVec.DeltaR(particle_LorentzVec)

                if Delta_R <= Delta_R_threshold:
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
            # print("s jet")
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
    if CSorUD == "cs":
        s_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==3]
        c_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==4]
        s_constituents = [s_neutral_parts_count, s_pos_parts_count, s_neg_parts_count]
        c_constituents = [c_neutral_parts_count, c_pos_parts_count, c_neg_parts_count]
        plot_hist(c_jetcharge, s_jetcharge, c_constituents, s_constituents, "cs", Delta_R_threshold)
        print(s_jetcharge)
    elif CSorUD == "ud":
        d_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==1]
        u_jetcharge = cs_subjet_events_jetCharge[np.array(np.abs(truth_cs_pid))==2]
        d_constituents = [d_neutral_parts_count, d_pos_parts_count, d_neg_parts_count]
        u_constituents = [u_neutral_parts_count, u_pos_parts_count, u_neg_parts_count]
        print(d_jetcharge)
        print(u_jetcharge)
        plot_hist(u_jetcharge, d_jetcharge, u_constituents, d_constituents, "ud", Delta_R_threshold)
    else:
        d_jetcharge = cs_subjet_events_jetCharge[(np.array(np.abs(truth_cs_pid))==1) | (np.array(np.abs(truth_cs_pid))==3)]
        u_jetcharge = cs_subjet_events_jetCharge[(np.array(np.abs(truth_cs_pid))==2)| (np.array(np.abs(truth_cs_pid))==4)]
        d_constituents = [d_neutral_parts_count, d_pos_parts_count, d_neg_parts_count]
        u_constituents = [u_neutral_parts_count, u_pos_parts_count, u_neg_parts_count]
        print(d_jetcharge)
        print(u_jetcharge)
        plot_hist(u_jetcharge, d_jetcharge, u_constituents, d_constituents, "udcs", Delta_R_threshold)


    
    



if __name__ == '__main__':

    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/QGCMS', help="Folder containing input files")
    parser.add_option("--CSorUD", type="string", default='cs', help="cs or ud")
    parser.add_option("--Delta_R", type="float", default=0.2 , help="Delta_R for finding constituents")
    (flags, args) = parser.parse_args()

    # preprocess(CSorUD=flags.CSorUD, Delta_R_threshold=flags.Delta_R)
    # preprocess_truth()
    preprocess_truth_omnilearn_features()
