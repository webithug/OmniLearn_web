import uproot
import h5py
import numpy as np
import awkward as ak
from ROOT import TLorentzVector
import matplotlib.pyplot as plt
from optparse import OptionParser



def check_qibin_mismatch(fjet1_pflows_pt, fjet1_pflows_eta, fjet1_pflows_phi, fjet1_pflows_mass, fjet_pt, fjet_eta, fjet_phi, fjet_mass):

    diff_pt_list=[]
    diff_eta_list=[]
    diff_phi_list=[]
    diff_m_list=[]
    
    for evt in range(len(fjet1_pflows_eta)):
        # check if the number of particles are the same
        a = len(fjet1_pflows_pt[evt])
        b = len(fjet1_pflows_eta[evt]) 
        c = len(fjet1_pflows_phi[evt]) 
        d = len(fjet1_pflows_mass[evt]) 
        if a != b:
            print(f"At event {evt}, number of particles not equal: {a, b, c, d}")
            
            continue

        # create fjet_LorentzVector
        fjet_LorentzVec = TLorentzVector()
        fjet_LorentzVec.SetPtEtaPhiM(fjet_pt[evt], fjet_eta[evt], fjet_phi[evt], fjet_mass[evt])

        # create LorentzVector to sum all particles in a fjet
        total_vector = TLorentzVector()

        # loop over particels and sum
        for i in range( len(fjet1_pflows_eta[evt]) ):
            particle_LorentzVec = TLorentzVector()
            particle_LorentzVec.SetPtEtaPhiM(fjet1_pflows_pt[evt][i], fjet1_pflows_eta[evt][i], fjet1_pflows_phi[evt][i], fjet1_pflows_mass[evt][i])
            
            total_vector += particle_LorentzVec


        # if fjet_LorentzVec == total_vector:
        #     print("4 vector match")
        # else:
        #     print(f"4 vector doesn't match at eventt {evt}: ({fjet_LorentzVec.Pt(), fjet_LorentzVec.Eta(), fjet_LorentzVec.Phi(), fjet_LorentzVec.M()}) and ({total_vector.Pt(), total_vector.Eta(), total_vector.Phi(), total_vector.M()})")

        diff_vec = (fjet_LorentzVec - total_vector)

        diff_pt_list.append(diff_vec.Pt())
        diff_m_list.append(diff_vec.E())
        diff_eta_list.append(diff_vec.Eta())
        diff_phi_list.append(diff_vec.Phi())

    for dim in ["pt", "m", "eta", "phi"]:
        
        if dim == "pt":
            plt.plot(diff_pt_list)
        elif dim == "m":
            plt.plot(diff_m_list)
        elif dim == "eta":
            plt.plot(diff_eta_list)
        else:
            plt.plot(diff_phi_list)
        
        plt.title(dim)
        plt.savefig(f"/global/homes/w/weipow/My_omnilearn_output/0813_qibin_4vec_mismatch/{dim}.png", dpi=600)
        plt.close()

def pad_array(array, n_events, n_subjets, n_particles):
        padded_array = np.zeros( (n_events, n_subjets, n_particles) )
        for evt in range( len(array) ):
            padded_array[evt][:len(array[evt])] += ak.to_numpy(array[evt])

        return padded_array

def balance_classes(x, y, z):
    """
    Balances the classes in the dataset by randomly discarding entries from the more populous class.

    Parameters:
        x (numpy.ndarray): Input array of shape (N, P, F) where N is the number of samples,
                           P is the dimension of each sample, and F is the number of features.
        y (numpy.ndarray): Label array of shape (N,) where each entry is either 0 or 1.

    Returns:
        tuple: A tuple containing the balanced input array and label array.
    """
    # Find indices for each class
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    
    # Determine the minority class and its size
    if len(indices_0) > len(indices_1):
        minority_size = len(indices_1)
        indices_0 = np.random.choice(indices_0, minority_size, replace=False)
    else:
        minority_size = len(indices_0)
        indices_1 = np.random.choice(indices_1, minority_size, replace=False)
    
    # Combine and shuffle indices
    balanced_indices = np.concatenate([indices_0, indices_1])
    np.random.shuffle(balanced_indices)
    
    # Extract samples and labels corresponding to these indices
    x_balanced = x[balanced_indices]
    y_balanced = y[balanced_indices]
    z_balanced = z[balanced_indices]
    return x_balanced, y_balanced, z_balanced

def preprocess(input_path="/pscratch/sd/w/weipow/QibinData/Dr7v8a.analysis_mtt800.300k.0.root", output_path="/pscratch/sd/w/weipow/QibinData"):
    # Open the ROOT file using uproot
    file = uproot.open(input_path)

    print(file.keys())

    tree = file["t"]

    # print branch names
    print("below are the branches:")
    branch_names = tree.keys()
    for branch in branch_names:
        print(branch)

    jets_eta = tree["jets_eta"].array()
    truth_ujet_pt = tree["truth_ujet_pt"].array()
    truth_djet_pt = tree["truth_djet_pt"].array()

    print(len(jets_eta))
    print(len(truth_ujet_pt))
    print(len(truth_djet_pt))
    raise

    # use less data when debug
    debug = False
    if debug == True:
        # subjet data
        jets_eta = tree["jets_eta"].array()[:10000]
        jets_phi = tree["jets_phi"].array()[:10000]
        jets_flavor_truth = tree["jets_flavor_truth"].array()[:10000]
        jets_charge = tree["jets_charge"].array()[:10000]
        jets_pt = tree["jets_pt"].array()[:10000]
        jets_mass = tree["jets_mass"].array()[:10000]
        jets_n = tree["jets_n"].array()[:10000]

        # particle data
        fjet1_pflows_eta = tree["fjet1_pflows_eta"].array()[:10000]
        fjet1_pflows_phi = tree["fjet1_pflows_phi"].array()[:10000]
        fjet1_pflows_pt = tree["fjet1_pflows_pt"].array()[:10000]
        fjet1_pflows_mass = tree["fjet1_pflows_mass"].array()[:10000]
        fjet1_pflows_charge = tree["fjet1_pflows_charge"].array()[:10000]
        fjet1_pflows_pid = tree["fjet1_pflows_pid"].array()[:10000]

        # # fatjet data
        # fjet_pt = tree["fjet_pt"].array()
        # fjet_eta = tree["fjet_eta"].array()
        # fjet_phi = tree["fjet_phi"].array()
        # fjet_mass = tree["fjet_mass"].array()

        # main top jet
        top_eta = tree["top_eta"].array()[:10000]
        top_phi = tree["top_phi"].array()[:10000]
        anti_top_eta = tree["anti_top_eta"].array()[:10000]
        anti_top_phi = tree["anti_top_phi"].array()[:10000]
        charge = tree["charge"].array()[:10000]

        reco_cut = tree["reco_cut"].array()[:10000]

    else:
        # subjet data
        jets_eta = tree["jets_eta"].array()
        jets_phi = tree["jets_phi"].array()
        jets_flavor_truth = tree["jets_flavor_truth"].array()
        jets_charge = tree["jets_charge"].array()
        jets_pt = tree["jets_pt"].array()
        jets_mass = tree["jets_mass"].array()
        jets_n = tree["jets_n"].array()

        # particle data
        fjet1_pflows_eta = tree["fjet1_pflows_eta"].array()
        fjet1_pflows_phi = tree["fjet1_pflows_phi"].array()
        fjet1_pflows_pt = tree["fjet1_pflows_pt"].array()
        fjet1_pflows_mass = tree["fjet1_pflows_mass"].array()
        fjet1_pflows_charge = tree["fjet1_pflows_charge"].array()
        fjet1_pflows_pid = tree["fjet1_pflows_pid"].array()

        # # fatjet data
        # fjet_pt = tree["fjet_pt"].array()
        # fjet_eta = tree["fjet_eta"].array()
        # fjet_phi = tree["fjet_phi"].array()
        # fjet_mass = tree["fjet_mass"].array()

        # main top jet
        top_eta = tree["top_eta"].array()
        top_phi = tree["top_phi"].array()
        anti_top_eta = tree["anti_top_eta"].array()
        anti_top_phi = tree["anti_top_phi"].array()
        charge = tree["charge"].array()

        reco_cut = tree["reco_cut"].array()



    # print(f"pflow number of particles each subjet={ak.num(fjet1_pflows_eta)}")
    # print(f"pflow len={len(fjet1_pflows_eta[1])}")
    # print( jets_flavor_truth[0] )


    # build masks
    # reco mask
    reco_mask = (reco_cut == 1)

    # select evts with more than 2b
    count_of_b = ak.sum(jets_flavor_truth == 5, axis=1)
    # count u-type quark in evt
    count_of_utype = ak.sum( (jets_flavor_truth == 2) |  (jets_flavor_truth == 4), axis=1)
    # count d-type quark in evt
    count_of_btype = ak.sum((jets_flavor_truth == 1) |  (jets_flavor_truth == 3), axis=1)

    flavor_mask = ((count_of_b >= 2) & (count_of_utype >= 1) & (count_of_btype >= 1))

    # particle number mismatch mask
    n_parts_mask = ak.num(fjet1_pflows_pt) == ak.num(fjet1_pflows_eta)

    # Total MASK
    total_mask = (reco_mask & flavor_mask & n_parts_mask)
  


    # Apply masks
    jets_eta = jets_eta[total_mask]
    jets_phi = jets_phi[total_mask]
    jets_pt = jets_pt[total_mask]
    jets_mass = jets_mass[total_mask]
    jets_flavor_truth = jets_flavor_truth[total_mask]

    fjet1_pflows_eta = fjet1_pflows_eta[total_mask]
    fjet1_pflows_phi = fjet1_pflows_phi[total_mask]
    fjet1_pflows_pt = fjet1_pflows_pt[total_mask]
    fjet1_pflows_mass = fjet1_pflows_mass[total_mask]
    fjet1_pflows_charge = fjet1_pflows_charge[total_mask]
    fjet1_pflows_pid = fjet1_pflows_pid[total_mask]

    # fjet_pt = fjet_pt[total_mask]
    # fjet_eta = fjet_eta[total_mask]
    # fjet_phi = fjet_phi[total_mask]
    # fjet_mass = fjet_mass[total_mask]

    top_eta = top_eta[total_mask]
    top_phi = top_phi[total_mask]
    anti_top_eta = anti_top_eta[total_mask]
    anti_top_phi = anti_top_phi[total_mask]
    charge = charge[total_mask]


    print( f"flavor len = {len(jets_flavor_truth)}" )
    print( f"pflow_eta len = {len(fjet1_pflows_eta)}" )
    print( f"top len = {len(top_eta)}" )
    print( f"antitop len = {len(anti_top_eta)}" )
    print(top_eta)
    print(anti_top_eta)


    # create empty 3 layer list to store the particle info
    particles_pt = []
    particles_eta = []
    particles_phi = []
    particles_mass = []
    particles_pid = []
    particles_charge = []

    # calculate delta_R to correspond particles to subjets
    print("calculating delta_R")
    for evt in range(len(jets_eta)):
        event_particles_pt = []
        event_particles_eta = []
        event_particles_phi = []
        event_particles_mass = []
        event_particles_pid = []
        event_particles_charge = []

        for subjet in range(len(jets_eta[evt])):
            subjet_LorentzVec = TLorentzVector()
            subjet_LorentzVec.SetPtEtaPhiM(jets_pt[evt][subjet], jets_eta[evt][subjet], jets_phi[evt][subjet], jets_mass[evt][subjet])

            subjet_particles_pt = []
            subjet_particles_eta = []
            subjet_particles_phi = []
            subjet_particles_mass = []
            subjet_particles_pid = []
            subjet_particles_charge = []

            # skip the events with not correct particle numbers
            if ( len(fjet1_pflows_pt[evt]) != len(fjet1_pflows_eta[evt]) ):
                print("n_particle mismatch")
                continue
            
            for particle in range(len(fjet1_pflows_eta[evt])):
                particle_LorentzVec = TLorentzVector()
                particle_LorentzVec.SetPtEtaPhiM(fjet1_pflows_pt[evt][particle], fjet1_pflows_eta[evt][particle], fjet1_pflows_phi[evt][particle], fjet1_pflows_mass[evt][particle])
                delta_R = subjet_LorentzVec.DeltaR(particle_LorentzVec)

                # save the particles with delta_R<=0.4
                if delta_R<=0.4:
                    subjet_particles_pt.append(fjet1_pflows_pt[evt][particle])
                    subjet_particles_eta.append(fjet1_pflows_eta[evt][particle])
                    subjet_particles_phi.append(fjet1_pflows_phi[evt][particle])
                    subjet_particles_mass.append(fjet1_pflows_mass[evt][particle])
                    subjet_particles_pid.append(fjet1_pflows_pid[evt][particle])
                    subjet_particles_charge.append(fjet1_pflows_charge[evt][particle])

            event_particles_pt.append(subjet_particles_pt)
            event_particles_eta.append(subjet_particles_eta)
            event_particles_phi.append(subjet_particles_phi)
            event_particles_mass.append(subjet_particles_mass)
            event_particles_pid.append(subjet_particles_pid)
            event_particles_charge.append(subjet_particles_charge)

        particles_pt.append(event_particles_pt)
        particles_eta.append(event_particles_eta)
        particles_phi.append(event_particles_phi)
        particles_mass.append(event_particles_mass)
        particles_pid.append(event_particles_pid)
        particles_charge.append(event_particles_charge)
                    
    # Convert the lists to Awkward Arrays
    particles_pt_array = ak.Array(particles_pt)
    particles_eta_array = ak.Array(particles_eta)
    particles_phi_array = ak.Array(particles_phi)
    particles_mass_array = ak.Array(particles_mass) 
    particles_pid_array = ak.Array(particles_pid)
    particles_charge_array = ak.Array(particles_charge)

    n_events = len(particles_pt_array)
    n_subjets_max = ak.max( ak.num(particles_pt_array, axis=1) )
    n_subjets_total = ak.sum( ak.num(particles_pt_array, axis=1) )
    n_particles = ak.max( ak.num(particles_pt_array, axis=2) )
    n_features = 12

    print(f"n_events = {n_events}")
    print(f"n_subjets_total = {n_subjets_total}")
    print(f"n_particles = {n_particles}")

    
    # create points, jets(subjets), y

    ## points has particle info. the dim is (n, p, d), 
    ## where n: number of subjets, p:numbr of particles inside subjet, d: number of features
    points = np.zeros((n_subjets_total, n_particles, n_features))
    
    ### make the features of particles
    print("Making particle data features")

    #### calculate delta_eta, delta_phi btween particle with top jet and subjet
    delta_eta_t_event_list = [] # each element is an event
    delta_phi_t_event_list = []
    delta_eta_j_event_list = []
    delta_phi_j_event_list = []

    for evt, lepton_charge in enumerate(charge):
        delta_eta_t_subjet_list = [] # each element is a subjet
        delta_phi_t_subjet_list = []
        delta_eta_j_subjet_list = []
        delta_phi_j_subjet_list = []

        # if lepton_charge>0, anti_top hadronic
        if lepton_charge > 0:
            t_eta = anti_top_eta[evt]
            t_phi = anti_top_phi[evt]
        else:
            t_eta = top_eta[evt]
            t_phi = top_phi[evt]

        for subjet in range(len(particles_eta_array[evt])):
            
            if particles_eta_array[evt][subjet] is None:
                break

            delta_eta_t_list = []
            delta_phi_t_list = []
            delta_eta_j_list = []
            delta_phi_j_list = []
            
            for i in range(len(particles_eta_array[evt][subjet])):
                if particles_eta_array[evt][subjet][i] is None:
                    break

                delta_eta_t = particles_eta_array[evt][subjet][i] - t_eta
                delta_phi_t = particles_phi_array[evt][subjet][i] - t_phi
                delta_eta_j = particles_eta_array[evt][subjet][i] - jets_eta[evt][subjet]
                delta_phi_j = particles_phi_array[evt][subjet][i] - jets_phi[evt][subjet]
                
                delta_eta_t_list.append(delta_eta_t)
                delta_phi_t_list.append(delta_phi_t)
                delta_eta_j_list.append(delta_eta_j)
                delta_phi_j_list.append(delta_phi_j)

            delta_eta_t_subjet_list.append(delta_eta_t_list)
            delta_phi_t_subjet_list.append(delta_phi_t_list)
            delta_eta_j_subjet_list.append(delta_eta_j_list)
            delta_phi_j_subjet_list.append(delta_phi_j_list)

        delta_eta_t_event_list.append(delta_eta_t_subjet_list)
        delta_phi_t_event_list.append(delta_phi_t_subjet_list)
        delta_eta_j_event_list.append(delta_eta_j_subjet_list)
        delta_phi_j_event_list.append(delta_phi_j_subjet_list)


    # convert list to ak.array
    delta_eta_t_event_array = ak.Array(delta_eta_t_event_list)
    delta_phi_t_event_array = ak.Array(delta_phi_t_event_list)
    delta_eta_j_event_array = ak.Array(delta_eta_j_event_list)
    delta_phi_j_event_array = ak.Array(delta_phi_j_event_list)

    # pad all subjets so they have same num of particles
    delta_eta_t_event_array = ak.pad_none(delta_eta_t_event_array, target=n_particles, axis=2)
    delta_phi_t_event_array = ak.pad_none(delta_phi_t_event_array, target=n_particles, axis=2)
    delta_eta_j_event_array = ak.pad_none(delta_eta_j_event_array, target=n_particles, axis=2)
    delta_phi_j_event_array = ak.pad_none(delta_phi_j_event_array, target=n_particles, axis=2)

    delta_eta_t_event_array = ak.fill_none(delta_eta_t_event_array, 0)
    delta_phi_t_event_array = ak.fill_none(delta_phi_t_event_array, 0)
    delta_eta_j_event_array = ak.fill_none(delta_eta_j_event_array, 0)
    delta_phi_j_event_array = ak.fill_none(delta_phi_j_event_array, 0)

    # # pad events, so all events have same num of subjets
    # delta_eta_t_event_array = pad_array(delta_eta_t_event_array, n_events, n_subjets, n_particles)
    # delta_phi_t_event_array = pad_array(delta_phi_t_event_array, n_events, n_subjets, n_particles)
    # delta_eta_j_event_array = pad_array(delta_eta_j_event_array, n_events, n_subjets, n_particles)
    # delta_phi_j_event_array = pad_array(delta_phi_j_event_array, n_events, n_subjets, n_particles)

    # flatten so there is no events, just subjets
    delta_eta_t_event_array = ak.flatten(delta_eta_t_event_array)
    delta_phi_t_event_array = ak.flatten(delta_phi_t_event_array)
    delta_eta_j_event_array = ak.flatten(delta_eta_j_event_array)
    delta_phi_j_event_array = ak.flatten(delta_phi_j_event_array)

    print(ak.num(delta_eta_t_event_array, axis=0))
    print(ak.num(delta_eta_t_event_array, axis=1))
    
    print(delta_eta_t_event_array[0])
    print(delta_eta_t_event_array[0][0])

    
    
    points[:,:,0] = ak.to_numpy( delta_eta_t_event_array )
    points[:,:,1] = ak.to_numpy( delta_phi_t_event_array )
    points[:,:,2] = ak.to_numpy( delta_eta_j_event_array )
    points[:,:,3] = ak.to_numpy( delta_phi_j_event_array )


    #### log(pt)
    particles_pt_array = ak.pad_none(particles_pt_array, target=n_particles, axis=2)
    particles_pt_array = ak.fill_none(particles_pt_array, 0)
    particles_pt_array = ak.flatten(particles_pt_array)

    points[:,:,4] = ak.to_numpy( particles_pt_array )

    
    #### log(E). E = sqrt(pt^2+m^2) * cosh(eta)
    particles_eta_array = ak.pad_none(particles_eta_array, target=n_particles, axis=2)
    particles_eta_array = ak.fill_none(particles_eta_array, 0)
    particles_eta_array = ak.flatten(particles_eta_array)

    log_E_array = particles_pt_array * np.cosh(particles_eta_array)

    points[:,:,5] = ak.to_numpy( log_E_array )
  

    #### charge of particle
    particles_charge_array = ak.pad_none(particles_charge_array, target=n_particles, axis=2)
    particles_charge_array = ak.fill_none(particles_charge_array, 0)
    particles_charge_array = ak.flatten(particles_charge_array)

    points[:,:,6] = ak.to_numpy( particles_charge_array ) 

    print(ak.num(particles_charge_array, axis=0))
    print(ak.num(particles_charge_array, axis=1))
    # print(particles_eta_array[3])
    # print(particles_charge_array[3])
    

    #### is_electron, is_muon, is_photon, is_ChargedHadron, is_NeutralHadron
    particles_pid_array = ak.pad_none(particles_pid_array, target=n_particles, axis=2)
    particles_pid_array = ak.fill_none(particles_pid_array, 0)
    particles_pid_array = ak.flatten(particles_pid_array)
    
    is_electron_array = ak.where(particles_pid_array == 11, 1, 0)
    is_muon_array = ak.where(particles_pid_array == 13, 1, 0)
    is_photon_array = ak.where(particles_pid_array == 22, 1, 0)
    
    hadron_array = ak.where(particles_pid_array > 37, 1, 0)
    is_ChargedHadron_array = ak.where((hadron_array == 1)&(particles_charge_array != 0), 1, 0)
    is_NeutralHadron_array = ak.where((hadron_array == 1)&(particles_charge_array == 0), 1, 0)

    points[:,:,7] = ak.to_numpy( is_electron_array ) 
    points[:,:,8] = ak.to_numpy( is_muon_array ) 
    points[:,:,9] = ak.to_numpy( is_photon_array ) 
    points[:,:,10] = ak.to_numpy( is_ChargedHadron_array ) 
    points[:,:,11] = ak.to_numpy( is_NeutralHadron_array ) 

    
    print(f"points[20][0][:] = {points[20][0][:]}")
    

    ### flatten the evt level direction of points
    # points = points.reshape(-1, *points.shape[2:])
    print(points)
    print(f"points shape: {points.shape}")
    

    ## jets is a 2d (n_subjets,4) array with the 4 columns are jets_pt, jets_eta, jets_phi, jets_m
    jets = np.stack( [ak.flatten(jets_pt), ak.flatten(jets_eta), ak.flatten(jets_phi), ak.flatten(jets_mass)] ,-1)
    jets = ak.to_numpy(jets)
    print(f"jets shape: {jets.shape}")

    ## y has the subjet pid
    y = ak.flatten(jets_flavor_truth)
    y = ak.to_numpy(y)
    print(f"y shape: {y.shape}")


    # keeping only subjets of u,c / d,s
    print("keeping only u,c / d,s ...")
    ucds_mask = (y==1)|(y==2)|(y==3)|(y==4)
    points = points[ucds_mask]
    jets = jets[ucds_mask]
    y = y[ucds_mask]
    ## set down-type jet to 0, uptype to 1
    y[(y==1)|(y==3)] = 0
    y[(y==2)|(y==4)] = 1
    
    print(f"points shape: {points.shape}")
    print(f"jets shape: {jets.shape}")
    print(f"y shape: {y.shape}")

    # balance data
    print("balancing data ...")
    points, y,jets = balance_classes(points, y,jets)

    print(f"points shape: {points.shape}")
    print(f"jets shape: {jets.shape}")
    print(f"y shape: {y.shape}")


    # output as cms form. jet: subjet info, data: particles info, pid: subjet pid
    train_nevt = int(0.7*jets.shape[0])
    val_nevt = train_nevt + int(0.2*jets.shape[0])
    with h5py.File('{}/train_qgcms_pid.h5'.format(output_path), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[:train_nevt])
        dset = fh5.create_dataset('jet', data= jets[:train_nevt])
        dset = fh5.create_dataset('pid', data=y[:train_nevt])

    with h5py.File('{}/val_qgcms_pid.h5'.format(output_path), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[train_nevt:val_nevt])
        dset = fh5.create_dataset('jet', data=jets[train_nevt:val_nevt])
        dset = fh5.create_dataset('pid', data=y[train_nevt:val_nevt])

    with h5py.File('{}/test_qgcms_pid.h5'.format(output_path), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[val_nevt:])
        dset = fh5.create_dataset('jet', data=jets[val_nevt:])
        dset = fh5.create_dataset('pid', data=y[val_nevt:])




    


if __name__ == '__main__':

    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/QGCMS', help="Folder containing input files")
    (flags, args) = parser.parse_args()

    preprocess()