import pandas as pd
import h5py
import os, gc
import numpy as np
from optparse import OptionParser
import energyflow as ef
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    ax1.set_title('CMS Data Jets')
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

    plt.savefig(f"/global/homes/w/weipow/My_omnilearn_output/0819_plot_jetcharge/cms_jetcharge.jpg", dpi=300)
    plt.close()

def pad_and_combine(arrays,M):
    F = arrays[0].shape[1] -1 #skip vertex information
    
    # Initialize the list to hold padded arrays
    padded_arrays = np.zeros((len(arrays),M,F))
    
    # Pad each subarray if necessary and append to padded_arrays
    for iarr, array in tqdm(enumerate(arrays), total = len(arrays), desc="Processing arrays"):        
        P = array.shape[0]        
        # Check if padding is needed (if P < M)
        if P < M:
            padded_arrays[iarr,:P] += array[:,:-1]
        else:
            padded_arrays[iarr] += array[:M,:-1]
    
    return  padded_arrays


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


#Preprocessing for the top tagging dataset
def preprocess(data,folder,nparts=100, use_pid = True):
    print("Creating labels")

    # y = data.jets_i[:,-1] # y is the hard_pid (1d array)

    # y[(np.abs(y)==1)|(np.abs(y)==2)|(np.abs(y)==3)] = 1 #uds all turn into 1
    # y[y==21] = 0

    # print(f"y = {y.shape}")
    # print(f"data.particles = {data.particles.shape}")

    # # jets is a 2d (n,4) array with the 4 columns are jet_pt, jet_eta, jet_phi, jet_m
    # jets = np.stack([data.jets_f[:,0],data.jets_f[:,4],data.jets_f[:,2],data.jets_f[:,3]],-1)  
    # jets = jets[np.abs(y)<2]
    # particles = data.particles[np.abs(y)<2]
    # y = y[np.abs(y) < 2] #Reject c and b jets



    # y is the hard_pid (1d array). This would be the label of data    
    y = data.jets_i[:,-1] 

    # jets is a 2d (n,4) array with the 4 columns are jet_pt, jet_eta, jet_phi, jet_m
    jets = np.stack([data.jets_f[:,0],data.jets_f[:,4],data.jets_f[:,2],data.jets_f[:,3]],-1) 

    # keep only s=3 and c=4 jets
    jets = jets[np.logical_or(np.abs(y) == 3, np.abs(y) == 4)]
    # particles is a list of 2d array. the length of this list is same as the number of jets. 
    # each entry of list is an 2d array of size (x, 6), where x is the number of particles in the jet. see /pfcs for the 6 info.
    particles = data.particles[np.logical_or(np.abs(y) == 3, np.abs(y) == 4)] 

    y = y[np.logical_or(np.abs(y) == 3, np.abs(y) == 4)]

    # make s jets 0, c jets 1
    y[np.abs(y) == 3] = 0
    y[np.abs(y) == 4] = 1


    # raise Exception(f"particles.shape = {particles[0]}")



    #Min pt cut for particles
    # particles = np.asarray([part[ef.mod.filter_particles(part, pt_cut = 1)] for part in particles])

    # select particles with pt greater than 1
    particles = [part[ef.mod.filter_particles(part, pt_cut = 1)] for part in particles]


    print("Start preparing the dataset")
    del data
    gc.collect()

    particles = pad_and_combine(particles,nparts)

    print("Balancing classes")
    #Balance the number of signal and background events
    particles, y, jets = balance_classes(particles, y,jets)

    print("Total sample size after balancing: {}".format(particles.shape[0]))
    pid = particles[:,:,-1]

    # calculate jet energy E^2 = p^2 + m^2
    jet_e = np.sqrt(jets[:,0]**2*np.cosh(jets[:,1])**2 + jets[:,3]**2)
    mask = particles[:,:,0]>0    

    p_e = particles[:,:,0]*np.cosh(particles[:,:,1])*mask
    particles[:,:,3] = p_e


    if use_pid:
        NFEAT=13
    else:
        NFEAT=7
        
    points = np.zeros((particles.shape[0],nparts,NFEAT))

    delta_phi = particles[:,:,2] - jets[:,None,2]
    delta_phi[delta_phi>np.pi] -=  2*np.pi
    delta_phi[delta_phi<= - np.pi] +=  2*np.pi


    # These are the 'data' in the .h5 file. Eventually, 'data' and 'jet' are used as X of the dataset.
    points[:,:,0] = (particles[:,:,1] - jets[:,None,1]) # particle and jet delta_eta
    points[:,:,1] = delta_phi # particle and jet delta_phi
    points[:,:,2] = np.ma.log(1.0 - particles[:,:,0]/jets[:,None,0]).filled(0) # log(particle_pt / jet_pt)
    points[:,:,3] = np.ma.log(particles[:,:,0]).filled(0) # log( particle_pt )
    points[:,:,4] = np.ma.log(1.0 - particles[:,:,3]/jet_e[:,None]).filled(0) # log( 1 - particle_energy / jet_energy )
    points[:,:,5] = np.ma.log(particles[:,:,3]).filled(0) # log( energy )
    points[:,:,6] = np.hypot(points[:,:,0],points[:,:,1]) # sqrt( pt^2 + rapidity^2 )
    if use_pid:
        points[:,:,7] = np.sign(pid) * (pid!=22) * (pid!=130) # the charge of particle, excluding photon and kaons
        points[:,:,8] = (np.abs(pid) == 211) | (np.abs(pid) == 321) | (np.abs(pid) == 2212) # set to True for pions, kaons, or protons
        points[:,:,9] = (np.abs(pid)==130) | (np.abs(pid) == 2112) | (pid == 0) # True for neutral kaons, neutrons, or unidentified particles
        points[:,:,10] = np.abs(pid)==22 # True for photons
        points[:,:,11] = np.abs(pid)==11 # True for electrons
        points[:,:,12] = np.abs(pid)==13 # True for muons



    mult = np.sum(mask,-1)
    points*=mask[:,:nparts,None]

    #delete phi
    jets = np.delete(jets,2,axis=1)

    jets = np.concatenate([jets,mult[:,None]],-1)

    train_nevt = int(0.7*jets.shape[0])
    val_nevt = train_nevt + int(0.2*jets.shape[0])
    
    # Eventually, 'data' and 'jet' are used as X of the dataset.
    with h5py.File('{}/train_qgcms_pid.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[:train_nevt])
        dset = fh5.create_dataset('jet', data= jets[:train_nevt])
        dset = fh5.create_dataset('pid', data=y[:train_nevt])

    with h5py.File('{}/val_qgcms_pid.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[train_nevt:val_nevt])
        dset = fh5.create_dataset('jet', data=jets[train_nevt:val_nevt])
        dset = fh5.create_dataset('pid', data=y[train_nevt:val_nevt])

    with h5py.File('{}/test_qgcms_pid.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[val_nevt:])
        dset = fh5.create_dataset('jet', data=jets[val_nevt:])
        dset = fh5.create_dataset('pid', data=y[val_nevt:])


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/QGCMS', help="Folder containing input files")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    NPARTS = flags.npoints

    dataset = ef.mod.load(amount=1.,
                          cache_dir=flags.folder,
                          collection='CMS2011AJets', 
                          dataset='sim', 
                          subdatasets = {'SIM300_Jet300_pT375-infGeV'},
                        #   subdatasets={'SIM300_Jet300_pT375-infGeV', 'SIM1400_Jet300_pT375-infGeV', 'SIM800_Jet300_pT375-infGeV', 'SIM1000_Jet300_pT375-infGeV',
                        #                'SIM170_Jet300_pT375-infGeV', 'SIM470_Jet300_pT375-infGeV', 'SIM1800_Jet300_pT375-infGeV', 'SIM600_Jet300_pT375-infGeV'}, 
                          validate_files=False,
                          store_pfcs=True, store_gens=False, verbose=0)
    
    print(dataset)

    preprocess(dataset,samples_path)

