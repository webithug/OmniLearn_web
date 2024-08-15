import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import plot_utils
import uproot
import awkward as ak

plot_utils.SetStyle()


def parse_options():
    """Parse command line options."""
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset", type=str, default="top", help="Folder containing input files")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--plot_folder", type=str, default="../plots", help="Folder to save the outputs")
    parser.add_argument("--n_bins", type=int, default=50, help="Number of bins for the histograms")
    return parser.parse_args()

def load_data(flags):
    """Load data based on dataset using specified data loaders and file naming conventions."""
    if flags.dataset == 'top':
        test = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'test_ttbar.h5'))

    if flags.dataset == 'tau':
        test = utils.TauDataLoader(os.path.join(flags.folder,'TAU', 'test_tau.h5'))
    elif flags.dataset == 'qg':
        test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'test_qg.h5'))
        
    elif flags.dataset == 'cms':
        test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'))

    elif flags.dataset == 'qibin':
        test = utils.QibinDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'))
        
    elif flags.dataset == 'jetnet150':
        test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_150.h5'),big=True)
        
    elif flags.dataset == 'eic':
        test = utils.EicPythiaDataLoader(os.path.join(flags.folder,'EIC_Pythia','val_eic.h5'))
    elif flags.dataset == 'jetnet30':
        test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_30.h5'))
        
    elif flags.dataset == 'jetclass':
        test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','test'))
        
    elif flags.dataset == 'h1':
        test = utils.H1DataLoader(os.path.join(flags.folder,'H1','val.h5'))
        
    elif flags.dataset == 'atlas':
        test = utils.H1DataLoader(os.path.join(flags.folder,'ATLASTOP','val_atlas.h5'))
        
    elif flags.dataset == 'omnifold':
        test = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','test_pythia.h5'))
    elif flags.dataset == 'lhco':
        test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO', 'val_background_SB.h5'))        
    else:
        raise ValueError("Unknown dataset specified or file name not provided")

    return test

def process_particles(test):
    """Process particles and jets from the test dataset."""
    parts, jets = [], []
    label = []
    for i, file_name in enumerate(test.files):
        if i > 40:
            break
        X, y = test.data_from_file(file_name)
        parts.append(X[0])
        jets.append(X[3])
        label.append(y)
        del X
    return np.concatenate(parts), np.concatenate(jets), np.concatenate(label)

def read_root_file(file_path, branch_names):
    file = uproot.open(file_path)

    tree = file["t"]

    truth_is_c = tree["truth_is_c"].array()
    truth_is_c = ak.to_numpy(truth_is_c)

    output = tree[branch_names].array()
    output = ak.to_numpy(output)

    output = output[truth_is_c]

    return output

def main():
    flags = parse_options()
    plot_utils.SetStyle()
    
    test = load_data(flags)

    parts, jets, label = process_particles(test)
    print('number of events',parts.shape[0])
    print("number of particles", parts.shape[1])
    print('particles mean',np.mean(parts,(0,1)))
    print('particles std',np.std(parts,(0,1)))
    
    print('jets mean',np.mean(jets,0))
    print('jets std',np.std(jets,0))

    print(f"label shape = {label.shape}")
    print(f"first ten label = {label[0:10]}")

    # make jet info histograms: comparing cms data with qibin data 0807
    for feat in range(len(test.jet_names)):
        flat = jets[:, feat]
        # fig, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat}, test.jet_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
        # fig.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_{feat}.pdf", bbox_inches='tight')

        # make pt histogram for c:1 and s:0
        if feat == 0:
            # prepare cms open data
            c_pt = flat[label==1]
            s_pt = flat[label==0]
            print(f"c_pt shape = {c_pt.shape}")
            print(f"s_pt shape = {s_pt.shape}")

            # prepare qibin data
            # data_qibin = np.load('/global/homes/w/weipow/qibin_cs.npz')
            # c_pt_qibin = data_qibin['cjet_pt']
            # s_pt_qibin = data_qibin['sjet_pt']

            c_pt_qibin = read_root_file("/pscratch/sd/w/weipow/QibinData/v5_weber_udjet_c_20M.root", "truth_ujet_pt")
            s_pt_qibin = read_root_file("/pscratch/sd/w/weipow/QibinData/v5_weber_udjet_c_20M.root", "truth_djet_pt")
    
            
            # CMS data
            # Create figure with two panels (upper and lower)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                               gridspec_kw={"height_ratios": (3, 1)})
            
            # Define bin edges and centers
            num_bins = 100  # Adjust this value based on your data and preference
            # Calculate histograms for both datasets
            counts_c, bins = np.histogram(c_pt, bins=num_bins)
            counts_s, _ = np.histogram(s_pt, bins=bins)            
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # Calculate the ratio and its uncertainty
            ratio = counts_c / counts_s
            uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

            # Upper panel: Histograms of C and S
            ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label='up-type jets', color='blue', density=True, log=True)
            ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label='down-type jets', color='hotpink', density=True, log=True)
            ax1.set_title('Qibin Data Jets')
            ax1.set_xlabel('Jet pt [GeV]')
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

            plt.savefig(f"{flags.plot_folder}/jets_pt_cms_new.jpg", dpi=300)
            plt.close()


            
            # # Qibin data 0807
            # # Create figure with two panels (upper and lower)
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
            #                    gridspec_kw={"height_ratios": (3, 1)})
            
            # # Define bin edges and centers
            # num_bins = 100  # Adjust this value based on your data and preference
            # # Calculate histograms for both datasets
            # counts_c, bins = np.histogram(c_pt_qibin, bins=num_bins)
            # counts_s, _ = np.histogram(s_pt_qibin, bins=bins)            
            # bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # # Calculate the ratio and its uncertainty
            # ratio = counts_c / counts_s
            # uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

            # # Upper panel: Histograms of C and S
            # ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label='C (Charm)', color='blue', density=True, log=True)
            # ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label='S (Strange)', color='hotpink', density=True, log=True)
            # ax1.set_title('Qibin Jets')
            # ax1.set_xlabel('Jet pt [GeV]')
            # ax1.set_ylabel('Normalized Events (log)')
            # ax1.legend()
            # ax1.grid(True)

            # # Lower panel: Ratio plot with uncertainties
            # ax2.errorbar(bin_centers, ratio, yerr=uncertainty_ratio, fmt='o', color='black', label='C/S')
            # ax2.axhline(1, color='gray', linestyle='--')  # Reference line at ratio=1
            # # ax2.set_xlabel('Bin')
            # ax2.set_ylabel('C/S')
            # ax2.legend()
            # ax2.grid(True)

            # plt.savefig(f"{flags.plot_folder}/jets_pt_qibin_new.jpg", dpi=300)
            # plt.close()

        # make eta histogram for c:1 and s:0
        elif feat == 1:

            # prepare cms open data
            c_eta = flat[label==1]
            s_eta = flat[label==0]
            print(f"c_pt shape = {c_eta.shape}")
            print(f"s_pt shape = {s_eta.shape}")

            # prepare qibin data
            c_eta_qibin = read_root_file("/pscratch/sd/w/weipow/QibinData/v5_weber_udjet_c_20M.root", "truth_ujet_eta")
            s_eta_qibin = read_root_file("/pscratch/sd/w/weipow/QibinData/v5_weber_udjet_c_20M.root", "truth_djet_eta")

    
            
            # CMS data
            # Create figure with two panels (upper and lower)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                               gridspec_kw={"height_ratios": (3, 1)})
            
            # Define bin edges and centers
            num_bins = 100  # Adjust this value based on your data and preference
            # Calculate histograms for both datasets
            counts_c, bins = np.histogram(c_eta, bins=num_bins)
            counts_s, _ = np.histogram(s_eta, bins=bins)            
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # Calculate the ratio and its uncertainty
            ratio = counts_c / counts_s
            uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

            # Upper panel: Histograms of C and S
            ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label='C (Charm)', color='blue', density=True, log=True)
            ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label='S (Strange)', color='hotpink', density=True, log=True)
            ax1.set_title('Qibin Data Jets')
            ax1.set_xlabel('eta')
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

            plt.savefig(f"{flags.plot_folder}/jets_eta_cms_new.jpg", dpi=300)
            plt.close()


            
            # # Qibin data 0807
            # # Create figure with two panels (upper and lower)
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
            #                    gridspec_kw={"height_ratios": (3, 1)})
            
            # # Define bin edges and centers
            # num_bins = 100  # Adjust this value based on your data and preference
            # # Calculate histograms for both datasets
            # counts_c, bins = np.histogram(c_eta_qibin, bins=num_bins)
            # counts_s, _ = np.histogram(s_eta_qibin, bins=bins)            
            # bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # # Calculate the ratio and its uncertainty
            # ratio = counts_c / counts_s
            # uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

            # # Upper panel: Histograms of C and S
            # ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label='C (Charm)', color='blue', density=True, log=True)
            # ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label='S (Strange)', color='hotpink', density=True, log=True)
            # ax1.set_title('Qibin Jets')
            # ax1.set_xlabel('eta')
            # ax1.set_ylabel('Normalized Events (log)')
            # ax1.legend()
            # ax1.grid(True)

            # # Lower panel: Ratio plot with uncertainties
            # ax2.errorbar(bin_centers, ratio, yerr=uncertainty_ratio, fmt='o', color='black', label='C/S')
            # ax2.axhline(1, color='gray', linestyle='--')  # Reference line at ratio=1
            # # ax2.set_xlabel('Bin')
            # ax2.set_ylabel('C/S')
            # ax2.legend()
            # ax2.grid(True)

            # plt.savefig(f"{flags.plot_folder}/jets_eta_qibin_new.jpg", dpi=300)
            # plt.close()





            # # Create the normalized histograms 
            # plt.figure(figsize=(10, 6))
            # plt.hist(s_pt_qibin, bins=30, alpha=0.5, label='strange', color='hotpink', density=True, log=True)
            # plt.hist(c_pt_qibin, bins=30, alpha=0.5, label='charm', color='blue', density=True, log=True)

            # plt.title('Qibin Jets')
            # plt.xlabel('Jet pt [GeV]')
            # plt.ylabel('Normalized Events')
            # plt.legend(loc='upper right')

            # plt.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_qibin.pdf")
            # plt.close()

            # # Create the normalized histograms for strange
            # plt.figure(figsize=(10, 6))
            # plt.hist(s_pt, bins=30, alpha=0.5, label='strange', color='hotpink', density=True, log=True)
            # plt.hist(c_pt, bins=30, alpha=0.5, label='charm', color='blue', density=True, log=True)

            # plt.title('CMS Open Data Jets')
            # plt.xlabel('Jet pt [GeV]')
            # plt.ylabel('Normalized Events')
            # plt.legend(loc='upper right')

            # plt.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_cms.pdf")
            # plt.close()

            # fig_c, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): c_pt}, "Charm " + test.jet_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
            # fig_c.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_c.pdf", bbox_inches='tight')

            # fig_s, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): s_pt}, "Strange " + test.jet_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
            # fig_s.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_s.pdf", bbox_inches='tight')





    print("Maximum number of particles",np.max(np.sum(parts[:, :, 0]!=0,1)))
    mask = parts[:, :, 0].reshape(-1) != 0

    # # make particle info histograms
    # for feat in range(len(test.part_names)):
    #     flat = parts[:, :, feat].reshape(-1)
    #     flat = flat[mask]
    #     fig, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat}, test.part_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
    #     fig.savefig(f"{flags.plot_folder}/parts_{flags.dataset}_{feat}.pdf", bbox_inches='tight')



if __name__ == "__main__":
    main()
