import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import awkward as ak

def plot_hist(c_pt, s_pt, feature_name, rank, AorY):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,gridspec_kw={"height_ratios": (3, 1)})
    
    # remove nan values
    c_pt = c_pt[~np.isnan(c_pt)]
    s_pt = s_pt[~np.isnan(s_pt)]

    
    # Calculate the minimum and maximum values for setting the x-axis range
    min_value = min(np.min(c_pt), np.min(s_pt))
    max_value = max(np.max(c_pt), np.max(s_pt))

    # Define bin edges and centers
    num_bins = 50
    bins = np.linspace(min_value, max_value, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Calculate histograms for both datasets
    counts_c, _ = np.histogram(c_pt, bins=bins)
    counts_s, _ = np.histogram(s_pt, bins=bins)           

    # Calculate the ratio and its uncertainty
    ratio = counts_c / counts_s
    uncertainty_ratio = ratio * np.sqrt((1 / counts_c) + (1 / counts_s))

    # Upper panel: Histograms of C and S
    ax1.hist(bin_centers, bins=bins, weights=counts_c, alpha=0.5, label='up-type jets', color='blue', density=True, log=False, histtype='step')
    ax1.hist(bin_centers, bins=bins, weights=counts_s, alpha=0.5, label='down-type jets', color='hotpink', density=True, log=False, histtype='step')
    ax1.set_title(f'{AorY} Data Jets')
    ax1.set_xlabel(f'{feature_name}')
    ax1.set_ylabel('Normalized Events (log)')
    ax1.legend()
    ax1.grid(True)

    # Lower panel: Ratio plot with uncertainties
    ax2.errorbar(bin_centers, ratio, yerr=uncertainty_ratio, fmt='o', color='black', label='up-type/down-type')
    ax2.axhline(1, color='gray', linestyle='--')  # Reference line at ratio=1
    # ax2.set_xlabel('Bin')
    ax2.set_ylabel('up-type/down-type')
    ax2.legend()
    ax2.grid(True)

    # Construct the base file path
    filepath = f"/global/homes/w/weipow/xgboost/{AorY}_hist/{rank}_{feature_name}.jpg"

    # # If file exists, increment the counter until a unique name is found
    # filepath = base_filepath
    # counter = 1
    # while os.path.exists(filepath):
    #     filepath = f"/global/homes/w/weipow/xgboost/{AorY}_hist/{feature_name}_{counter}.jpg"
    #     counter += 1

    # Save the figure
    plt.savefig(filepath, dpi=300)
    plt.close()

# find feature name for given feature index
def find_feature_name(feat_index):

    feat_name_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_m', 'part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel', 
                        'part_logptrel', 'part_logerel',
                        'part_charge', 'hypot_partpt_parteta',
                        'part_pionkaonproton', 'part_kaonneutronundefine',
                            'part_photon', 'part_electron',
                            'part_muon']
    
    if feat_index <= 3:
        feat_name = feat_index
    else:
        feat_name = int(np.round((feat_index - 4) / 40) + 4)
    
    # else: # Yulei
    #     feat_name_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_m=0', 'delta_eta', 'delta_phi', 'log_partpt_jetpt', 'log_partpt', 
    #                             'log_parte_jete', 'log_parte', 
    #                             'hypot_partpt_parteta', 'part_charge',
    #                             'part_pionkaonproton', 'part_kaonneutronundefine',
    #                             'part_photon', 'part_electron',
    #                             'part_muon']
        
    #     if feat_index >= 3:
    #         feat_name = (feat_index-3) % 15
    #     else: 
    #         feat_name = feat_index

    

    return feat_name_list[feat_name]


if __name__=='__main__':

    # Load data
    AlbertoORYulei = "Alberto"
    debug = False

    if debug == True:
        data_2d = np.load(f"/global/homes/w/weipow/xgboost/{AlbertoORYulei}_data_X.npy")[0:50]
        labels = np.load(f"/global/homes/w/weipow/xgboost/{AlbertoORYulei}_data_Y.npy")[0:50]
    else:
        data_2d = np.load(f"/global/homes/w/weipow/xgboost/{AlbertoORYulei}_data_X.npy")
        labels = np.load(f"/global/homes/w/weipow/xgboost/{AlbertoORYulei}_data_Y.npy")


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_2d, labels, test_size=0.2, random_state=42)

    # print(X_test[0])
    # print(X_test[1])
    # print(X_test[2])
    # print(X_test[3])
    # raise

    # Create and train the XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state = 43)
    model.fit(X_train, y_train)

    # Feature Importance
    importance = model.feature_importances_

    # print and plot features
    print(f"importance shape: {importance.shape}")
    topten_index = ak.argsort(importance, ascending=False)[0:10]
    print(f"top ten index: {topten_index}")
    print("top ten feature name:")
    for r, i in enumerate(topten_index):
        feat_name = find_feature_name(i)
        print(feat_name)

        up_data = data_2d[:,i][labels==0]
        down_data = data_2d[:,i][labels==1]

        print(data_2d.shape)
        print(up_data)
        print(down_data)
        print(f"c_pt has NaNs: {np.isnan(up_data).any()}, NaN count: {np.isnan(up_data).sum()}")
        print(f"s_pt has NaNs: {np.isnan(down_data).any()}, NaN count: {np.isnan(down_data).sum()}")

        # make nan values to zero
        # up_data = np.nan_to_num(up_data, nan=0.0)
        # down_data = np.nan_to_num(down_data, nan=0.0)

        plot_hist(up_data, down_data, feature_name=feat_name, rank=r, AorY=AlbertoORYulei)


    

    # Plotting Feature Importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title(f'{AlbertoORYulei} Feature Importance')
    plt.savefig("/global/homes/w/weipow/xgboost/feature_importance.jpg", dpi=300)

    # Predict probabilities for the test data
    y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities of the positive class

    # Calculate accuracy
    y_pred = model.predict(X_test)
    print(y_probs[0:20])
    print(y_pred[0:20])
    print(y_test[0:20])
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{AlbertoORYulei} Data ')
    plt.legend(loc="lower right")
    plt.savefig("/global/homes/w/weipow/xgboost/0827.jpg", dpi=300)
    plt.close()

    # plot xgboost output distribution
    # Predict probabilities for the training and test data
    y_train_probs = model.predict_proba(X_train)[:, 1]  # Positive class probabilities
    y_test_probs = model.predict_proba(X_test)[:, 1]  # Positive class probabilities

    # Plotting the distribution of predictions for positive and negative classes in train and test sets
    plt.figure(figsize=(10, 6))

    # Plot training positive class
    plt.hist(y_train_probs[y_train == 1], bins=50, alpha=0.5, label='Train Positive Class', color='navy', density=True)

    # Plot training negative class
    plt.hist(y_train_probs[y_train == 0], bins=50, alpha=0.5, label='Train Negative Class', color='maroon', density=True)

    # Plot test positive class
    plt.hist(y_test_probs[y_test == 1], bins=50, alpha=0.5, label='Test Positive Class', color='lime', density=True)

    # Plot test negative class
    plt.hist(y_test_probs[y_test == 0], bins=50, alpha=0.5, label='Test Negative Class', color='orange', density=True)

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'{AlbertoORYulei} Prediction Distribution by Class')
    plt.legend(loc='upper right')

    # Save the plot
    plt.savefig(f"/global/homes/w/weipow/xgboost/{AlbertoORYulei}_xgb_prediction_distribution.jpg", dpi=300)