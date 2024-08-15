import uproot
import h5py
import numpy as np
import awkward as ak

# Open the ROOT file using uproot
file = uproot.open("/pscratch/sd/w/weipow/QibinData/Dr7v8.analysis_mtt800.10k.0.root")

print(file.keys())

tree = file["t"]

# print branch names
print("below are the branches:")
branch_names = tree.keys()
for branch in branch_names:
    print(branch)

# ujet_pt = tree["truth_ujet_pt"].array()
# ujet_pt = ak.to_numpy(ujet_pt)

# djet_pt = tree["truth_djet_pt"].array()
# djet_pt = ak.to_numpy(djet_pt)

# truth_is_c = tree["truth_is_c"].array()
# truth_is_c = ak.to_numpy(truth_is_c)

# print(f"ujet_pt  {ujet_pt}")
# print(f"truth_is_c  {truth_is_c}")

# cjet_pt = ujet_pt[truth_is_c]
# sjet_pt = djet_pt[truth_is_c]

# print(f"cjet_pt  {cjet_pt}")
# print(f"sjet_pt  {sjet_pt}")

# # Save arrays to a .npz file
# np.savez('qibin_cs.npz', cjet_pt=cjet_pt, sjet_pt=sjet_pt)