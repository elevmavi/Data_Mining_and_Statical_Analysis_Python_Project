import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.shared.data_processing import etl_mat_file


def cjkcalc(idxTest, cidx):
    # This is a placeholder implementation, replace it with your actual calculation
    # You can use any similarity/distance metric calculation here
    return np.sum(np.abs(idxTest - cidx))  # Example calculation, replace it with actual calculation


def fss(xV, initidx, thres, nclust):
    replic = 100
    CRItable = []
    idxTest = initidx
    index_cjk = 1
    fcols = xV.shape[1]

    for j in range(fcols):
        a = np.sum(xV[:, j])
        if np.isnan(a):
            CRItable.append([np.nan, j, j])
        else:
            cidx, _ = KMeans(n_clusters=nclust, n_init=replic, random_state=0).fit(
                xV[:, j].reshape(-1, 1)).labels_, None
            cri = cjkcalc(idxTest, cidx)
            CRItable.append([cri, j, j])

    crimax, mno = np.nanmax(CRItable, axis=0), np.nanargmax(CRItable, axis=0)[0]
    selmescap = CRItable[mno][1]
    feature1 = str(selmescap)
    cri_last = crimax

    while True:
        for ai in range(fcols):
            a = np.sum(xV[:, ai])
            if np.mean(xV[:, ai]) <= 10 ** -4 or np.isnan(a):
                CRItable.append([np.nan, ai, ai])
            else:
                kmeanscols = f"{selmescap} {ai}"
                fc = ai not in list(map(int, feature1.split()))
                if fc:
                    cidx, _ = KMeans(n_clusters=nclust, random_state=0).fit(
                        xV[:, list(map(int, kmeanscols.split()))]).labels_, None
                    cri = cjkcalc(idxTest, cidx)
                    CRItable.append([cri, ai, ai])

        crimax, mno = np.nanmax(CRItable, axis=0), np.nanargmax(CRItable, axis=0)[0]
        selmescap = CRItable[mno][1]
        q1 = (crimax - cri_last) / cri_last

        if np.any(q1 > thres) and np.all(q1 >= 0.0):
            feature1 += f", {selmescap}"
            cri_last = crimax

        if np.all(q1 < thres):
            break

    return feature1, cri_last


# Load data from MATLAB file
xV1 = etl_mat_file('resources/data/xV600x470.mat', 'xV1')

# Extract parameters
initidx = xV1.iloc[:, 0]
thres = 0.01
nclust = 3
xV = xV1.iloc[:, 1:]

# Remove columns with NaN values
xV2 = xV.dropna(axis=1)  # Drop columns with NaN values

# Execute fss function
feature1, mat = fss(xV2.values, initidx, thres, nclust)  # Convert to numpy array

# Write results to a DataFrame
results_df = pd.DataFrame({'Feature1': [feature1], 'CRI_last': [mat]})

# Save results to a CSV file
results_df.to_csv('fss_results.csv', index=False)
