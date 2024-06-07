import os
import numpy as np
import pandas as pd
from probaforms import metrics
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from .evaluation_utils import compare_on_various_runs


def get_metrics(X_real, C_real, X_fake, C_fake):
    CX_real = np.concatenate((C_real, X_real), axis=1)
    CX_fake = np.concatenate((C_fake, X_fake), axis=1)
    
    mu, sigma = metrics.frechet_distance(CX_real, CX_fake, standardize=True)
    print(r"Frechet Distance         = %.6f +- %.6f" % (mu, sigma))
    mu, sigma = metrics.kolmogorov_smirnov_1d(CX_real, CX_fake)
    print(r"Kolmogorov-Smirnov       = %.6f +- %.6f" % (mu, sigma))
    mu, sigma = metrics.cramer_von_mises_1d(CX_real, CX_fake)
    print(r"Cramer-von Mises         = %.6f +- %.6f" % (mu, sigma))
    mu, sigma = metrics.anderson_darling_1d(CX_real, CX_fake)
    print(r"Anderson-Darling         = %.6f +- %.6f" % (mu, sigma))
    # mu, sigma = metrics.roc_auc_score_1d(CX_real, CX_fake)
    # print(r"ROC AUC                  = %.6f +- %.6f" % (mu, sigma))
    # mu, sigma = metrics.kullback_leibler_1d(CX_real, CX_fake)
    # print(r"Kullback-Leibler         = %.6f +- %.6f" % (mu, sigma))
    # mu, sigma = metrics.jensen_shannon_1d(CX_real, CX_fake)
    # print(r"Jensen-Shannon           = %.6f +- %.6f" % (mu, sigma))
    mu, sigma = metrics.kullback_leibler_1d_kde(CX_real, CX_fake, n_iters=10)
    print(r"Kullback-Leibler KDE     = %.6f +- %.6f" % (mu, sigma))
    mu, sigma = metrics.jensen_shannon_1d_kde(CX_real, CX_fake, n_iters=10)
    print(r"Jensen-Shannon KDE       = %.6f +- %.6f" % (mu, sigma))
    #mu, sigma = metrics.maximum_mean_discrepancy(*shuffle(CX_real, CX_fake, random_state=11, n_samples=1000))
    #print(r"Maximum Mean Discrepancy = %.6f +- %.6f" % (mu, sigma))


def tprs_fprs_sics(preds_matrix, y_test):
    n_runs = preds_matrix.shape[0]
    n_epochs = preds_matrix.shape[1]
    tprs, fprs, sics = {}, {}, {}
    for run_id in range(n_runs):
        for ep_id in range(n_epochs):
            # signal_index = np.argwhere(y_test == 1).flatten()
            fpr, tpr, thresholds = roc_curve(y_test, preds_matrix[run_id, ep_id])
            fpr_nonzero = np.delete(fpr, np.argwhere(fpr == 0))
            tpr_nonzero = np.delete(tpr, np.argwhere(fpr == 0))
            tprs[run_id, ep_id] = tpr_nonzero
            fprs[run_id, ep_id] = fpr_nonzero
            sics[run_id, ep_id] = tprs[run_id, ep_id] / fprs[run_id, ep_id] ** 0.5
    return tprs, fprs, sics


def auc_pr(y_true, probas_pred, **kwargs):
    pr, rec, _ = precision_recall_curve(y_true, probas_pred, **kwargs)
    return auc(rec, pr)


class DetectionMetrics(object):
    def show(self, preds_matrix, is_sig_test, save_dir=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            savefig = os.path.join(save_dir, "sic_curves.jpg")
            savedf = os.path.join(save_dir, "auc_curves.csv")
        else:
            savefig = None
            savedf = None

        n_runs = preds_matrix.shape[0]
        tprs, fprs, sics = tprs_fprs_sics(preds_matrix, is_sig_test)
        _ = compare_on_various_runs([tprs], [fprs], [np.zeros(n_runs)], [""],
                                    sic_lim=(0, 20), savefig=savefig, only_median=False,
                                    reduced_legend=False, suppress_show=False, return_all=False)
        tprs_mat = np.array(list(tprs.values()), dtype="object")
        sics_mat = np.array(list(sics.values()), dtype="object")

        self.auc_roc = []
        self.auc_pr = []
        self.auc_sic = []
        for run_id in range(n_runs):
            run_preds = preds_matrix[run_id].flatten()
            self.auc_roc.append(roc_auc_score(is_sig_test, run_preds))
            self.auc_pr.append(auc_pr(is_sig_test, run_preds))
            self.auc_sic.append(auc(tprs_mat[run_id], sics_mat[run_id]))

        self.auc_df = pd.DataFrame(index=[f'run {i + 1}' for i in range(n_runs)],
                data={'AUC-ROC': self.auc_roc, 'AUC-PR': self.auc_pr, "AUC-SIC": self.auc_sic})
        if savedf is not None:
            self.auc_df.to_csv(savedf, index=False)
        return self.auc_df
