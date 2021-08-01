from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np
from datetime import datetime


# all AUC results - from the Excel file
# mix_match
AUC0 = [0.678757639, 0.686533702, 0.723166411, 0.744005868, 0.715618233, 0.722449811, 0.77076312, 0.776765735,
        0.743803242, 0.762129524, 0.74705665, 0.679233415, 0.806059037, 0.804351206, 0.810804222, 0.723452551,
        0.710875952, 0.672010153, 0.698364823, 0.669251633]

# mix_match improved
AUC1 = [0.598008542, 0.626392505, 0.694067535, 0.683758308, 0.593249598, 0.75098382, 0.74827277, 0.786368526,
        0.701608543, 0.714754386, 0.661322393, 0.65183743, 0.772974, 0.7505499, 0.765586529, 0.73137112,
        0.712571019, 0.665545621, 0.732333543, 0.716352194]

# baseline
AUC2 = [0.458255892, 0.523676639, 0.631067695, 0.524292372, 0.504393654, 0.607128536, 0.616248749, 0.59534918,
        0.511331653, 0.62898656, 0.550372374, 0.625536191, 0.696363034, 0.651505074, 0.611181172, 0.550990483,
        0.540720513, 0.544127818, 0.534100446, 0.567894717]


def friedman_test(AUC_0, AUC_1, AUC_2, dt, alpha=0.05):
    """
    This function performs the friedman test for the results of the three algorithms.
    If the test is significant it calls the Nemenyi post-hoc test
    :param AUC_0: AUC result of the original MixMatch for all datasets
    :param AUC_1: AUC result of the improved MixMatch for all datasets
    :param AUC_2: AUC result of the baseline for all datasets
    :param dt: date and time
    :param alpha: significance level, 0.05 by default
    """
    stat, pvalue = friedmanchisquare(AUC_0, AUC_1, AUC_2)
    if pvalue <= alpha:
        print(f'test is significant, statistic: {stat}, p-value: {pvalue}')
        post_hoc_test(AUC_0, AUC_1, AUC_2, dt)
    else:
        print(f'test is significant, statistic: {stat}, p-value: {pvalue}')


def post_hoc_test(AUC_0, AUC_1, AUC_2, dt):
    """
    This function performs the Nemenyi post-hoc test
    :param AUC_0: AUC result of the original MixMatch for all datasets
    :param AUC_1: AUC result of the improved MixMatch for all datasets
    :param AUC_2: AUC result of the baseline for all datasets
    :param dt: date and time
    """
    all_AUC = np.array([AUC_0, AUC_1, AUC_2])
    # perform Nemenyi post-hoc test
    p_values_df = sp.posthoc_nemenyi_friedman(all_AUC.T)
    p_values_df.to_csv(f'post_hoc_result_{dt}.csv')


dt = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
friedman_test(AUC0, AUC1, AUC2, dt)
