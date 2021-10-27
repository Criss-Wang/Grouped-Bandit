'''
This is an implementation of Successive Elimination algorithm mentioned in the paper.
Note in this algorithm, we do not specify confidence constant c because it is specified in the bandit instance.
Further notice that we have a thy parameter here to adjust the choice of confidence value (1/sqrt(# of arm pulls) vs theoretical).
thy = 1 refers to using the theoretical confidence value while thy = 0 refers to otherwise
'''

import numpy as np
 
def compute_m_t(c, m, gb, thy):
    '''
    For each group, compute the candidates for its worst-case arms.
    '''
    res = []
    for m_i in m:
        if thy:
            UCB_i = [gb.empirical_means[arm_idx] + gb.confidence_U_theoretical(gb.individual_arm_pulls[arm_idx], gb.num_arms) for arm_idx in m_i]
            LCB_i = [gb.empirical_means[arm_idx] - gb.confidence_U_theoretical(gb.individual_arm_pulls[arm_idx], gb.num_arms) for arm_idx in m_i]
        else:
            UCB_i = [gb.empirical_means[arm_idx] + gb.confidence_U(gb.individual_arm_pulls[arm_idx], gb.num_arms) for arm_idx in m_i]
            LCB_i = [gb.empirical_means[arm_idx] - gb.confidence_U(gb.individual_arm_pulls[arm_idx], gb.num_arms) for arm_idx in m_i]
        min_UCB = min(UCB_i)
        for idx in range(len(m_i)):
            if LCB_i[idx] > min_UCB:
                m_i[idx] = -1
        res.append(list(filter(lambda x: x != -1, m_i)))
    return np.array(res)

def compute_c_t(c, m, gb, thy):
    '''
    Compute the candidate groups.
    '''
    if thy:
        UCB_f = lambda arm_idx: gb.empirical_means[arm_idx] + gb.confidence_U_theoretical(gb.individual_arm_pulls[arm_idx], gb.num_arms)
        LCB_f = lambda arm_idx: gb.empirical_means[arm_idx] - gb.confidence_U_theoretical(gb.individual_arm_pulls[arm_idx], gb.num_arms)
    else:
        UCB_f = lambda arm_idx: gb.empirical_means[arm_idx] + gb.confidence_U(gb.individual_arm_pulls[arm_idx], gb.num_arms)
        LCB_f = lambda arm_idx: gb.empirical_means[arm_idx] - gb.confidence_U(gb.individual_arm_pulls[arm_idx], gb.num_arms)
        
    min_UCBs = np.array([])
    min_LCBs = np.array([])
    for m_i in m:
        min_UCBs = np.append(min_UCBs, min([UCB_f(idx) for idx in m_i]))
        min_LCBs = np.append(min_LCBs, min([LCB_f(idx) for idx in m_i]))
    res = []
    for group_idx in c:
        min_LCB_c = max(list(min_LCBs[c]))
        if min_LCB_c <= min_UCBs[group_idx]:
            res.append(group_idx)
    return np.array(res)

def compute_a_t(c, m):
    '''
    Compute the set of arms to be sampled in the next iteration base on the remaining candidate groups c_t.
    '''
    res = set()
    for i in c:
        for arm in m[i]:
            res.add(arm)
    return np.array(list(res))

def succ_elim(gb, thy):
    '''
    Main function which terminates when a good choice of G is found.
    '''
    m = gb.groups
    c = np.arange(gb.num_groups)
    while c.size > 1:
        m = compute_m_t(c, m, gb, thy)
        c = compute_c_t(c, m, gb, thy)
        a = compute_a_t(c, m)
        gb.one_iteration(a)

        ### Stops when we obtain a highly-probably optimal group(s) based on its definition
        if np.array_equal(np.tile(a, (len(c), 1)), m[c]):
            return c
        if len(a) == 1:
            return c

        ### Early stopping to check for erroneous behaviors in this algorithm
        if (gb.iter > 1000000):
            print(m, c, a)
            return [-1]
    return c