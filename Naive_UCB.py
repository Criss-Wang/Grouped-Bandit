'''
This is an implementation of the Group-wise naive UCB algorithm which performs very suboptimally 
under some bandit instance.
Make sure a proper instance of Grouped_Bandit is input to the function and the
confidence constant c / tolerance value eta (> 0) is supplied.
'''

import numpy as np

def find_g_min(gb, g, c, eta):
    '''
    Identify the worst-case arm within a group g using confidence-based method.
    '''
    candidate = g
    if len(g) == 1:
        return g[0]
    while True:
        ucb_t = gb.empirical_means + c / np.sqrt(gb.individual_arm_pulls)
        lcb_t = gb.empirical_means - c / np.sqrt(gb.individual_arm_pulls)

        g_min = lcb_t[g].min()
        candidates = np.argwhere(lcb_t[g] == g_min).reshape(1,-1)[0]

        # Tie-breaker if multiple candidates are present
        [idx] = np.random.choice(candidates, 1)
        x_t = g[idx]
        if (ucb_t[x_t] < min([lcb_t[a] for a in g if a != x_t]) + eta):
            # terminate when a highly-probably worst-case arm is found
            break
        gb.one_iteration(np.array([x_t]))

    return x_t

def group_wise_UCB(gb, c, eta):
    '''
    Within each group, we run find_g_min first to obtain the worst-case arm for the group.
    We proceed to compare these worst-case arms by furthur pulling them and comparing via confidence-based method.
    '''
    group_min = []
    
    for g in gb.groups:
        g_min_x_t = find_g_min(gb, g, c, eta)
        group_min.append(g_min_x_t)
    
    group_min_set = list(set(group_min))
    while True:
        ucb_t = gb.empirical_means + c / np.sqrt(gb.individual_arm_pulls)
        lcb_t = gb.empirical_means - c / np.sqrt(gb.individual_arm_pulls)
        g_worst_min = lcb_t[group_min_set].max()
        candidates = np.argwhere(lcb_t[group_min_set] == g_worst_min).reshape(1,-1)[0]

        #Tie-breaker
        [idx] = np.random.choice(candidates, 1)
        x_t = group_min_set[idx]
        if (lcb_t[x_t] > max([ucb_t[a] for a in group_min_set if a != x_t]) - eta):
            # terminate when a highly-probably best worst-case arm is found
            break
        gb.one_iteration(np.array([x_t]))
        
    # Identify all the groups with this worst-case arm
    optimal_G = [g for g in gb.groups if min(gb.empirical_means[g]) == gb.empirical_means[x_t]]
    
    G = []
    for grp in optimal_G:
        G.append(list(gb.groups).index(grp))
    return G