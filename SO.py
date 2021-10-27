'''
This is an implementation of the StableOpt variant algorithm mentioned in the paper.
We have three helper functions: select_G_t, select_a_t and terminate.
The main function stable_opt combines these functions and return a guess of the optimal group G_t.
Make sure a proper instance of Grouped_Bandit is input to the function and the
confidence constant / tolerance value (> 0) is supplied.
'''
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)

def select_G_t(gb, ucb_t):
    '''
    Choose the group to evaluate in the next iteration based on posterior distribution and confidence interval.
    '''
    G_max = np.array([ucb_t[g].min() for g in gb.groups]).max()
    candidates = np.argwhere([ucb_t[g].min() for g in gb.groups] == G_max).reshape(1,-1)[0]

    ## Random selection in case of multiple candidates present
    [G_t] = np.random.choice(candidates, 1)
    return G_t

def select_a_t(gb, lcb_t, G_t):
    '''
    Choose the arm to evaluate in the next iteration based on posterior distribution and confidence interval.
    '''
    a_min = lcb_t[gb.groups[G_t]].min()
    candidates = np.argwhere(lcb_t[gb.groups[G_t]] == a_min).reshape(1,-1)[0]
    
    ## Random selection in case of multiple candidates present
    [idx] = np.random.choice(candidates, 1)
    return gb.groups[G_t][idx]

def terminate(gb, ucb, lcb, G, a, c, eta):
    '''
    Check if the candidate is good enough so that the algorithm can terminate.
    '''
    sub_G = [x for x in [a for a in range(len(gb.groups))] if x != G]
    if (len(sub_G) == 0):
        ucb_sub = 1e9
    else:
        ucb_sub = max([ucb[gb.groups[grp]].min() for grp in sub_G])
    if lcb[a] >= ucb_sub - eta: ## eta is a tolerance value to allow early convergence in experiments
        return True

    return False

def stable_opt(gb, c, eta):
    '''
    Main StableOpt variant function
    '''
    t = 0
    terminated = False
    while not terminated:
        t += 1
        ucb_t = gb.empirical_means + c / np.sqrt(gb.individual_arm_pulls)
        lcb_t = gb.empirical_means - c / np.sqrt(gb.individual_arm_pulls) # np.sqrt(beta) * std

        G_t = select_G_t(gb, ucb_t)
        a_t = select_a_t(gb, lcb_t, G_t)
        terminated = terminate(gb, ucb_t, lcb_t, G_t, a_t, c, eta)

        gb.one_iteration(np.array([a_t]))
        
    return [G_t]