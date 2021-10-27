'''
Here are some utility functions used for generating a proper instance of grouped bandit.
In our experiment, we choose the set all arms' mean reward to be between 0 and 1.
Therefore, make sure the parameter value of gap does not fall out of this range.
'''

import numpy as np

def generate_groups(num_groups, num_arms):
    '''
    Generate an instance with group size = num_groups and total arm amount = num_arms.
    Randomly assign each arm to some of the groups.
    '''
    groups = []
    found = np.zeros(num_arms)
    
    for i in range(num_groups):
        group = []
        while len(group) < 1: # Ensure every group is non-empty
            for j in range(num_arms):
                # Each arm is put in one gropu with probability 1/num_groups to ensure even allocation
                if np.random.rand() <= 1/num_groups:
                    group.append(j)
                    found[j] = 1
        groups.append(group)

    # Deal with any arm not allocated to any groups yet: randomly put it in a group 
    for k in range(num_arms):
        if not found[k]:
            groups[np.random.randint(0, num_groups)].append(k)
    
    # Sort arms in each group by index before assigning mean reward to each arm.
    for i in range(num_groups):
        groups[i].sort()
    return np.array(groups)

def generate_non_overlap(num_groups, num_arms):
    '''
    Generate a set of groups with no arm being present in multiple groups (the non-overlapping setup).
    '''
    groups = []
    for i in range(num_groups):
        group = []
        groups.append(group)
    for idx in range(num_arms):
        # Randomly assign a group to each arm
        g_idx = np.random.randint(0,num_groups)
        groups[g_idx].append(idx)
    g_suff = [g for g in groups if len(g) > 10]
    for g in groups:
        # Deal with empty group: pop an arm from some groups with >= 10 arms and put it into the empty group
        if len(g) == 0:
            g.append(g_suff[0][0])
            g_suff[0] = g_suff[0].remove(g_suff[0][0])
    for i in range(num_groups):
        groups[i].sort()
    return np.array(groups)

def generate_means(num_groups, num_arms, gap, groups):
    '''
    Generate mean reward for each arm:
    - The optimal worst-case mean = 0.5
    - The best suboptimal worst-case mean = 0.5 - gap
    - Randomly assign a value in (0, 0.5-gap) for other worst-case arms
    - Randomly assign a value in (worst-case mean, 1) for other arms
    '''
    means = [0] * num_arms
    assigned = np.zeros(num_arms)
    worst_min = 0.5
    for k in range(num_groups):
        i = 0
        while assigned[groups[k][i]]:
            i += 1
        if i >= len(groups[k]):
            i -= 1
            
        means[groups[k][i]] = worst_min
        assigned[groups[k][i]] = 1
        while i < len(groups[k]):
            if not assigned[groups[k][i]]:
                means[groups[k][i]] = worst_min + np.random.rand() * (1 - worst_min)
                assigned[groups[k][i]] = 1
            i += 1
        
        if k == 0:
            worst_min -= gap
        else:
            worst_min = np.random.rand() * (0.5 - gap)
  
    return np.array(means)