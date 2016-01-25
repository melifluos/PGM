#!/usr/bin/env python
"""
The chinese restaurant process (CRP) prior for Bayesian nonparametric inference
author: Ben Chamberlain 19/12/2014
"""

import random as rd

# MAKE SURE THIS IS A FLOAT!!
alpha = 1000.0 # the concentration parameter, a high value will cause new tables to be filled faster

n_samples = 10
rd.seed(234987546)

table_plan = [1] # The list indices are the table numbers and the first customer sits at table zero

for sample_num in xrange(2,n_samples+2):
    cum_prob = 0
    print table_plan
    sample_val = rd.random()
    print 'sample value', sample_val
    print 'new table cutoff',alpha / (alpha - 1 + sample_num)
    cum_prob += alpha / (alpha - 1 + sample_num)
    if sample_val <= cum_prob:
        table_plan.append(1)
    else:        
        for table in range(len(table_plan)):
            cum_prob += table_plan[table] / (alpha - 1 + sample_num)
            print 'cummulative probability',cum_prob
            if sample_val <= cum_prob:
                table_plan[table] += 1
                break
            
            