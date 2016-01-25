#!/usr/bin/env python
"""
A node in a probabilistic graphical model
author: Ben Chamberlain 19/11/2014
"""

from sets import Set
from operator import mul # used to calculate the number of rows in a factor
import numpy as np
import copy

VERTEX_NAMES = {
                1:'gender',
                2:'S1',
                3:'S2'
                }

GENDER = 0
S1 = 1
S2 = 2

class Factor():
    """
    A factor maps a value to all possible combinations of values of the variables in the factor
    We can think of it as a component of the factorised joint distribution
    card - The number of different values the factor can take
    vars - The variables represented by this factor
    vals - The probability of each possible configutation of variables
    HOW DO I KNOW IF THESE ENCODE JOINT OR CONDITIONAL DISTRIBUTION. WHICH IS CORRECT? USE BOTH
    
    
    """    
    def __init__(self,card=[],vars=[],vals=[]):
        self.card = np.array(card)
        self.vars = np.array(vars)
        self.vals = np.array(vals)
        
    def copy(self):
        return copy.deepcopy(self)
        
    def __str__(self):
        return 'Factor instance: card = ' + str(self.card) + ', vars = ' + str(self.vars) + ', vals = ' + str(self.vals) 
    
    def print_factor(self):
        """ 
        display the factor in table form
        """
        n_rows = np.prod(self.card)
        n_cols = len(self.card)
        
        print 'factor values'
        header_str = ''
        
        for var in range(n_cols):
            header_str = header_str + 'V' + str(self.vars[var]) + '\t'
        print header_str,'val'
        
        
        assignments = index_to_assignment(range(n_rows),self.card)
            #test_row = assignment_to_index(var_vals,self)
            #print 'test: ',row,test_row
        row = 0
        for assignment in assignments:
            row_str = ''
            for var in assignment:
                row_str = row_str + str(var) + '\t'  
            
            print row_str,self.vals[row] 
            row+=1                         
        
    def marginalise(self,vars):
        """
        sums a factor over all possible values of the variables in vars
        """
        # produce new card and var vectors
        if isinstance(vars,list):
            all_vars = list(Set(self.vars).difference(Set(vars)))
        elif isinstance(vars,int):
            var_set = set()
            var_set.add(vars)
            all_vars = list(Set(self.vars).difference(var_set))
        
        vars = np.array(all_vars).flatten()
        # Get the indices of the vars of the input factors into the vars of the output factor
        map = get_member_indices(vars,self.vars)                            
                
        card = np.array(self.card[map]).flatten()       
        
        vals = np.zeros(np.prod(card))
        
        # find all rows that have the same assignments to all of the variables not being marginalised out
        
        # find the assignments of all vars in the input
        assignments = index_to_assignment(range(len(self.vals)),self.card)
        
        # Now get the indices into the output that correspond to each of these assignments
        indices = assignment_to_index(assignments[:,map],card)
        
        # finally sum up the vals for the input for each index into the output
        row = 0
        for idx in indices:
            vals[idx] += self.vals[row]
            row += 1
            
        return Factor(card,vars,vals)  
        
        # sum them up and reset self.vars, self.vals and self.card
        
        
    
def get_member_indices(l1,l2):
    """
    finds the indices of elements of l1 in l2
    """
    retval = []
    if len(l1) > len(l2):
        raise Exception
    for elem in l1:
        retval.append(np.where(l2==elem))
    return np.array(retval)
    
def factor_product(f1,f2):
    """
    takes the product of two factors. This is basically an inner / natural join
    """
    l1 = len(f1.vars)
    l2 = len(f2.vars)
    if l1 == 0:
        return f2
    if l2 == 0:
        return f1
    
    all_vars = list(Set(f1.vars).union(Set(f2.vars)))
    
    # Get the indices of the vars of the input factors into the vars of the output factor
    map1 = get_member_indices(f1.vars,all_vars)
    map2 = get_member_indices(f2.vars,all_vars)
    
    card = np.zeros(len(all_vars))
    
    card[map1] = f1.card
    card[map2] = f2.card
    
    ret_factor = Factor(card.astype(int),all_vars,[])        
    
    # For each val in the output factor multiply the correct val in f1 with that in f2
    # get the assignments to variables for each index in the vals list
    assignments = index_to_assignment(range(np.prod(ret_factor.card)),card)
    # extract the indices from the val list for each factor
    idx1 = assignment_to_index(assignments[:,map1],f1.card).astype(int)
    idx2 = assignment_to_index(assignments[:,map2],f2.card).astype(int)
    # multiply together the correct elements from the val lists
    ret_factor.vals = np.multiply(np.array(f1.vals)[idx1],np.array(f2.vals)[idx2])
        
    return ret_factor
        
            

class Graph():
    def __init__(self,vertices,edges):
        self.vertices = vertices
        self.edges = edges
        self.factors = {}
    
    def build_graph(self):
        """
        use the edge list and the nodes to construct the graph
        
        """
        
        for edge in self.edges:
            # outgoing vertex
            vertex_name0 = edge[0]
            vertex0 = self.vertices[vertex_name0]
            if vertex_name0 not in self.factors:                    
                self.factors[vertex_name0] = Factor([vertex0.card],[edge[0]],[])
            # incoming vertex
            vertex_name1 = edge[1]
            vertex1 = self.vertices[vertex_name1]
            if vertex_name1 not in self.factors:                    
                self.factors[vertex_name1] = Factor([vertex0.card,vertex1.card],[vertex_name0,vertex_name1],[])
                # vertex 0 has a directed edge to this vertex so increase the scope of the factor
            else:
                self.factors[vertex_name1].card.append(vertex1.card)
                self.factors[vertex_name1].vars.append(vertex_name0)
            
    def load_probabilites(self):
        """
        get the conditional probabilites into the graph factors
        THIS DOESN'T SEEM LIKE A VERY SATISFACTORY WAY OF GETTING THE DATA IN
        """
        self.factors[0].vals = [0.6,0.4]
        self.factors[1].vals = [0.1,0.9,0.6,0.4]
        self.factors[2].vals = [0.6,0.4,0.3,0.7]
    
    def do_inference(self,obs,target):
        """
        performs inference on the network given a set of observations 
        and returns a pdf for the target variable
        """
        # find the target variable
        
        # somehow get rid of the now impossible elments
        # find incoming links to the target
        
        incoming_edges = []
        
        for vertex in vertices:
            if target == vertex[1]:
                incoming_edges.append(edge)
        
        return incoming_edges    
    
        # use the 
        
class Node():
    """
    the node class 
    """
    def __init__(self,card,vals):
        self.card = card
        self.vals = vals
    
    def __str__(self):
        return 'Node instance: card = ' + str(self.card) + ', vals = ' + str(self.vals)
    

def index_to_assignment(idx,card):
    """
    converts from an index into the values list of 
    a node to the values of each variable
    """
    retval = []
    # get the card in reverse order
    inv_card = card[::-1]
    # get the period for which each variable has a constant assignment
    change_period = np.cumprod(inv_card[1:])
    # the last variable changes every time
    change_period = np.append(change_period,1).astype(int)

    # how many times has each variable changed up until now
    n_changes = np.divide(np.tile(idx,(len(card),1)).T,change_period)
    
    return np.mod(n_changes,card)


def assignment_to_index(assignments,card):
    """
    converts from the values of all variables to the index in vals
    of that configuration
    """
    # The values of the vars encode a number of len(vars) where the base of each column is card[col]
    
    # get the card in reverse order
    try:
        inv_card = card[::-1]
        # get the period for which each value is constant
        change_period = np.cumprod(inv_card[1:])
        # the last variable changes every time
        change_period = np.append(change_period,1).astype(int)
    except TypeError: 
        # card is just an int
        change_period = 1
        
    idx = np.dot(change_period,assignments.T)

    return idx.flatten()        

if __name__ == '__main__':
    node1 = Node(2,[0.4,0.6])
    node_list = {GENDER:Node(2,[0.4,0.6]),S1:Node(2,[0.1,0.9,0.7,0.3]),S2:Node(2,[0.1,0.9,0.7,0.3])}
    #node_list = [GENDER,S1,S2]
    edge_list = [(GENDER,S2),(GENDER,S1)]
    graph = Graph(node_list,edge_list)
    graph.build_graph()
    graph.load_probabilites()
    fac_vec = []
    for vertex,factor in graph.factors.iteritems():
        factor.print_factor()
        fac_vec.append(factor)
    factor_product(fac_vec[0],fac_vec[1]).print_factor()
    m = fac_vec[1].marginalise(0)
    m.print_factor()
    #print graph.do_inference([(GENDER,0)], S1)
    
    
    

