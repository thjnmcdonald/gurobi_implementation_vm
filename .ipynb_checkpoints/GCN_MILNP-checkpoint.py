# !/usr/bin/env python
# coding: utf-8

from os import chdir
import torch
import pandas as pd
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
from smiles_to_molecular_graphs.single_molecule_conversion import process
from torch_geometric.utils import to_dense_adj
from GCN.optimization.nn_milp_builder import *
from optimization.nn_milp_builder_tom_version_for_GCN import *
from optimization.nn_milp_builder_tom_version import *
from optimization.bound_tightening.bt_lp import bt_lrr as bt_lrr_ann
from GCN.optimization.bound_tightening.bt_lp import bt_lrr as bt_lrr_gcn

print(f'done loading directories')

state_dict = torch.load("trained_models/multi_layer_24_may.tar")

hardcode = process('CCCCl')
edge_index = hardcode.edge_index
adjacency_matrix = to_dense_adj(edge_index)
hardcode_A = adjacency_matrix
hardcode_features = hardcode.x
rel_data = [edge_index, hardcode_A, hardcode_features]   


def normalization_term(d_max):
    g = np.zeros((d_max + 1)**2)
    for i in range(d_max+1):
        for j in range(d_max+1):            
            if (i == 0) or (j==0):
                g[i*(d_max + 1) + j] = 0
            else:
                g[i*(d_max + 1) + j] = 1/((math.sqrt(i))*math.sqrt(j))
    return g

g = normalization_term(4)


def make_norm_layers(m: gp.Model, d_max, F, n, A, x):
    M_1 = 1
    M_2 = 1
    g = normalization_term(d_max)

    
    c = m.addVars(n, n, (d_max + 1)**2, vtype = GRB.BINARY, name = "c")
    b = m.addVars(n, n, F, vtype = GRB.BINARY, name = "b")
    sp = m.addVars(n, n, lb = 0, ub=1, vtype = GRB.CONTINUOUS, name = "sp")
    d_plus = m.addVars(n, lb = 0, ub=(n+1), vtype = GRB.INTEGER, name="d_plus")
    p = m.addVars(n, n, ub = (d_max + 1)**2, vtype = GRB.INTEGER, name = "p")
    t = m.addVars(n, n, lb = 0, ub=1, vtype = GRB.CONTINUOUS, name = "t")
    m.update()

    m.addConstrs(((d_plus[i] == gp.quicksum(A[i,j] for j in range(n)))
                 for i in range(n)), name = '6a')

    m.addConstrs(((p[i,k] == d_plus[i]*(d_max + 1) + d_plus[k])
                 for i in range(n) for k in range(n)), name='6b')

    m.addConstrs(((0 == p[i,k] - (gp.quicksum(t*c[i,k,t] for t in range((d_max + 1)**2))) )
                 for i in range(n) for k in range(n)), name='6c')

    m.addConstrs(((1 == (gp.quicksum(c[i,k,t] for t in range((d_max + 1)**2))) )
                 for i in range(n) for k in range(n)), name='6d')

    m.addConstrs(((sp[i,k] == (gp.quicksum(g[t]*c[i,k,t] for t in range((d_max + 1)**2))) )
                 for i in range(n) for k in range(n)), name='6e')
    
    # for bi-linear case (17.g and 17.h)
    m.addConstrs(((sp[i,k] - M_1*(1 - A[i,k]) <= t[i,k])
                 for i in range(n) for k in range(n)), name='17g-a')
    m.addConstrs(((t[i,k] <= M_1*(1 - A[i,k]) + sp[i,k])
                 for i in range(n) for k in range(n)), name='17g-b')    
    
    
    m.addConstrs(((-M_1*(A[i,k]) <= t[i,k])
                 for i in range(n) for k in range(n)), name='17h-a')
    m.addConstrs(((t[i,k] <= M_1*(A[i,k]))
                 for i in range(n) for k in range(n)), name='17h-b')    
    
    m.update()
    
    # add hardcoded conconstraints
    
    m.update()
    return m, t



def make_input_constraints(m: gp.Model, rel_data, n, F):
    features = rel_data[2]
    edge_matrix = rel_data[1][0]
    feature_vectors = rel_data[-1]
    
    feat_vec = m.addVars(n, F, vtype = GRB.BINARY, name = "feat_vec")
    A = m.addVars(n, n, vtype = GRB.BINARY, name= "A")
    m.update() 
    
    for i in range(n):
        for j in range(n):
            if i == j:
                m.addConstr(A[i,j] == 1)
            else:
                m.addConstr((A[i,j] == int(edge_matrix[i][j])))
                
    m.update()
    
    for i in range(n):
        for j in range(F):
            m.addConstr(feat_vec[i,j] == feature_vectors[i,j])
            
    m.update()
    
    return m, A, feat_vec


def make_GCN_milp(m: gp.Model, bt_procedures, x, n, F, d_max, 
                  rel_data = rel_data, state_dict = state_dict, 
                  bilinear = False, norm_term = None, multi_layer = False):
    
    GCN_output_len = 32
    
    h_prime = m.addVars(n, GCN_output_len, lb = 0, vtype = GRB.CONTINUOUS, name="h_prime")
    h_prime_vars = [[h_prime[i,j] for j in range(GCN_output_len)] for i in range(n)]
    m.update()
    
    
    # GCN layer builder
    m, bounds= build_GCN_milp_and_run_bt(m, x, h_prime_vars, state_dict, 
                                         n, d_max, c_id=0, bt_procedures=bt_procedures, 
                                         bilinear = bilinear, norm_term = norm_term, 
                                         multilayer = multi_layer)
    m.update()
    
    h = m.addVars(GCN_output_len, vtype = GRB.CONTINUOUS, name = 'h')
    m.update()
    m.addConstrs((( ( (gp.quicksum(h_prime[i, j] for i in range(n))) == h[j])) for j in range(GCN_output_len))
                 , name = 'max_pool')
    
    j = 0
    for (lb, ub) in (bounds[-1]):
        h[j].setAttr(gpy.GRB.Attr.LB, (lb))
        h[j].setAttr(gpy.GRB.Attr.UB, (ub))
        j += 1
        
    m.update()
    
    # m.update()
    # for i in range(GCN_output_len):
    #     h_ub = n * bounds[1][i][1]
    #     h[i].setAttr(gpy.GRB.Attr.UB, h_ub)
    m.update()
    
    return h_prime, h, m, bounds




def make_ANN_milp(m: gp.Model, bt_procedures, n, h, rel_data = rel_data, state_dict = state_dict):
    ann_bt_procedures = [bt_lrr_ann]
    
    y = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'y')
    
    m.update()
    hi = [h[i] for i in range(len(h))]
        
    yi = [y]
    
    
    m = build_milp_and_run_bt(m, hi, yi, state_dict, bt_procedures = ann_bt_procedures)
    m.update()
    
    return m, y    
    
    




def make_model_optimize_and_save(bt_procedures = [bt_lrr_gcn], 
                                 ann_bt_procedures = [bt_lrr_ann], 
                                 n = 4, F = 21, d_max = 4,
                                rel_data = rel_data, state_dict = state_dict, bilinear = False):
    m = gp.Model("GNN")  
    bt_procedures = [bt_lrr_gcn]
    ann_bt_procedures = [bt_lrr_ann]
    m.update()
    
    m, A, x = make_input_constraints(m, rel_data, n, F)
    m.update()
    
    m, t_vars = make_norm_layers(m, d_max, F, n, A, x)
    m.update()
    
    h_prime, h, m, bounds = make_GCN_milp(m, bt_procedures, x, n, F, d_max, bilinear = bilinear, norm_term = t_vars, multi_layer = True)  
    m.update()    
    
    # file_name = 'delete_this_pls_16'
    # m.write(file_name + '.lp')
    # # m.write(file_name + '.sol')
    
    m, y =  make_ANN_milp(m, ann_bt_procedures, n, h, rel_data, state_dict)
    m.update()   
    
    objective = m.getVarByName('y')
    m.setObjective(objective, GRB.MAXIMIZE)
    m.update()
    
    file_name = 'here_we_go_again'
    m.write(file_name + '.lp')

    # if bilinear == True:
    #     m.Params.NonConvex = 2
    
    m.optimize()
    
    obj = m.getObjective()
    val = obj.getValue()
    # set the objective and see where things go wrong. 
    m.write(file_name + '.sol')
    m.printAttr('x')
    return val





n = 4
F = 21

def test_make_model_optimize_and_save(t, bt_procedures = [bt_lrr_gcn], 
                                 ann_bt_procedures = [bt_lrr_ann], 
                                 n = 4, F = 21, d_max = 4,
                                rel_data = rel_data, state_dict = state_dict, bilinear = True):
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("GNN", env=env)  
    bt_procedures = [bt_lrr_gcn]
    ann_bt_procedures = [bt_lrr_ann]

    
    m, A, x = make_input_constraints(m, rel_data, n, F)
    m.update()
    
    m, t_vars = make_norm_layers(m, d_max, F, n, A, x)
    m.update()
    
    h_prime, h, m, bounds = make_GCN_milp(m, bt_procedures, x, n, F, d_max, bilinear = bilinear, norm_term = t_vars)  
    m.update()    
    
    
    # file_name = 'delete_this_pls_16'
    # m.write(file_name + '.lp')
    # # m.write(file_name + '.sol')
    
    # m, y =  make_ANN_milp(m, ann_bt_procedures, n, h, rel_data, state_dict)
    # m.update()   
    
    # objective = m.getVarByName('h_prime[0,0]')
    # m.setObjective(objective, GRB.MAXIMIZE)
    # m.update()
    
    file_name = 'here_we_go_again'
    m.write(file_name + '.lp')

    if bilinear == True:
        m.Params.NonConvex = 2
    

    # objective = m.getVarByName(f'h_prime[{t}][{feature}]')
    # objective = h_prime[t,feature]
    objective = h[t]

    m.setObjective(objective, GRB.MAXIMIZE)
    # m.setObjective(objective, GRB.MINIMIZE)
    m.update()
    m.optimize()
    
    
    obj = m.getObjective()
    val = obj.getValue()
    
    return val

def sqrt(i):
    squareroot = i**(1/2)
    return squareroot


def find_h_prime_val(rel_data,i,j):    
    adj = rel_data[1][0]
    for f in range(n):
        adj[f,f] = 1
    
    degrees = [int(num) for num in list(np.array(adj.sum(axis = 0)))]
    
    sum_val = 0
    
    node_index_array  = np.array(rel_data[0])
    indices = np.where(node_index_array[0] == i)[0]
    adj_node_list = [node_index_array[1][l] for l in indices]
    adj_node_list.append(i)

    for k in adj_node_list:
        norm_term = 1/(sqrt(degrees[i]) * sqrt(degrees[k]))
        additive = norm_term * (np.array(rel_data[-1][k]) * np.array(state_dict['convGNN.lin.weight'][j])).sum()
        sum_val += additive
    
    sum_val = max(0,sum_val)
    return(sum_val)

def find_neg_val(n, F, rel_data = rel_data):
    for i in range(n):
        for j in range(F):
            if find_h_prime_val(rel_data, i, j) <= 0:
                print(i,j)

def find_h_val(j):
    sum_val = 0
    for i in range(n):
        additive = find_h_prime_val(rel_data,i,j)
        # print(additive)
        sum_val = sum_val + additive
    
    sum_val = max(0.0, sum_val)
    return(sum_val)

def compare_h_val():
    for w in range(32):
        print(f'manual h_val node {w}: {find_h_val(w)}')
        print(f'found h_val node  {w}: {test_make_model_optimize_and_save(t = w)}')
        
def compare_h_prime_val():
    for w in range(32):
        print(f'manual h_val node {w}: {find_h_prime_val(rel_data, 0, w)}')
        print(f'found h_val node  {w}: {test_make_model_optimize_and_save(t = w, bilinear=True)}')



def main():
    make_model_optimize_and_save(bilinear = True)


if __name__ == "__main__":
    main()
    

