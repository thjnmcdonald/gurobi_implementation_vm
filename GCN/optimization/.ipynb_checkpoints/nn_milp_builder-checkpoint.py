"""
Created 17 January 2019
Bjarne Grimstad, bjarne.grimstad@gmail.no

Methods for building MILP models (Gurobi) of neural networks (TensorFlow/Keras models)
"""

import gurobi as gpy
import numpy as np
from GCN.optimization.bound_tightening.bt_utils import run_bound_tightening
from GCN.optimization.bound_tightening import *
from GCN.optimization.bound_tightening.bt_lp import bt_lrr, bt_rr


def get_weights(state_dict, multilayer = False):
    """
    Return weights of Keras model as a list of tuples (w, b), where w is a numpy array of weights and b is a numpy array
    of biases for a layer. The order of the list is the same as the layers in the model.
    :param model: Keras model
    :return: List of layer weights (w, b)
    """
    weight_list = []
    bias_list = []
    
    for i, weights in enumerate(state_dict):
        if 'weight' in weights:
            weight_list.append(np.transpose(np.array(state_dict[str(weights)])))
    
    
    if multilayer == True:
        for i, weights in enumerate(state_dict):
            if 'convGNN' in weights:
                weight_list.append(np.transpose(np.array(state_dict[str(weights)])))
                bias_list.append(np.zeros(np.shape(weight_list[i][1])))
        weight = list(zip(weight_list, bias_list))
    else:
        for i, weights in enumerate(state_dict):
            if 'weight' in weights:
                weight_list.append(np.transpose(np.array(state_dict[str(weights)])))
        b = [np.zeros(np.shape(weight_list[0])[1])]
        weight = list(zip([weight_list[0]], b))
        
    return(weight)


def update_bounds(x_var, s_var, z_var, lb=None, ub=None):
    """
    Update upper bound of neuron modeled as
    W x_prev + b = x - s,
    s >= 0
    x >= 0
    z in {0, 1}
    NOTE: Remember to update model after updating bounds
    :param lb: Lower bound
    :param ub: Upper bound
    :param x_var: x variable
    :param s_var: s variable
    :param z_var: z variable used in indicator constraints (z=0 => x <= 0 and z=1 => s <= 0)
    :return: None
    """
    # NOTE: default feasibility tolerance of Gurobi is 1e-6 so we cannot expect higher accuracy
    gurobi_ftol = 1e-6  # default feasibility tolerance of Gurobi
    ftol = gurobi_ftol + 1e-9  # Feasibility tolerance
    strict_ftol = 1e-12  # Strict feasibility tolerance

    if lb and ub:
        # If bounds are infeasible by a tiny amount, we remedy.
        if lb > ub and abs(ub - lb) < strict_ftol:
            diff = lb - ub
            lb -= diff/2
            ub += diff/2
        assert lb <= ub

    if lb:
        # TODO: This is very strict since constraint violation is allowed in Gurobi (will fail at some point)
        # assert x_var.getAttr(gpy.GRB.Attr.UB) + ftol >= lb >= -(s_var.getAttr(gpy.GRB.Attr.UB) + ftol)

        # Update bounds
        if lb >= 0:
            x_var.setAttr(gpy.GRB.Attr.LB, lb)
            s_var.setAttr(gpy.GRB.Attr.UB, 0)
            z_var.setAttr(gpy.GRB.Attr.LB, 1)  # Neuron activated
        else:
            s_var.setAttr(gpy.GRB.Attr.UB, -lb)

    if ub:
        # TODO: This is very strict since constraint violation is allowed in Gurobi (will fail at some point)
        # assert x_var.getAttr(gpy.GRB.Attr.UB) + ftol >= ub >= -(s_var.getAttr(gpy.GRB.Attr.UB) + ftol)

        # Update bounds
        if ub >= 0:
            x_var.setAttr(gpy.GRB.Attr.UB, ub)
        else:
            x_var.setAttr(gpy.GRB.Attr.UB, 0)
            s_var.setAttr(gpy.GRB.Attr.LB, -ub)
            z_var.setAttr(gpy.GRB.Attr.UB, 0)  # Neuron deactivated

    # Test bounds (allowing infeasibility with a strict tolerance)
    assert x_var.getAttr(gpy.GRB.Attr.UB) >= x_var.getAttr(gpy.GRB.Attr.LB) - strict_ftol
    assert s_var.getAttr(gpy.GRB.Attr.UB) >= s_var.getAttr(gpy.GRB.Attr.LB) - strict_ftol
    assert z_var.getAttr(gpy.GRB.Attr.UB) >= z_var.getAttr(gpy.GRB.Attr.LB) - strict_ftol


def GCN_milp_builder_1(model: gpy.Model, input_vars, output_vars, weights, n, hidden_bounds=None, c_id=0, bilinear = False, norm_term = None):
    # this is a version without layers
    w, b = weights[0]
    nx, nu = w.shape
    x_prev = input_vars
    s, z = [], []
    
    for i in range(n):
        s_i, z_i = [], []
        for j in range(nu):
            # we willen van onze input en output de bounds updaten, of eigenlijk
            # alleen voor output bounds and de bijbehorden s en z var. Dit kan nog
            # steeds prima werken
            s_var = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS,
                                 name=f's_{i}_{j}_{c_id}')
            z_var = model.addVar(vtype=gpy.GRB.BINARY, 
                                 name=f'z_{i}_{j}_{c_id}')
            
            s_i.append(s_var)
            z_i.append(z_var)
            model.update()

            # die s_vars en z_vars moeten wel nog gecreerd worden
            # TODO: ik zou graag die [0] weg willen halen
            if hidden_bounds[i][j]:
                lb, ub = hidden_bounds[i][j]
                update_bounds(output_vars[i][j], s_var, z_var, lb, ub)
                model.update()
                
            # Affine combination constraints
            if bilinear == False:
                model.addConstr(gpy.quicksum(w[f, j] * x_prev[i][k][f] 
                                             for f in range(nx) for k in range(n)),
                                gpy.GRB.EQUAL, output_vars[i][j] - s_i[j])
            else:
                model.addConstr(gpy.quicksum(w[f, j] * norm_term[i,k]  * x_prev[k][f] 
                                             for f in range(nx) for k in range(n)),
                                gpy.GRB.EQUAL, output_vars[i][j] - s_i[j]) 
                
            
            # Constraints for ReLU logic
            if hidden_bounds[0][j]:
                lb, ub = hidden_bounds[0][j]
                if ub >= 0:
                    model.addConstr(output_vars[i][j] <= ub*z_i[j])
                if lb <= 0:
                    model.addConstr(s_i[j] <= -lb*(1 - z_i[j]))
            else:
                # Otherwise, we use indicator constraints
                model.addGenConstrIndicator(z_i[j], False, output_vars[i][j] <= 0)
                model.addGenConstrIndicator(z_i[j], True, s_i[j] <= 0)
                
        s.append(s_i)
        z.append(z_i)
        
    return model

def GCN_milp_builder_2(model: gpy.Model, input_vars, output_vars, weights, n, hidden_bounds=None, c_id="", bilinear = False, norm_term = None):
    """
    Builds a MILP model of a neural network with ReLU activations.

    Assuming that NN model has K+1 layers, one input layers, K-1 hidden layers with ReLU activation,
    and one linear output layer. That is, the assumed architecture is:
    input -> ReLU -> ... -> ReLU -> output.

    Variables are named as x_i_j_k, where i is the layer number, and j is the unit number (of layer i), and k is the
    constraint ID (c_id).

    :param model: MILP model to add variables and constraints (Gurobi model)
    :param input_vars: List of input variables (Gurobi variables)
    :param output_vars: Output variables (Gurobi variables)
    :param weights: NN model weights
    :param hidden_bounds: Bounds on hidden neurons in NN model
    :param c_id: Constraint ID (required to uniquely name variables when adding multiple NN models to MILP model)
    :return: model
    """
    # Hidden layers (ReLU)
    # Layers are numbered from 1 and up
    x_prev = input_vars
    for k, weights_k, in enumerate(weights[:-1]):
        w, b = weights_k
        nx, nu = w.shape  # Number of inputs to layer and number of units in layer
        
        
        assert nx == len(x_prev[0])
        assert w.shape[1] == b.shape[0]

        # Layer variables
        x_i, s_i, z_i = [], [], []
        
        for i in range(n):
            
            x_j, s_j, z_j = [], [], []            
            
            for j in range(nu):
                # Create variables for layer

                x_var = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f'x_{k}_{i}_{j}_{c_id}')
                s_var = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f's_{k}_{i}_{j}_{c_id}')
                z_var = model.addVar(vtype=gpy.GRB.BINARY, name=f'z_{k}_{i}_{j}_{c_id}')
                model.update()

                x_j.append(x_var)
                s_j.append(s_var)
                z_j.append(z_var)

                # Update bounds
                if hidden_bounds[k][j]:
                    lb, ub = hidden_bounds[k][j]
                    update_bounds(x_var, s_var, z_var, lb, ub)
                    model.update()

                # Affine combination constraints
                if bilinear == False:
                    model.addConstr(gpy.quicksum(w[f, j] * x_prev[l][f] 
                                                 for f in range(nx) for l in range(n)),
                                    gpy.GRB.EQUAL, x_j[j] - s_j[j])
                else:
                    model.addConstr(gpy.quicksum(w[f, j] * norm_term[i,l]  * x_prev[l][f] 
                                                 for f in range(nx) for l in range(n)),
                                    gpy.GRB.EQUAL, x_j[j] - s_j[j]) 

                # Constraints for ReLU logic
                if hidden_bounds[k][j]:
                    # If we have bounds, we use the big-M formulation
                    lb, ub = hidden_bounds[k][j]
                    if ub >= 0:
                        model.addConstr(x_j[j] <= ub*z_j[j])
                    if lb <= 0:
                        model.addConstr(s_j[j] <= -lb*(1 - z_j[j]))
                else:
                    # Otherwise, we use indicator constraints
                    model.addGenConstrIndicator(z_j[j], False, x_j[j] <= 0)
                    model.addGenConstrIndicator(z_j[j], True, s_j[j] <= 0)
            
            x_i.append(x_j)
            s_i.append(s_j)
            z_i.append(z_j)

        x_prev = x_i

    # Output layer (affine)
    w, b = weights[-1]
    nx, nu = w.shape  # Number of inputs to layer and number of units in layer
    i = 0
    j = 0
    
    if bilinear == False:
        for i in range(n):
            for j in range(nu):
                model.addConstr(gpy.quicksum(w[f, j] * x_prev[l][f] 
                                                     for f in range(nx) for l in range(n)),
                                        gpy.GRB.EQUAL, output_vars[i][j] - s_j[j])
    else:
        for i in range(n):
            for j in range(nu):
                model.addConstr(gpy.quicksum(w[f, j] * norm_term[i,l]  * x_prev[l][f] 
                                             for f in range(nx) for l in range(n)),
                                gpy.GRB.EQUAL, output_vars[i][j] - s_j[j])
    
    
    model.update()

    return model


def GCN_milp_builder_3(model: gpy.Model, input_vars, output_vars, weights, n, hidden_bounds=None, c_id=0, bilinear = False, norm_term = None):
    """
    Builds a MILP model of a neural network with ReLU activations.

    Assuming that NN model has K+1 layers, one input layers, K-1 hidden layers with ReLU activation,
    and one linear output layer. That is, the assumed architecture is:
    input -> ReLU -> ... -> ReLU -> output.

    Variables are named as x_i_j_k, where i is the layer number, and j is the unit number (of layer i), and k is the
    constraint ID (c_id).

    :param model: MILP model to add variables and constraints (Gurobi model)
    :param input_vars: List of input variables (Gurobi variables)
    :param output_vars: Output variables (Gurobi variables)
    :param weights: NN model weights
    :param hidden_bounds: Bounds on hidden neurons in NN model
    :param c_id: Constraint ID (required to uniquely name variables when adding multiple NN models to MILP model)
    :return: model
    """
    # Hidden layers (ReLU)
    # Layers are numbered from 1 and up
    x_prev = input_vars
    num_layers = len(weights)
    print(f'these are the number of layers: {num_layers}')
    for k, weights_k, in enumerate(weights):
        print(f'we are now in layer {k}, if we are in the last layer then the following says true: {k == (num_layers - 1)}')
        w, b = weights_k
        nx, nu = w.shape  # Number of inputs to layer and number of units in layer
        
        
        assert nx == len(x_prev[0])
        assert w.shape[1] == b.shape[0]

        # Layer variables
        x_i, s_i, z_i = [], [], []
        
        for i in range(n):
            
            x_j, s_j, z_j = [], [], []            
            
            for j in range(nu):
                # Create variables for layer

                x_var = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f'x_{k}_{i}_{j}_{c_id}')
                s_var = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f's_{k}_{i}_{j}_{c_id}')
                z_var = model.addVar(vtype=gpy.GRB.BINARY, name=f'z_{k}_{i}_{j}_{c_id}')
                model.update()

                x_j.append(x_var)
                s_j.append(s_var)
                z_j.append(z_var)

                # Update bounds
                if hidden_bounds[k][j]:
                    lb, ub = hidden_bounds[k][j]
                    update_bounds(x_var, s_var, z_var, lb, ub)
                    model.update()

                # Affine combination constraints
                if bilinear == False:
                    if k != (num_layers - 1):
                        model.addConstr(gpy.quicksum(w[f, j] * x_prev[l][f] 
                                                     for f in range(nx) for l in range(n)),
                                        gpy.GRB.EQUAL, x_j[j] - s_j[j])
                    else:
                        model.addConstr(gpy.quicksum(w[f, j] * x_prev[l][f] 
                                                     for f in range(nx) for l in range(n)),
                                        gpy.GRB.EQUAL, output_vars[i][j] - s_j[j])                        
                else:
                    if k != (num_layers - 1):
                        model.addConstr(gpy.quicksum(w[f, j] * norm_term[i,l]  * x_prev[l][f] 
                                                     for f in range(nx) for l in range(n)),
                                        gpy.GRB.EQUAL, x_j[j] - s_j[j])
                    else:
                        model.addConstr(gpy.quicksum(w[f, j] * norm_term[i,l]  * x_prev[l][f] 
                                                     for f in range(nx) for l in range(n)),
                                        gpy.GRB.EQUAL, output_vars[i][j] - s_j[j])

                # Constraints for ReLU logic
                if k != (num_layers - 1):                   
                    if hidden_bounds[k][j]:
                        # If we have bounds, we use the big-M formulation
                        lb, ub = hidden_bounds[k][j]
                        if ub >= 0:
                            model.addConstr(x_j[j] <= ub*z_j[j])
                        if lb <= 0:
                            model.addConstr(s_j[j] <= -lb*(1 - z_j[j]))
                    else:
                        # Otherwise, we use indicator constraints
                        model.addGenConstrIndicator(z_j[j], False, x_j[j] <= 0)
                        model.addGenConstrIndicator(z_j[j], True, s_j[j] <= 0)
                else:
                    if hidden_bounds[k][j]:
                        # If we have bounds, we use the big-M formulation
                        lb, ub = hidden_bounds[k][j]
                        if ub >= 0:
                            model.addConstr(output_vars[i][j] <= ub*z_j[j])
                        if lb <= 0:
                            model.addConstr(s_j[j] <= -lb*(1 - z_j[j]))
                    else:
                        # Otherwise, we use indicator constraints
                        model.addGenConstrIndicator(z_j[j], False, x_j[j] <= 0)
                        model.addGenConstrIndicator(z_j[j], True, s_j[j] <= 0)
                    
            
            x_i.append(x_j)
            s_i.append(s_j)
            z_i.append(z_j)

        x_prev = x_i

    model.update()

    return model


def build_GCN_milp_and_run_bt(model: gpy.Model, input_vars, output_vars, state_dict, n, d_max, c_id=0, bt_procedures=None, bilinear = False, norm_term = None, multilayer = False):
    """
    Build a bound tightened MILP for a neural network

    Assuming that NN model has K hidden layers with ReLU activation and 1 linear output layer

    Variables are named as x_i_j_k, where i is the layer number, and j is the unit number (of layer i), and k is the
    constraint ID (c_id).

    TODO: add list of FBBT procedures to run (will be run in the order of the list)

    :param model: MILP model to add variables and constraints (Gurobi model)
    :param input_vars: List of input variables (Gurobi variables)
    :param output_vars: Output variables (Gurobi variables)
    :param nn_model: NN model (Keras model)
    :param c_id: Constraint ID (required to uniquely name variables when adding multiple NN models to MILP model)
    :param bt_procedures: Bound tightening procedures to run
    :return: model
    """
    # Get NN weights
    weights = get_weights(state_dict, multilayer = multilayer)
    
    if bilinear == True:
        F = int(len(input_vars) / n)
        input_vars = [[input_vars[i,j] for j in range(F)] for i in range(n)]
        input_bounds = [(v.getAttr(gpy.GRB.Attr.LB), 
                         v.getAttr(gpy.GRB.Attr.UB)) for v in input_vars[0]]
    
    # Tighten bounds
    else:
        input_bounds = [(v.getAttr(gpy.GRB.Attr.LB), 
                         v.getAttr(gpy.GRB.Attr.UB)) for v in input_vars[0][0]]
        
    output_bounds = [(v.getAttr(gpy.GRB.Attr.LB), 
                      v.getAttr(gpy.GRB.Attr.UB)) for v in output_vars[0]]
    bounds = run_bound_tightening(weights, input_bounds, output_bounds, bt_procedures, d_max)
    
    

    # print("Bounds of", nn_model.name)
    # print(bounds)

    # Build NN constraints
    hidden_bounds = bounds[1:]
    # print(f'the shape of the hidden bounds is; {np.shape(hidden_bounds)}')
    # print(f'this is hidden bounds 0: {hidden_bounds[0]}')
    # print(f'this is hidden bounds 1: {hidden_bounds[1]}')
    # print(f'this is hidden bounds 2: {hidden_bounds[2]}')    
    if multilayer == False:
        GCN_milp_builder_1(model, input_vars, output_vars, weights, n, hidden_bounds, c_id = 100, bilinear = bilinear, norm_term = norm_term)
    else:
        GCN_milp_builder_3(model, input_vars, output_vars, weights, n, hidden_bounds, c_id = 100, bilinear = bilinear, norm_term = norm_term)
        
    
    return model, bounds


