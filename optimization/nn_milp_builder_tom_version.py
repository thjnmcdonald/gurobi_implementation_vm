"""
Created 17 January 2019
Bjarne Grimstad, bjarne.grimstad@gmail.no

Methods for building MILP models (Gurobi) of neural networks (TensorFlow/Keras models)
"""

import gurobi as gp
import numpy as np
from optimization.bound_tightening.bt_utils import run_bound_tightening
from optimization.bound_tightening import *
from optimization.bound_tightening.bt_lp import bt_lrr, bt_rr


def get_weights(state_dict, GCN_layer_included = True, multilayer = False):
    """
    Return weights of Keras model as a list of tuples (w, b), where w is a numpy array of weights and b is a numpy array
    of biases for a layer. The order of the list is the same as the layers in the model.
    :param model: Keras model
    :return: List of layer weights (w, b)
    """
    # weights = model.get_weights()
    # assert len(weights) % 2 == 0
    # return list(zip(weights[0::2], weights[1::2]))
    
    # assert len(state_dict) % 2 == 0
    weight_list = []
    bias_list = []
    
    if multilayer == False:
        for i, weights in enumerate(state_dict):
            if 'weight' in weights:
                weight_list.append(np.transpose(np.array(state_dict[str(weights)])))
            if 'bias' in weights:
                bias_list.append(np.transpose(np.array(state_dict[str(weights)])))

        if GCN_layer_included:
            if len(weight_list) == len(bias_list): 
                bias_list = bias_list[1:]
            else:
                bias_list = bias_list
            weight_list = weight_list[1:]

        else:
            weight_list = weight_list
            bias_list = bias_list

        # for weight in weight_list:
        #     None
        #     # print(weight.shape)

        # for bias in bias_list:
            
            # print(bias.shape)
    else:
        for weights in state_dict:
            if 'fc' in weights:
                if 'weight' in weights:
                    weight_list.append(np.transpose(np.array(state_dict[str(weights)])))
                if 'bias' in weights:
                    bias_list.append(np.transpose(np.array(state_dict[str(weights)])))
        
    return list(zip(weight_list, bias_list))

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
        # assert x_var.getAttr(gp.GRB.Attr.UB) + ftol >= lb >= -(s_var.getAttr(gp.GRB.Attr.UB) + ftol)

        # Update bounds
        if lb >= 0:
            x_var.setAttr(gp.GRB.Attr.LB, lb)
            s_var.setAttr(gp.GRB.Attr.UB, 0)
            z_var.setAttr(gp.GRB.Attr.LB, 1)  # Neuron activated
        else:
            s_var.setAttr(gp.GRB.Attr.UB, -lb)

    if ub:
        # TODO: This is very strict since constraint violation is allowed in Gurobi (will fail at some point)
        # assert x_var.getAttr(gp.GRB.Attr.UB) + ftol >= ub >= -(s_var.getAttr(gp.GRB.Attr.UB) + ftol)

        # Update bounds
        if ub >= 0:
            x_var.setAttr(gp.GRB.Attr.UB, ub)
        else:
            x_var.setAttr(gp.GRB.Attr.UB, 0)
            s_var.setAttr(gp.GRB.Attr.LB, -ub)
            z_var.setAttr(gp.GRB.Attr.UB, 0)  # Neuron deactivated

    # Test bounds (allowing infeasibility with a strict tolerance)
    assert x_var.getAttr(gp.GRB.Attr.UB) >= x_var.getAttr(gp.GRB.Attr.LB) - strict_ftol
    assert s_var.getAttr(gp.GRB.Attr.UB) >= s_var.getAttr(gp.GRB.Attr.LB) - strict_ftol
    assert z_var.getAttr(gp.GRB.Attr.UB) >= z_var.getAttr(gp.GRB.Attr.LB) - strict_ftol


def nn_milp_builder(model: gp.Model, input_vars, output_vars, weights, hidden_bounds=None, c_id=0):
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
    for i, weights_i, in enumerate(weights[:-1]):
        w, b = weights_i
        nx, nu = w.shape  # Number of inputs to layer and number of units in layer
        
        
        assert nx == len(x_prev)
        assert w.shape[1] == b.shape[0]

        # Layer variables
        x, s, z = [], [], []

        for j in range(nu):
            # Create variables for layer

            x_var = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f'x_{i + 1}_{j}_{c_id}')
            s_var = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f's_{i + 1}_{j}_{c_id}')
            z_var = model.addVar(vtype=gp.GRB.BINARY, name=f'z_{i + 1}_{j}_{c_id}')
            model.update()

            x.append(x_var)
            s.append(s_var)
            z.append(z_var)

            # Update bounds
            if hidden_bounds[i][j]:
                lb, ub = hidden_bounds[i][j]
                update_bounds(x_var, s_var, z_var, lb, ub)
                model.update()

            # Affine combination constraints
            model.addConstr(gp.quicksum(w[k, j] * x_prev[k] for k in range(nx)) + b[j], gp.GRB.EQUAL, x[j] - s[j])

            # Constraints for ReLU logic
            if hidden_bounds[i][j]:
                # If we have bounds, we use the big-M formulation
                lb, ub = hidden_bounds[i][j]
                if ub >= 0:
                    model.addConstr(x[j] <= ub*z[j])
                if lb <= 0:
                    model.addConstr(s[j] <= -lb*(1 - z[j]))
            else:
                # Otherwise, we use indicator constraints
                model.addGenConstrIndicator(z[j], False, x[j] <= 0)
                model.addGenConstrIndicator(z[j], True, s[j] <= 0)

        x_prev = x

    # Output layer (affine)
    w, b = weights[-1]
    nx, nu = w.shape  # Number of inputs to layer and number of units in layer
    for i in range(nu):
        model.addConstr(gp.quicksum(w[k, i] * x_prev[k] for k in range(nx)) + b[i], gp.GRB.EQUAL, output_vars[i])
    model.update()

    return model


def build_milp_and_run_bt(model: gp.Model, input_vars, output_vars, state_dict, c_id=0, bt_procedures=None):
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
    weights = get_weights(state_dict, multilayer = True)
    print(f'these are the shape of the weights: {np.shape(weights)}')

    # Tighten bounds
    input_bounds = [(v.getAttr(gp.GRB.Attr.LB), v.getAttr(gp.GRB.Attr.UB)) for v in input_vars]
    output_bounds = [(v.getAttr(gp.GRB.Attr.LB), v.getAttr(gp.GRB.Attr.UB)) for v in output_vars]
    bounds = run_bound_tightening(weights, input_bounds, output_bounds, bt_procedures)

    # print("Bounds of", nn_model.name)
    # print(f'there are the boudns {bounds}')

    # Build NN constraints
    hidden_bounds = bounds[1:-1]
    # print(f'these are the hidden bounds {hidden_bounds}')
    nn_milp_builder(model, input_vars, output_vars, weights, hidden_bounds, c_id)
    model.update()
    return model
