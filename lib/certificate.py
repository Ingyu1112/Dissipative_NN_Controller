import cvxpy as cp
import numpy as np

def cal_param(m_p, l_p, i_p, m_c):
    l_c = l_p / 2.0

    a = m_c + m_p
    b = m_p*l_c
    c = m_p * l_c**2 + i_p
        
    m1 = ( a + c + np.sqrt((a-c)**2+4*b**2) ) / 2
    m2 = ( a + c - np.sqrt((a-c)**2+4*b**2) ) / 2
    c2 = m_p * l_c
    g = c2 * 9.81

    return m1, m2, c2, g

def cal_e_prev(m1,m2,c,dq_m,g):
    a = m2**2/m1
    b = 1/2*c*dq_m
    c = g**2
    e = 1

    while (1):
        e_next = (b/a*e + c/a) ** (1/3)
        if abs((e_next-e)/e) <= 1e-4:
            break
        e = e_next

    return e_next

    
def cal_e(m1,m2,c,dq_m,g):
    a = m2
    b = 1/2*c*dq_m
    c = g**2
    e = 1

    while (1):
        e_next = (b/a*e + c/a) ** (1/3)
        if abs((e_next-e)/e) <= 1e-4:
            break
        e = e_next

    return e_next

def cal_q_matrix(m1, m2, c, dq_m, g, e):
    q1 = 0.5*e*c*dq_m + g**2
    q2 = e*(m2 + 0.5*c*dq_m)

    return q1, q2

def qsr_plant_var(q_dim, dq_dim, e, q1, q2, g):
    K = cp.Variable((q_dim,q_dim), symmetric=True)
    D = cp.Variable((dq_dim,dq_dim), symmetric=True)

    q_11 = -e*K + q1*np.eye(q_dim)
    q_21 = -0.5*e*D
    q_22 = -D + q2*np.eye(dq_dim)
    q_31 = -0.5*e*np.eye(q_dim)
    q_32 = -0.5*np.eye(dq_dim)
    q_33 = -np.eye(dq_dim)
    
    s_11 = .5*e*np.eye(q_dim)
    s_12 = g**2*np.eye(q_dim)
    s_21 = .5*np.eye(q_dim)
    s_22 = np.zeros((2,2))
    s_31 = np.zeros((2,2))
    s_32 = np.zeros((2,2))

    r_11 = np.zeros((2,2))
    r_21 = np.zeros((2,2))
    r_22 = g**2*np.eye(q_dim)

    q_plant = cp.bmat([[q_11, q_21.T, q_31.T],
                       [q_21, q_22,   q_32.T],
                       [q_31, q_32,   q_33]])
    s_plant = cp.bmat([[s_11, s_12],
                       [s_21, s_22],
                       [s_31, s_32]])
    r_plant = cp.bmat([[r_11, r_21.T],
                       [r_21, r_22]])
    
    return K, D, q_plant, s_plant, r_plant

def qsr_resnet_var(layer_dim):
    q_linear = make_cvxpy_block_diag([cp.Variable((layer_dim[-1],layer_dim[-1]), symmetric=True)] + 
        [cp.Variable((layer_dim[i],layer_dim[i]), symmetric=True) for i in range(1, len(layer_dim)-1)])
    s_linear = cp.Variable((np.sum(layer_dim[1:]),np.sum(layer_dim[:-1])))
    r_linear = make_cvxpy_block_diag([cp.Variable((layer_dim[i],layer_dim[i]), symmetric=True) for i in range(len(layer_dim)-1)])
    # r_linear = make_cvxpy_block_diag([
    #     cp.Variable((4,4), symmetric=True),
    #     cp.Variable((20,20), symmetric=True),
    #     cp.Variable((10,10), symmetric=True)
    # ])

    lambda_activation = cp.Variable((1))
    q_activation = -lambda_activation*np.eye(np.sum(layer_dim[1:-1]))
    s_activation = .5*lambda_activation*np.eye(np.sum(layer_dim[1:-1]))
    r_activation = 0*lambda_activation*np.eye(np.sum(layer_dim[1:-1]))

    W_cp = cp.bmat([
        [np.zeros((2,4)),     np.zeros((2,20)),     cp.Variable((2,10))],
        [cp.Variable((20,4)), np.zeros((20,20)),    np.zeros((20,10))],
        [np.zeros((10,4)),    cp.Variable((10,20)), np.zeros((10,10))]
    ])

    return (q_linear, s_linear, r_linear,
            q_activation, s_activation, r_activation,
            W_cp)

def qsr_network_var(
    q_plant, s_plant, r_plant,
    q_linear, s_linear, r_linear,
    q_activation, s_activation, r_activation, H
):
    q_diag = make_cvxpy_block_diag( [q_plant, q_linear, q_activation] )
    s_diag = make_cvxpy_block_diag( [s_plant, s_linear, s_activation] )
    r_diag = make_cvxpy_block_diag( [r_plant, r_linear, r_activation] )

    q_network = q_diag + s_diag@H + H.T@s_diag.T + H.T@r_diag@H

    return q_network, q_diag, s_diag, r_diag

def set_constraint(e, m1, m2, q_network,
    K, D, q_plant, s_plant, r_plant,
    q_linear, s_linear, r_linear,
    q_activation, s_activation, r_activation, tol = -1e-7
):
    pre_constraint = {
        "network":  [q_network << tol*np.eye(q_network.shape[0])],
        "plant":    [q_plant << tol*np.eye(q_plant.shape[0])],
        "linear":   [q_linear << tol*np.eye(q_linear.shape[0])],
        "activation": [q_activation << np.zeros(q_activation.shape)],
        "K" : [K >> e**2*m2**2/m1*np.eye(K.shape[0])],
        "D" : [D >> 0]
    }
    return pre_constraint



def make_cvxpy_block_diag(block_list):
    l = len(block_list)
    bmat_grid = []
    
    for i in range(l):
        row = []
        for j in range(l):
            if i == j:
                row.append(block_list[i])
            else:
                r = block_list[i].shape[0]
                c = block_list[j].shape[1]
                row.append(np.zeros((r, c)))
        bmat_grid.append(row)
        
    return cp.bmat(bmat_grid)