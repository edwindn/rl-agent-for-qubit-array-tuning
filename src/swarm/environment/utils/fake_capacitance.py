import numpy as np


def fake_capacitance_model(n_steps, max_steps, cgd, alpha=0.10, beta=0.01):

    # Create base error matrix (zeros)
    error = np.zeros((cgd.shape[0], cgd.shape[1]))
    
    # Create mask for one-off diagonal elements
    one_off_mask = np.eye(cgd.shape[0], cgd.shape[1], k=1) + np.eye(cgd.shape[0], cgd.shape[1], k=-1)
    
    # Create mask for two-off diagonal elements  
    two_off_mask = np.eye(cgd.shape[0], cgd.shape[1], k=2) + np.eye(cgd.shape[0], cgd.shape[1], k=-2)
    
    # Apply different error patterns
    base_std = beta + alpha * (1 - n_steps / max_steps)

    
    error[one_off_mask == 1] = np.random.normal(0, base_std, int(np.sum(one_off_mask)))
    error[two_off_mask == 1] = np.random.normal(0, base_std * 0.5, int(np.sum(two_off_mask)))
    
    pred_cgd = np.clip(cgd + error, 0, 1)

    return pred_cgd
