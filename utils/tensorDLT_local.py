import torch
import numpy as np

#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1  = torch.tensor([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=torch.float32).unsqueeze(0)


Aux_M2  = torch.tensor([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=torch.float32).unsqueeze(0)



Aux_M3  = torch.tensor([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=torch.float32).unsqueeze(0)



Aux_M4  = torch.tensor([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=torch.float32).unsqueeze(0)


Aux_M5  = torch.tensor([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=torch.float32).unsqueeze(0)



Aux_M6  = torch.tensor([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=torch.float32).unsqueeze(0)


Aux_M71 = torch.tensor([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=torch.float32).unsqueeze(0)


Aux_M72 = torch.tensor([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=torch.float32).unsqueeze(0)



Aux_M8  = torch.tensor([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=torch.float32).unsqueeze(0)


Aux_Mb  = torch.tensor([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=torch.float32).unsqueeze(0)
########################################################

def solve_DLT(orig_pt4, pred_pt4):
    device = orig_pt4.device
    batch_size = orig_pt4.shape[0]
    orig_pt4 = orig_pt4.unsqueeze(2)
    pred_pt4 = pred_pt4.unsqueeze(2)
    # print(orig_pt4.device)
    # Auxiliary tensors used to create Ax = b equation
    M1_tile = Aux_M1.expand(batch_size, -1, -1).to(device)
    M2_tile = Aux_M2.expand(batch_size, -1, -1).to(device)
    M3_tile = Aux_M3.expand(batch_size, -1, -1).to(device)
    M4_tile = Aux_M4.expand(batch_size, -1, -1).to(device)
    M5_tile = Aux_M5.expand(batch_size, -1, -1).to(device)
    M6_tile = Aux_M6.expand(batch_size, -1, -1).to(device)
    M71_tile = Aux_M71.expand(batch_size, -1, -1).to(device)
    M72_tile = Aux_M72.expand(batch_size, -1, -1).to(device)
    M8_tile = Aux_M8.expand(batch_size, -1, -1).to(device)
    Mb_tile = Aux_Mb.expand(batch_size, -1, -1).to(device)

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = torch.matmul(M1_tile, orig_pt4) # Column 1
    A2 = torch.matmul(M2_tile, orig_pt4) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = torch.matmul(M4_tile, orig_pt4) # Column 4
    A5 = torch.matmul(M5_tile, orig_pt4) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = torch.matmul(M71_tile, pred_pt4) *  torch.matmul(M72_tile, orig_pt4  )# Column 7
    A8 = torch.matmul(M71_tile, pred_pt4) *  torch.matmul(M8_tile, orig_pt4  )# Column 8

    # tmp = tf.reshape(A1, [-1, 8])  #batch_size * 8
    # A_mat: batch_size * 8 * 8          A1-A8相当�?*8中的每一�?
    A_mat = torch.transpose(torch.stack([A1.view(-1,8) ,A2.view(-1,8) , \
                                   A3.view(-1,8) , A4.view(-1,8) , \
                                   A5.view(-1,8) ,A6.view(-1,8) ,\
                                   A7.view(-1,8) ,A8.view(-1,8) ,] ,dim=1),1 ,2) # BATCH_SIZE x 8 (A_i) x 8
    # print('--Shape of A_mat:', A_mat.shape)
    # Form b matrix
    b_mat = torch.matmul(Mb_tile, pred_pt4)
    # print('--shape of b:', b_mat.shape)

    # Solve the Ax = b
    H_8el = torch.linalg.solve(A_mat, b_mat) # BATCH_SIZE x 8.
    # print('--shape of H_8el', H_8el.shape)


    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = torch.ones([batch_size, 1, 1]).to(device)
    H_9el = torch.cat([H_8el ,h_ones] ,dim=1)
    H_flat = H_9el.view(-1 ,9)
    #H_mat = tf.reshape(H_flat ,[-1 ,3 ,3])   # BATCH_SIZE x 3 x 3
    return H_flat

