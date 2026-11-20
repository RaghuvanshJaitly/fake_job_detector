import numpy as np

# suspicious phrases
X = [[0], [1], [2],[3], [4]]
#labels
y = [0, 0, 1, 0, 1]

#slope
#before first epoch: w = [0]
#before epoch 1: w = [0.2]
w = [0] * len(X[0])

#intercept
#before first epoch: b = 0
#before epoch 1 : b = -0.1
b = 0
learning_rate = 1
#before first epoch: prev_loss = None
#before epoch 1: prev_loss = 0.693
prev_loss = None
#constant 
tolerance = 0.000001

for epoch in range(0,1000):
    #first epoch (0)
    #first epoch: loss_total = 0
    #epoch (1) = loss_total = 0
    loss_total = 0
    #first epoch: intercept_total = 0
    #epoch (1): intercept_total = 0
    intercept_total = 0
    #first epoch: slope_contribution = 0
    #epoch (1): slope_contribution = 0
    slope_contribution = 0

    #first inner loop: x_val = [0], label = 0
    #inner loop iter2 (epoch0): x_val = [1], label = 0
    #inner loop iter3 (epoch 0): x_val = [2], label = 1
    #inner loop iter4 (epoch0): x_val = [3], label = 0
    #inner loop iter5 (epoch0): x_val = [4], label = 1
    
    #inner loop iter1 (epoch1): x_val = [0], label = 0
    #inner loop iter2 (epoch1): x_val = [1], label = 0
    #inner loop iter3 (epoch1): x_val = [2], label = 1
    #inner loop iter4 (epoch1): x_val = [3], label = 0
    #inner loop iter5 (epoch1): x_val = [4], label = 1
    for x_val, label in zip(X,y):
        #iter1 (inner loop): z = 0 + (0 * 0) = 0
        #iter2 (inner loop): z = 0 + (0 * 1) = 0
        #iter3 (inner loop): z = 0 + (0 * 2) = 0
        #iter4 (inner loop): z = 0 + (0 * 3) = 0
        #iter5 (inner loop): z = 0 + (0 * 4) = 0
        
        #iter1 (inner loop epoch1): z = -0.1 + (0.2 * 0) = -0.1
        #iter2 (inner loop epoch1): z = -0.1 + (0.2 * 1) = 0.1
        #iter3 (inner loop epoch1): z = -0.1 + (0.2 * 2) = 0.3
        #iter4 (inner loop epoch1): z = -0.1 + (0.2 * 3) = 0.5
        #iter5 (inner loop epoch1): z = -0.1 + (0.2 * 4) = 0.7
        z = b + (w[0] * x_val[0])
        #iter1 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter2 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter3 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter4 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        #iter5 (inner loop): p_fraud = 1/(1 + e^(-0)) = 0.5
        
        #iter1 (inner loop epoch1): p_fraud = 1/(1 + e^(-(-0.1))) = 0.475
        #iter2 (inner loop epoch1): p_fraud = 1/(1 + e^(-(0.1))) = 0.525
        #iter3 (inner loop epoch1): p_fraud = 1/(1 + e^(-(0.3))) = 0.574
        #iter4 (inner loop epoch1): p_fraud = 1/(1 + e^(-(0.5))) = 0.622
        #iter5 (inner loop epoch1): p_fraud = 1/(1 + e^(-(0.7))) = 0.668
        p_fraud = 1/(1 + np.exp(-z))
        
        #iter1 (inner loop): label = 0 so we skip to else
        #iter2 (inner loop): label = 0 so we skip to else
        #iter3 (inner loop): label = 1 so we go to if
        #iter4 (inner loop): label = 0 so we skip to else
        #iter5 (inner loop): label = 1 so we go to if
        
        #iter1 (inner loop epoch1): label = 0 so we skip to else
        #iter2 (inner loop epoch1): label = 0 so we skip to else
        #iter3 (inner loop epoch1): label = 1 so we go to if
        #iter4 (inner loop epoch1): label = 0 so we skip to else
        #iter5 (inner loop epoch1): label = 1 so we go to if
        if label == 1:
            #iter3 (inner loop): label_prob = 0.5
            #iter5 (inner loop): label_prob = 0.5
            
            #iter3 (inner loop epoch1): label_prob = 0.574
            #iter5 (inner loop epoch1): label_prob = 0.668
            label_prob = p_fraud
        else:
            #iter1 (inner loop): label_prob = 1 - 0.5 = 0.5
            #iter2 (inner loop): label_prob = 1 - 0.5 = 0.5
            #iter4 (inner loop): label_prob = 1 - 0.5 = 0.5
            
            #iter1 (inner loop epoch1):label_prob = 1 - 0.475 = 0.525
            #iter2 (inner loop epoch1): label_prob = 1 - 0.525 = 0.475
            #iter4 (inner loop epoch1): label_prob = 1 - 0.622 = 0.378
            label_prob = 1 - p_fraud

        #iter1 (inner loop): error = 0.5 - 0 = 0.5
        #iter2 (inner loop): error = 0.5 - 0 = 0.5
        #iter3 (inner loop): error = 0.5 - 1 = -0.5
        #iter4 (inner loop): error = 0.5 - 0 = 0.5
        #iter5 (inner loop): error = 0.5 - 1 = -0.5
        
        #iter1 (inner loop epoch1): error = 0.475 - 0 = 0.475
        #iter2 (inner loop epoch1): error = 0.525 - 0 = 0.525
        #iter3 (inner loop epoch1): error = 0.574 - 1 = -0.426
        #iter4 (inner loop epoch1): error = 0.622 - 0 = 0.622
        #iter5 (inner loop epoch1): error = 0.668 - 1 = -0.332
        error = p_fraud - label
        #iter1 (inner loop): row_loss = -log(0.5) = 0.693
        #iter2 (inner loop): row_loss = -log(0.5) = 0.693
        #iter3 (inner loop): row_loss = -log(0.5) = 0.693
        #iter4 (inner loop): row_loss = -log(0.5) = 0.693
        #iter5 (inner loop): row_loss = -log(0.5) = 0.693
        
        #iter1 (inner loop epoch1): row_loss = -log(0.525) = 0.644
        #iter2 (inner loop epoch1): row_loss = -log(0.475) = 0.744
        #iter3 (inner loop epoch1): row_loss = -log(0.574) = 0.555
        #iter4 (inner loop epoch1): row_loss = -log(0.378) = 0.973
        #iter5 (inner loop epoch1): row_loss = -log(0.668) = 0.404
        row_loss = -np.log(label_prob)
        #iter1 (inner loop): starts as loss_total = 0
        #iter1 (inner loop) : loss_total = loss_total + row_loss = 0 + 0.693 = 0.693
        #iter2 (inner loop): loss_total = loss_total + row_loss = 0.693 + 0.693 = 1.386
        #iter3 (inner loop): loss_total = loss_total + row_loss = 1.386 + 0.693 = 2.079
        #iter4 (inner loop): loss_total = loss_total + row_loss = 2.079 + 0.693 = 2.772
        #iter5 (inner loop): loss_total = loss_total + row_loss = 2.772 + 0.693 = 3.465
        
        #iter1 epoch 1 (inner loop): starts as loss_total = 0
        #iter1 (inner loop epoch1): loss_total = loss_total + row_loss = 0 + 0.644 = 0.644
        #iter2 (inner loop epoch1): loss_total = loss_total + row_loss = 0.644 +0.744 = 1.388
        #iter3 (inner loop epoch1): loss_total = loss_total + row_loss = 1.388 + 0.555 = 1.943
        #iter4 (inner loop epoch1): loss_total = loss_total + row_loss = 1.943 + 0.973 = 2.916
        #iter5 (inner loop epoch1): loss_total = loss_total + row_loss = 2.916 + 0.404 = 3.320
        loss_total += row_loss
        #iter1 (inner loop): starts as intercept_total = 0
        #iter1 (inner loop): intercept_total = intercept_total + error = 0 + 0.5 = 0.5
        #iter2 (inner loop): intercept_total = intercept_total + error = 0.5 + 0.5 = 1
        #iter3 (inner loop): intercept_total = intercept_total + erorr = 1 + (-0.5) = 0.5
        #iter4 (inner loop): intercept_total = intercept_total + erorr = 0.5 + 0.5 = 1
        #iter5 (inner loop): intercept_total = intercept_total + erorr = 1 + (-0.5) = 0.5
        
        #iter1 (inner loop epoch1): intercept_total = intercept_total + erorr = 0 + 0.475 = 0.475
        #iter2 (inner loop epoch1): intercept_total = intercept_total + erorr = 0.475 + 0.525 = 1
        #iter3 (inner loop epoch1): intercept_total = intercept_total + erorr = 1 + -0.426 = 0.574
        #iter4 (inner loop epoch1): intercept_total = intercept_total + erorr = 0.574 + 0.622 = 1.196
        #iter4 (inner loop epoch1): intercept_total = intercept_total + erorr = 1.196 + (-0.332) = 0.864
        intercept_total += error 
        #iter1 (inner loop): starts as slope_contribution = 0
        #iter1 (inner loop): slope_contribution = slope_contribution + error * x_val[0] = 0.5 * 0 = 0
        #iter2 (inner loop): slope_contribution = slope_contribution + erorr * x_val[0] = 0 + 0.5 * 1 = 0.5
        #iter3 (inner loop): slope_contribution = slope_contribution + erorr * x_val[0] = 0.5 + (-0.5) * 2 = -0.5
        #iter4 (inner loop): slope_contribution = slope_contribution + erorr * x_val[0] = -0.5 + 0.5 * 3 = 1
        #iter5 (inner loop): slope_contribution = slope_contribution + erorr * x_val[0] = 1 + (-0.5) * 4 = -1
        
        #iter1 (inner loop epoch1): slope_contribution = slope_contribution + erorr * x_val[0] = 0 + 0.475 * 0 = 0
        #iter2 (inner loop epoch1): slope_contribution = slope_contribution + erorr * x_val[0] = 0 + 0.525 * 1 = 0.525
        #iter3 (inner loop epoch1): slope_contribution = slope_contribution + erorr * x_val[0] = 0.525 + (-0.426) * 2 = -0.327
        #iter4 (inner loop epoch1): slope_contribution = slope_contribution + erorr * x_val[0] = -0.327 + 0.622 * 3 = 1.539
        #iter5 (inner loop epoch1): slope_contribution = slope_contribution + erorr * x_val[0] = 1.539 + (-0.332) * 4 = 0.211
        slope_contribution += error * x_val[0]

    #epoch 0: intercept_gradient = 0.5/5 = 0.1
    #epoch 1: intercept_gradient = 0.864/5 = 0.1728
    intercept_gradient = intercept_total/len(X)
    #epoch 0: slope_gradient = -1/5 = -0.2
    #epoch 1: slope_gradient = 0.211/5 = 0.0422
    slope_gradient = slope_contribution/len(X)
    #epoch 0: avg_loss = 3.465/5 = 0.693
    #epoch 0: avg_loss = 3.320/5 = 0.664
    avg_loss = loss_total/len(X)
    
    #epoch 0: prev_loss =  None so we skip
    #epoch 1: prev_loss =  0.693 so we enter if
    if prev_loss != None:
        #epoch 1: loss_change = abs(0.693 -  0.664) = 0.029
        loss_change = abs(prev_loss - avg_loss)
        #epoch 1: 0.029 < 0.000001 = False so we continue 
        if loss_change < tolerance:
            break
    # epoch 0: w[0] = 0 - 1 * (-0.2) = 0.2
    # epoch 1: w[0] = 0.2 - 1 * (0.0422) = 0.1578
    w[0] = w[0] - learning_rate * slope_gradient
    #epoch 0: b = 0 - 1 * 0.1 = -0.1
    #epoch 1: b = -0.1 - 1 * 0.1728 = -0.2728
    b = b - learning_rate * intercept_gradient
    #epoch 0: prev_loss = 0.693
    #epoch 0: prev_loss = 0.664
    prev_loss = avg_loss
    #epoch 0 ends, go back up to epoch 1
    
    

z = b + (w[0] * 1)
p = 1/(1 + np.exp(-z))

