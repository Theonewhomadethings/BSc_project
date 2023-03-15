import numpy as np
from math import comb
import math
import matplotlib.pyplot as plt 

integerN = []

binaryN = []
E_n = []
k1 = []
k2 = []
k3 = []
k4 = []
k5 = []
k6 = []
k7 = []
k8 = []
k9 = []
k10 = []
def main(K, N, stand_dev, delta_t, iteration2):
    e_k = np.random.normal(loc = 0, scale = 1, size = K) #question 1a code, 10 single electron orbital energies chosen accordinbg to a standard normal gaussian 
    #Loc = Mean, Scale = Standard deviation, size = Output shape (10, )
    #print("n  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] N") # debug print statement
    #print("\n")
    numbOfStatesN = comb(K, N) #252 number of possible given states according to binomial distribution where paramaeters K = 10 and N = 5.   
    for n in range(0, 2**K): #for loop here builds all of the 
        n_k = list(f"{n:010b}") #makes the binary representation of all possible states from [0000000000] to [1111111111]
        test_list = list(map(int, n_k)) #integerises the otherwise string list of all possible states, for later code.
        dotP = np.dot(test_list, e_k) #dot product of the binary states list and the 10 single body energies list.
        tempSum = np.sum(test_list) #sums the binary list to get the fermion count. if
        if (tempSum == N): #conditional where we select the lists which have 5 fermions or 5 1s in the binary list
            integerN.append(n) #saves the integer form of the binary list
            binaryN.append(test_list) #saves the binary list in another list to form a list of a list
            E_n.append(dotP) #saves the non interacting many body energies calcauted in the dotp line above in a list
            #print(n, test_list, tempSum) #debug print statement.
            if n == 31: #for question 3
                initial_stateInt = 31  #in question 3 we have selected this integer as the initial state
                initial_stateBin = test_list #in question 3
            
    #print("How many fermions =  ", len(indicesN))
    #print("size of energy array", len(E_n))
    E_n.sort() #sort energy values in ascending order
    H_0 = np.zeros(shape = (numbOfStatesN, numbOfStatesN))
    np.fill_diagonal(H_0, val = E_n)
   # print(H_0)
    wMatrix = np.random.normal(loc = 0, scale = stand_dev, size = (numbOfStatesN, numbOfStatesN))
    wMatrixT = np.transpose(wMatrix)
   # print("W matrix =", wMatrix)
    #print("W transpose m = ", wMatrixT)
    wPrimeT = np.add(wMatrix, wMatrixT)
    wPrime = np.divide(wPrimeT, 2)  #(wMatrix + wMatrix) / 2
   # print("wPrime = ", wPrime)
 #   print(np.shape(wPrime))
   # print(len(wMatrix))
    H = np.add(H_0, wPrime)

    eVal, eVect = np.linalg.eig(H)
    #print("These are the eigenvalues: ", w)
    #print("This is the shape of the eigenvalues", np.shape(w))
    eD = np.zeros(shape = (numbOfStatesN, numbOfStatesN), dtype= "complex")#empty matrix
   # np.fill_diagonal(D, val = w)
    for i in range(252):
        eD[i][i] = np.exp(-1j*eVal[i]*delta_t)
   # print(eD)
    '''
    w = (252, 252)
    v = (252)
    Note for the eigenvector the column v[:,i] is the eigenvector
1x    corresponding to the eigenvalue w[i]    
    '''
  #  eigenvector1= v[:,1]
   # eigvalue1 = w[1]
    p = eVect
    p_t = np.transpose(p)
    p_dagger = np.conjugate(p_t)
   # evolution_op1 = v*eD*p_t
    evolOPt = np.dot(p, eD) 
    evolOP = np.dot(evolOPt, p_dagger)
    check1 = np.dot(p, p_dagger) #p inverse
   # checkI =  np.linalg.inv(v)
   # print(np.shape(evolOP))
    #checkIfull = np.multiply(v, checkI)
#    print(checkI)
 #   check2 = np.multiply(check1, p) # pinverse * p = I
  #  print(check2)
  #  print(np.shape(evolution_op1)) #DEBUG 2
   # pr
    #print(initial_stateBin, initial_stateInt)
    init_state = np.zeros(shape = (numbOfStatesN))
    init_state[initial_stateInt] = 1
   # for k in range(len(init_state)):
    #    if init_state[k] == 1:
     #       print("there is a 1 present")
    #print(np.shape(init_state))
    #print(evolOP)
    for h in range(iteration2):
      init_state = np.dot(evolOP, init_state) 
    final_state = init_state
   # print(init_state)  
  #  print(np.shape(integerN)) #(252, )
   # print(binaryN) #(252,10)
    v = 9
    while v >= 0:
        u = 0
        while u < 252:
            if v == 0:
                k10.append(binaryN[u][v])
                u = u + 1
            if v == 1:
                k9.append(binaryN[u][v])
                u = u + 1
            if v == 2:
                k8.append(binaryN[u][v])
                u = u + 1
            if v == 3:
                k7.append(binaryN[u][v])
                u = u + 1
            if v == 4:
                k6.append(binaryN[u][v])
                u = u + 1
            if v == 5:
                k5.append(binaryN[u][v])
                u = u + 1
            if v == 6:
                k4.append(binaryN[u][v])
                u = u + 1
            if v == 7:
                k3.append(binaryN[u][v])
                u = u + 1
            if v == 8:
                k2.append(binaryN[u][v])
                u = u + 1
            if v == 9:
                k1.append(binaryN[u][v])
                u = u + 1
        v = v - 1
    #print(np.shape(final_state))
#    final_state = np.absolute(final_state)
#build the k = 1 case first 
    sum1 = 0
    sum2 = 0
    sum3 = 0 
    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0
    sum8 = 0
    sum9 = 0
    sum10 = 0
    k1Contents = []
    k2Contents = []
    k3Contents = []
    k4Contents = []
    k4Contents = []
    k5Contents = []
    k6Contents = []
    k7Contents = []
    k8Contents = []
    k9Contents = []
    k10Contents = []
    expValues = []
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum1 = sum1 + ((absSquare)*(k1[e]))     #expectation value of k = 1 case
      k1Contents.append(sum1)
    expValues.append(sum1)  
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum2 = sum2 + ((absSquare)*(k2[e])) #expectation value of k = 2 case 
      k2Contents.append(sum2)
    expValues.append(sum2)  
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum3 = sum3 + ((absSquare)*(k3[e]))
      k3Contents.append(sum3)
    expValues.append(sum3)           #expectation value of k = 3 case
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum4 = sum4 + ((absSquare)*(k4[e])) #expectation value of k = 4 case
      k4Contents.append(sum4)
    expValues.append(sum4)  
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum5 = sum5 + ((absSquare)*(k5[e])) #expectation value of k = 5 case
      k5Contents.append(sum5)
    expValues.append(sum5)
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum6 = sum6 + ((absSquare)*(k6[e])) #expectation value of k = 6 case
      k6Contents.append(sum6)
    expValues.append(sum6)  
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum7 = sum7 + ((absSquare)*(k7[e])) #expectation value of k = 7 case
      k7Contents.append(sum7)
    expValues.append(sum7)  
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum8 = sum8 + ((absSquare)*(k8[e])) #expectation value of k = 8 case
      k8Contents.append(sum8)
    expValues.append(sum8)
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum9 = sum9 + ((absSquare)*(k9[e])) #expectation value of k = 9 case
      k9Contents.append(sum9)
    expValues.append(sum9) 
    e = 0
    for e in range(252):
      absSquare = (np.absolute( final_state[e] ))**2
      sum10 = sum10 + ((absSquare)*(k10[e])) #expectation value of k = 10 case
      k10Contents.append(sum10)
    expValues.append(sum10)    
    print("This is sum10: ", sum10)
    stepper = (delta_t*1000) / 252
    time = np.arange(delta_t, 1000*delta_t, step = stepper)
   # time = np.linspace(delta_t, 1000*delta_t, num = 10 )
    #print(k1Contents)

   #  time = np.arange(delta_t, 1000*delta_t)
    print("<n_k array> array:",expValues)
    print("time:", np.shape(time))
    print("energy:", e_k)
 #   print(stepper)
    plt.title("Q4 plot: Fitting the numerical results with the Fermi Dirac distribution") 
    plt.xlabel("e_k (single electron orbital energies of a multi electron atom)")
  #  plt.xlabel("time (s)")
    plt.ylabel("<n_k>  expectation values")
  #  plt.plot(np.absolute(e_k), expValues)
   # plt.plot(time, k1Contents, label = "K = 1")
  #  plt.plot(time, k2Contents, label = "K = 2")
  #  plt.plot(time, k3Contents, label = "K = 3")
  #  plt.plot(time, k4Contents, label = "K = 4")
  #  plt.plot(time, k5Contents, label = "K = 5")
   # plt.plot(time, k6Contents, label = "K = 6")
  #  plt.plot(time, k7Contents, label = "K = 7")
   # plt.plot(time, k8Contents, label = "K = 8")
   # plt.plot(time, k9Contents, label = "K = 9")
   # plt.plot(time, k10Contents, label = "K = 10")
   # plt.legend()
    plt.show()
main(K=10, N = 5, stand_dev = 0.05, delta_t=0.01, iteration2 = 1000)

