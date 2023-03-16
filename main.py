
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
    init_state[integerN[40]] = 1
   # for k in range(len(init_state)):
    #    if init_state[k] == 1:
     #       print("there is a 1 present")
    #print(np.shape(init_state))
    #print(evolOP)
    '''
    need to build a 2d Matrix of 
    iterations[kvalues = 10, 9, 8, 7, 6, 5, 4, 3, 2,   1]

    iteration1[<n_k=10>, <n_k=9>,                  <n_1>]
    iteration2[<n_k=10>, <n_k=9>,                  <n_1>]
    iteration1000[<n_k=10>, <n_k=9>,               <n_1>]
    '''
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

    dataY = [[0 for col in range(10)] for row in range(1000)]
    print(np.shape(dataY))
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
    state = init_state 
    for h in range(iteration2):
      state = np.dot(evolOP, state)
      prob = np.absolute(state)**2
      for k in range(1, 11):
          if h == 1:
             
            print(k)
          #sum = 0
          #for m in range(1, numbOfStatesN):
               #binaryN, current k level
          if k == 1:
            n_kExpVal = np.dot(prob, k1)
            dataY[h][0] = n_kExpVal
            plt.plot(h, dataY[h][0])
          if k == 2:
            n_kExpVal = np.dot(prob, k2)
            dataY[h][1] = n_kExpVal

          if k == 3:
            n_kExpVal = np.dot(prob, k3)
            dataY[h][2] = n_kExpVal
          if k == 4:
            n_kExpVal = np.dot(prob, k4)
            dataY[h][3] = n_kExpVal
          if k == 5:
            n_kExpVal = np.dot(prob, k5)
            dataY[h][4] = n_kExpVal
          if k == 6:
            n_kExpVal = np.dot(prob, k6)
            dataY[h][5] = n_kExpVal
          if k == 7:
            n_kExpVal = np.dot(prob, k7)
            dataY[h][6] = n_kExpVal
          if k == 8:
            n_kExpVal = np.dot(prob, k8)
            dataY[h][7] = n_kExpVal
          if k == 9:
            n_kExpVal = np.dot(prob, k8)
            dataY[h][8] = n_kExpVal
          if k == 10:
            n_kExpVal = np.dot(prob, k10)
            dataY[h][9] = n_kExpVal
          
          #print(n_kExpVal, h)
   #  sum = sum + np.dot(prob, n_kBin)
    k1Vals = [row[0] for row in dataY]
    k2Vals = [row[1] for row in dataY]
    k3Vals = [row[2] for row in dataY]
    k4Vals = [row[3] for row in dataY]
    k5Vals = [row[4] for row in dataY]
    k6Vals = [row[5] for row in dataY]
    k7Vals = [row[6] for row in dataY]
    k8Vals = [row[7] for row in dataY]
    k9Vals = [row[8] for row in dataY]
    k10Vals = [row[9] for row in dataY]
    #print(np.shape(k1Vals))
    t = np.linspace(1, 1000, num = 1000, dtype=int)
    for row in dataY[:5]:
        print(row) 
    plt.title("New plot for q3.")
    plt.xlabel("t")
    plt.ylabel("<n_k>")
    plt.plot(t, k1Vals, label = "k1")
    plt.plot(t, k2Vals, label = "k2")
    plt.plot(t, k3Vals, label = "k3")
    plt.plot(t, k4Vals, label = "k4")
    plt.plot(t, k5Vals, label = "k5")
    plt.plot(t, k6Vals, label = "k6")
    plt.plot(t, k7Vals, label = "k7")
    plt.plot(t, k8Vals, label = "k8")
    plt.plot(t, k9Vals, label = "k9")
    plt.plot(t, k10Vals, label = "k10")
    #plt.ylim(0, 1.2)
    plt.legend()
    plt.show()
#main(K=10, N = 5, stand_dev = 0.05, delta_t=0.01, iteration2 = 1000)
main(K=10, N = 5, stand_dev = 0.03, delta_t=0.2, iteration2 = 1000)


