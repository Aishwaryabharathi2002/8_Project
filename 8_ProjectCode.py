

import numpy as np
import scipy.linalg

# Weather Problem
states= {
         0 :  "Rainy",
         1 : "Sunny",
         2 : "Cloudy"}

A= np.array([[ 0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.4, 0.2, 0.4]])
print(A)

def Finding_Probability(List, A, pi):
    initial_state= List[0]
    probability= pi[initial_state]
    prev_state, current_state= initial_state, initial_state
    for i in range(1, len(List)):
        current_state= List[i]
        probability*= A[prev_state][current_state]
        prev_state= current_state
    return probability


#Random Walk on Markov Chain

n=20
initial_state= 0
current_state= initial_state
print(states[current_state], "--->", end=" ")

while n-1:
    current_state= np.random.choice([0,1,2], p= A[current_state])
    print(states[current_state], "--->", end=" ")
    n-=1
print("Done")

# Method-1:
# Finding Left Eigen Vectors
print("Method-1 \n")
print("Finding Left EigenVectors\n")
values, left = scipy.linalg.eig(A, right = False, left = True)

print(" Left Eigenvectors = \n", left, "\n")

print("Eigenvalues= \n", values, "\n")

pi = left[:,0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
print(pi_normalized,"\n")

# Method-2
# Repeated Matrix Multiplication
print("Method-2")
print("Repeated Matrix Multiplication")
A_s = A
total= 10**5

i=0
while i<total:
    A_s =  np.matmul(A_s, A)
    i+=1

print("A^n = \n", A_s, "\n")
print("π = ", A_s[0], "\n")

#Method-3
# Monte Carlo Method
print("Method-3")
print("Monte Carlo Method (Taking long random walk) ")
total = 10**5
initial_state = 0
current_state = initial_state
pi = np.array([0, 0, 0])
pi[current_state] = 1

i = 0
while i<total:
    current_state = np.random.choice([0,1,2], p=A[current_state])
    pi[current_state]+=1
    i +=1

print("π = ", pi/total)

print(" From all the three methods, we are obtaining stationary probability distributions of the transition matrix of the Markov Chain")
print("We can observe the same values from all 3 methods by validating our process\n")


n=int(input("Enter the no of events"))
L=[]
print("Enter the order of events\n")
for i in range(n):
    a= int(input("Enter a value:"))
    L.append(a)
print("The probability of occurrence of events in the given order is:") 
print(Finding_Probability(L, A, pi_normalized))

