"""
2. For a given set of training data examples stored in a .CSV file, implement and demonstrate
the Candidate-Elimination algorithm to output a description of the set of all hypotheses
consistent with the training examples.


CANDIDATE-ELIMINATION Learning Algorithm

The CANDIDATE-ELIMINTION algorithm computes the version space containing all
hypotheses from H that are consistent with an observed sequence of training examples.
Initialize G to the set of maximally general hypotheses in H
Initialize S to the set of maximally specific hypotheses in H
For each training example d, do

• If d is a positive example
• Remove from G any hypothesis inconsistent with d
• For each hypothesis s in S that is not consistent with d
• Remove s from S
• Add to S all minimal generalizations h of s such that
• h is consistent with d, and some member of G is more general than h
• Remove from S any hypothesis that is more general than another hypothesis in S

If d is a negative example
• Remove from S any hypothesis inconsistent with d
• For each hypothesis g in G that is not consistent with d
• Remove g from G
• Add to G all minimal specializations h of g such that
• h is consistent with d, and some member of S is more specific than h
• Remove from G any hypothesis that is less general than another hypothesis in G

CANDIDATE- ELIMINTION algorithm using version spaces

"""


"""
The iloc indexer for Pandas Dataframe is used for integer-location 
based indexing / selection by position.
# Single selections using iloc and DataFrame
# Rows:
data.iloc[0] # first row of data frame (Aleshia Tomkiewicz) - Note a Series data type output.
data.iloc[1] # second row of data frame (Evan Zigomalas)
data.iloc[-1] # last row of data frame (Mi Richan)
# Columns:
data.iloc[:,0] # first column of data frame (first_name)
data.iloc[:,1] # second column of data frame (last_name)
data.iloc[:,-1] # last column of data frame (id)

# Multiple row and column selections using iloc and DataFrame
data.iloc[0:5] # first five rows of dataframe
data.iloc[:, 0:2] # first two columns of data frame with all rows
data.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
data.iloc[0:5, 5:8] # first 5 rows and 5th, 6th, 7th columns of data frame (county -> phone1).


Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object. This enumerate object can then be used directly in 
for loops or be converted into a list of tuples using list() method.
"""
import numpy as np 
import pandas as pd

data = pd.read_csv('enjoysport1.csv')
concepts = np.array(data.iloc[:,0:-1])
print(concepts) 
target = np.array(data.iloc[:,-1])  
print(target)
def learn(concepts, target): 
    specific_h = concepts[0].copy()     
    print("initialization of specific_h and general_h")     
    print(specific_h)  
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]     
    print(general_h)  

    for i, h in enumerate(concepts):
        print("For Loop Starts")
        if target[i] == "yes":
            print("If instance is Positive ")
            for x in range(len(specific_h)): 
                if h[x]!= specific_h[x]:                    
                    specific_h[x] ='?'                     
                    general_h[x][x] ='?'
                   
        if target[i] == "no":            
            print("If instance is Negative ")
            for x in range(len(specific_h)): 
                if h[x]!= specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        

        print(" steps of Candidate Elimination Algorithm",i+1)        
        print(specific_h)         
        print(general_h)
        print("\n")
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        general_h.remove(['?', '?', '?', '?', '?', '?']) 
    return specific_h, general_h 

s_final, g_final = learn(concepts, target)

print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")
