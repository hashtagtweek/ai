import pandas as pd 
#from pandas import DataFrame
csvfile=pd.read_csv('weatherid3.csv')
df_tennis=pd.DataFrame(csvfile)  
df_tennis
 
def entropy(probs):  # Calulate the Entropy of given probability     
    import math     
    return sum( [-prob*math.log(prob, 2) for prob in probs] )      

def entropy_of_list(a_list): # Entropy calculation of list of discrete val ues (YES/NO)     
    from collections import Counter     
    cnt = Counter(x for x in a_list)
    print("No and Yes Classes:",a_list.name,cnt)     
    num_instances = len(a_list)*1.0     
    probs = [x / num_instances for x in cnt.values()]     
    return entropy(probs) # Call Entropy: 

# The initial entropy of the YES/NO attribute for our dataset. 
print(df_tennis['PlayTennis']) 
total_entropy = entropy_of_list(df_tennis['PlayTennis']) 
print("Entropy of given PlayTennis Data Set:",total_entropy)

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):     
    print("Information Gain Calculation of ",split_attribute_name)
    '''     
    Takes a DataFrame of attributes,and quantifies the entropy of a target 
    attribute after performing a split along the values of another attribute.   
    '''          
    # Split Data by Possible Vals of Attribute:     
    df_split = df.groupby(split_attribute_name)     
    #print(df_split.groups)     
    for name,group in df_split:         
        print(name)        
        print(group)          
        # Calculate Entropy for Target Attribute, as well as     
        # Proportion of Obs in Each Data-Split     
    nobs = len(df.index) * 1.0     
        #print("NOBS",nobs)     
    df_agg_ent= df_split.agg({target_attribute_name : 
            [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]     
    #print("DFAGGENT",df_agg_ent)     
    df_agg_ent.columns = ['Entropy', 'PropObservations']     
        #if trace: # helps understand what fxn is doing:      
        #print(df_agg_ent)          
        # Calculate Information Gain:     
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )     
    old_entropy = entropy_of_list(df[target_attribute_name]) 
    return old_entropy - new_entropy 
print('Info-gain for Outlook is :'+str( information_gain(df_tennis, 'Outlook', 'PlayTennis')),"\n") 
print('\n Info-gain for Humidity is: ' + str( information_gain(df_tennis, 'Humidity', 'PlayTennis')),"\n") 
print('\n Info-gain for Wind is:' + str( information_gain(df_tennis, 'Wind', 'PlayTennis')),"\n") 
print('\n Info-gain for Temperature is:' + str( information_gain(df_tennis , 'Temperature','PlayTennis')),"\n") 

def id3(df, target_attribute_name, attribute_names, default_class=None):   
       ## Tally target attribute:     
       from collections import Counter     
       cnt = Counter(x for x in df[target_attribute_name])# class of YES /NO          
       ## First check: Is this split of the dataset homogeneous?     
       if len(cnt) == 1:         
           return next(iter(cnt))          
       ## Second check: Is this split of the dataset empty?     
       # if yes, return a default value     
       elif df.empty or (not attribute_names): return default_class           
       ## Otherwise: This dataset is ready to be divvied up!     
       else:         
           # Get Default Value for next recursive call of this function:         
           default_class = max(cnt.keys()) #[index_of_max] # most common valu e of target attribute in dataset                  
           # Choose Best Attribute to split on:         
           gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]         
           index_of_max = gainz.index(max(gainz))          
           best_attr = attribute_names[index_of_max]                  
           # Create an empty tree, to be populated in a moment         
           tree = {best_attr:{}}         
           remaining_attribute_names = [i for i in attribute_names if i != best_attr]   
           # Split dataset         
           # On each split, recursively call this algorithm.         
           # populate the empty tree with subtrees, which         
           # are the result of the recursive call 
           for attr_val, data_subset in df.groupby(best_attr):             
               subtree = id3(data_subset,target_attribute_name,remaining_attribute_names,default_class)
               tree[best_attr][attr_val] = subtree        
           return tree 
       
# Get Predictor Names (all but 'class') 
attribute_names = list(df_tennis.columns) 
print("List of Attributes:", attribute_names)  
attribute_names.remove('PlayTennis')  #Remove the class attribute  
print("Predicting Attributes:", attribute_names) 

# Run Algorithm: 
from pprint import pprint 
tree = id3(df_tennis,'PlayTennis',attribute_names) 
print("\n\nThe Resultant Decision Tree is :\n") 
pprint(tree)
