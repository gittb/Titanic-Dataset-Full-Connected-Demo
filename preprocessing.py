

import pandas as pd
import numpy as np

def dataParser(data):
    # makes the root frame to do input for
    fclean = {
                'Sex': [],
                'Pclass': [],
                'Age': [],
                'Ticket': []
    }
    # one-hot gender
    for i in data['Sex']: # for row in column 'Sex'
        if i == 'male':
            fclean['Sex'].append([0, 1])
        else:
            fclean['Sex'].append([1, 0])
    #print(cleandata['Sex'])
    #----------------------------------------------------------------------------------------------
    # one-hot Pclass
    for i in data['Pclass']:
        if i == 1:
            fclean['Pclass'].append([1, 0, 0])
        elif i == 2:
            fclean['Pclass'].append([0, 1, 0])
        else:
            fclean['Pclass'].append([0, 0, 1])
    #print(fclean['Pclass'])

    # age management
    # note could also do average age per class and sex
    agetot = 0
    agect = 0
    for i in data['Age']:
        try: 
            int(i) # if not int, will fail 
            agetot += i # populates the age tot and count for finding average
            agect +=1
        except:
            pass 

    avgage = int(agetot/agect)
    for i in data['Age']:
        try: # add the actual age
            int(i)
            fclean['Age'].append([i])
        except: #if no age given, add the average age calcd above
            fclean['Age'].append([avgage])
    #print(fclean['Age'])
    #----------------------------------------------------------------------------------------------------------------------------------
    fclean['SibSp'] = [] #assign new key to be a list in the main frame, b/c using '=' no need to predefine key
    for i in data['SibSp']: 
        fclean['SibSp'].append([i])
    #----------------------------------------------------------------------------------------------------------------------------------
    fclean['Parch'] = []
    for i in data['Parch']:
        fclean['Parch'].append([i])
    #---------------------------------------------------------------------------------------------------------------------------------
    for i in data['Ticket']:
        try:
            ticket = int(i) #if the ticket is all numbers, one class. Some have letters and this will fail logic gate
            fclean['Ticket'].append([0])
        except:
            fclean['Ticket'].append([1])
    #---------------------------------------------------------------------------------------------------------------------------------
    fclean['Fare'] = [] 
    for i in data['Fare']:
        fclean['Fare'].append([i])
    #-----------------------------------------------------------------------------------------------------------------------------------
    cabin = []
    for i in data['Cabin']:
        if i is not 'NaN': # another method for weeding out NaN values
            cabin.append([1])
        else:
            cabin.append([0])

    fclean['Cabin'] = cabin # making new 'Cabin' dict key and assigning it to pre made list, just another way to skin the cat
    #----------------------------------------------------------------------------------------------------------------------------------
    elist = []
    for i in data['Embarked']:
        if i not in elist:
            elist.append(i)

    embarked = []
    for i in data['Embarked']:
        buffer = [0] * 4 # creates empty one hot encoded list, initalized to all 0s
        buffer[elist.index(i)] = 1 # uses the literal value from elist to index buffer, to flip one hot 0 to a 1
        embarked.append(buffer)
    #print(embarked)
    fclean['Embarked'] = embarked
    #print(embarked)
    #print(elist)

    """
    for i in fclean:
        print(len(fclean[i]))
    """

    outlist = []
    for i in range(0, len(fclean['Fare'])): #len(fclean['Fare']) finds range to iterate through
        buffer = [] # buffer for each line of data
        for key in fclean: #for each column
            #print(key)
            #print(fclean[key][i])
            
            for j in fclean[key][i]: # for the row of j, pull info from each column
                buffer.append(j)
            
        outlist.append(buffer)
        #print(buffer, len(buffer))

    #print(len(elist))
    return outlist # a list of lists [ [], [], [] ]

def targetsMaker(data):
    targets = []
    for i in data['Survived']:
        # could also do targets.append([int(i)]) instead of if/else gates, int(i) converts to int in case was read as string initially
        if i == 0:
            targets.append([0])
        else:
            targets.append([1])
    
    return targets


def pickleWrite(data, filename):
    import pickle
    if '.dat' not in filename:
        filename = filename + '.dat'
    pickle.dump(data, open(filename, 'wb'), -1)
    print('[+] Successfully saved', filename, '[+]\n')


###########################################################################################
################################ Main sequence ###########################################
# reads in the data
data = pd.read_csv('train.csv', header=0)

# processes all the training data
traindata = dataParser(data) # see def dataParser above

# extracts the targets from the training data
traintargets = targetsMaker(data) # see def targetsMaker above

print('-------------------------------------------------')

# extracts training and validation sets from the training data
training = {
    'X': np.array(traindata[:500]), # use np.array for feeding to tensorflow eventually
    'Y': np.array(traintargets[:500])
}

validation = {
    'X': np.array(traindata[500:]),
    'Y': np.array(traintargets[500:])
}


# reads in the testing data
test_raw = pd.read_csv('test.csv', header=0)
testdata = dataParser(test_raw)
# creates list of testing_ids to assign predictions to
ids = test_raw['PassengerId']

testing = {
    'X': np.array(testdata),
    'ID': ids
}



# saves all the data in .dat, maintaining pythonic list/dict structure, for easy reading into tensorflow scripts
pickleWrite(training, 'training.dat')
pickleWrite(validation, 'validation.dat')
pickleWrite(testing, 'testing.dat')
    






