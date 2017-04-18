from node import Node
import math
import collections
import random

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  #make list of attributes
  attribute_list = []
  for a in examples[0]:
      attribute_list.append(a)
  attribute_list.remove('Class')
  sameClass = True
  for e in range(len(examples)):
    if examples[e]['Class'] != examples[0]['Class']:
      sameClass = False
  attributes_same = True
  for i in attribute_list:
      for x in range(len(examples)):
          if examples[x][i] !=  examples[0][i]:
              attributes_same = False
  classes = [] #compute MODE value of 'class'. this is list of all class values observed in dataset
  for e in range(len(examples)): #populating list
      classes.append(examples[e]['Class'])
  c = collections.Counter(classes).most_common(1) #returns [dictionary] (i.e. list of dictionaries) with key=class, value=frequency
  mode_class = c[0][0]
  if examples == []: #case 1
    tree = Node()
    tree.label = default
    return tree
  elif sameClass == True:
    tree = Node()
    tree.label = examples[0]['Class']
    return tree #change to return a node
  elif attributes_same == True: #all the attributes are the same for this examples set: return mode for class

      tree = Node()
      tree.label = mode_class
      return tree
  else:
      best = chooseAttribute(examples)
      tree = Node()
      tree.attribute = best
      tree.label = mode_class #is this necessary?
      best_values = [] #create list of possible attribute values (for training data, Y/N/? for best voting issue)
      for e in range(len(examples)):
          if examples[e][best] not in best_values:
              best_values.append(examples[e][best])
      for v in best_values:
          new_examples = [] #create example subset for Y/N/? branches
          for e in range(len(examples)):
              if examples[e][best]==v:
                  new_examples.append(examples[e])
          child = ID3(new_examples,mode_class)
          tree.children[v] = child
      return tree

     
  #best <- choose-attribute(attributes, examples)
  #tree <- decision tree with root test best
  #for each value v of best, do:
    #examples 
    #subtree
    #return tree
def chooseAttribute(examples):
    # tempgain is just the value you find from each infogain you try
    # maxgain will always hold the heighest gain found as you infogain each attribute
    maxgain = 0
    tempgain = 0

    # bestattribute is the one that has the heighest gain so far
    # checkattribute is just to hold the current attribute being checked
    bestattribute = None
    checkattribute = None

    # goes through the attributes
    listOfAttributes = examples[0].keys()
    listOfAttributes.remove("Class")

    for key in listOfAttributes:

        # sets the current attribute
        checkattribute = key

        tempgain = infoGain(examples, checkattribute)

        # adjust the bestgain and bestattribute if the current attribute's
        # infogain being checked is higher
        if tempgain > maxgain:
            maxgain = tempgain
            bestattribute = checkattribute

    return bestattribute

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''

  #initialize counters to hold the right and wrong classifications
  correctcounter=0.0
  wrongcounter=0.0

  #resultclass is the classification given from evaluate
  resultclass=False
  #actualclass is the actual example's class
  actualclass=False

  #goes through all the single dictionaries in the examples
  for example in examples:

    #finds what the tree classifies the example
    #print example
    resultclass=evaluate(node,example)
    #finds what the actual class is in the example
    actualclass=example['Class']

    if resultclass==actualclass:
      correctcounter+=1.0
    else:
      wrongcounter+=1.0

  totalcounter=correctcounter+wrongcounter
  #print  'corrects:'+ str(correctcounter)
  #print  'total:'+ str(totalcounter)
  return correctcounter/totalcounter

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  #end case for when its at a terminal node, returns the class
  if len(node.children)==0:
    #print  'label:'+ str(node.label)
    return node.label
  
  #when at non-terminal node
  else:
    #initalize the current attribute and whether or not that attribute is true for
    #the example
  
    tempattribute=node.attribute

    exampleresult=example[tempattribute]
    if exampleresult not in node.children.keys():
      return node.label
    #print node.attribute
    #print example[node.attribute]
    #print node.children.keys()
    #print node.children[exampleresult]
    # when the attribute of the node is true for the example, recurse and use the true 'Y' node next

    return evaluate(node.children[exampleresult], example)
    
def entropy(examples):
  '''
  takes dictionary of examples and target attribute, returns entropy
  '''
  classes_distinct = []
  classes_all = []
  for e in range(len(examples)):
      classes_all.append(examples[e]['Class'])
      if examples[e]['Class'] not in classes_distinct:
          classes_distinct.append(examples[e]['Class'])
  n = len(examples) #total number of data points in example
  c = collections.Counter(classes_all)
  sum = 0
  for i in classes_distinct:
      if c[i] > 0:
          sum += (-1)*(c[i] / float(n)) * math.log((c[i]/float(n)), 2)
      else:
          sum += 0
  return sum


def infoGain(data, targetAttribute):
    listOfAttributes = data[0].keys()
    listOfAttributes.remove("Class")

    # initialize variables
    valueFreq = {}
    subEntropy = 0.0

    # find index of attribute
    i = listOfAttributes.index(targetAttribute)

    # calculate frequency of values (yes, no, ?) of selected attribute (topic)
    for entry in data:
        if (valueFreq.has_key(entry[targetAttribute])):  # checks if the value already exists
            valueFreq[entry[targetAttribute]] += 1.0  # if it does, then adds to existing frequency
        else:
            valueFreq[entry[targetAttribute]] = 1.0  # if not, adds new key and frequency

    # calculate sum of entropy for each weighted subset based off probability
    for v in valueFreq.keys():  # for each value in valueFreq
        valProb = valueFreq[v] / sum(valueFreq.values())  # find probability weight of certain value compared to all
        dataSubset = [entry for entry in data if entry[targetAttribute] == v]  # subset of each value
        subEntropy += valProb * entropy(dataSubset)  # add to subsetEntropy with weighted probability

    return (entropy(data) - subEntropy)  # returns info gain for a targeted attribute



def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  currAcc = test(node,examples)
  nodes=[node]

  while len(nodes)>0:
    n = nodes.pop()
    if n.get_children() != {}:
      tempChilds = n.children
      n.children = {}
      tempAcc = test(node, examples)
      n.children = tempChilds
      if tempAcc > currAcc:
        currAcc = tempAcc
        n.children = {}
      else:
        nodes.extend(x for x in n.children.values())
  return node
