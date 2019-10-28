from numpy import exp
import numpy as np

#inputs and outputs data

'''
100000 training
 -----------------------------------------------------
a31 = 0.5032072519603431  print desired output = 0
********************************training2********************************
 -----------------------------------------------------
a31 = 0.9997439244081319  print desired output = 1
********************************training3********************************
 -----------------------------------------------------
a31 = 0.9997658359993862  print desired output = 1
********************************training4********************************
 -----------------------------------------------------
a31 = 0.010304073519704792  print desired output = 0


1000 training
 -----------------------------------------------------
a31 = 0.5924623845688971  print desired output = 0
********************************training2********************************
 -----------------------------------------------------
a31 = 0.9691510209509325  print desired output = 1
********************************training3********************************
 -----------------------------------------------------
a31 = 0.9800212044810988  print desired output = 1
********************************training4********************************
 -----------------------------------------------------
a31 = 0.12214661143413504  print desired output = 0


[111]->[0]
[101]->[1]
[011]->[1]
[001]->[0]
'''

#activation funciton -> this for normalize the outputs between 0 and 1
def nonlin(x, deriv=False):
  if(deriv==True):
     return (x*(1-x))
  return 1/(1+exp(-x))

#inputs layer 1 [nodes]
a11=1
a12=1
a13=1

np.random.seed(1)

#weights between layer 1 and layer 2 [weights]

weighta11a21=np.random.random()
weighta12a21=np.random.random()
weighta13a21=np.random.random()


weighta11a22=np.random.random()
weighta12a22=np.random.random()
weighta13a22=np.random.random()


weighta11a23=np.random.random()
weighta12a23=np.random.random()
weighta13a23=np.random.random()


weighta11a24=np.random.random()
weighta12a24=np.random.random()
weighta13a24=np.random.random()

#####################################3


#hidden layer2 nodes [nodes]
a21=1
a22=1
a23=1
a24=1

#weights between layer 2 and layer 3 [weights]

weighta21a31=np.random.random()
weighta22a31=np.random.random()
weighta23a31=np.random.random()
weighta24a31=np.random.random()

##############################


#outputpred=(weighta11a21*a11+weighta12a21*a12+weighta13a21*a13)*weighta21a31+(weighta11a22*a11+weighta12a22*a12+weighta13a22*a13)*weighta22a31+(weighta11a23*a11+weighta12a23*a12+weighta13a23*a13)*weighta23a31+(weighta11a24*a11+weighta12a24*a12+weighta13a24*a13)*weighta24a31
#outputpred=(weighta11a21*weighta21a31*a11+weighta12a21*weighta21a31*a12+weighta13a21*weighta21a31*a13)+(weighta11a22*weighta22a31*a11+weighta12a22*weighta22a31*a12+weighta13a22*weighta22a31*a13)+(weighta11a23*weighta23a31*a11+weighta12a23*weighta23a31*a12+weighta13a23*weighta23a31*a13)+(weighta11a24*weighta24a31*a11+weighta12a24*weighta24a31*a12+weighta13a24*weighta24a31*a13)

#print("fullequation outputpred = ",outputpred)

a31=0


for numberoftrainingc in range(0,1000):
    # ********************************training1********************************


    #let suppose the following inputs [1,1,1]
    a11=0
    a12=0
    a13=1


    #clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

    a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
    a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
    a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
    a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


    #normalize the layer2
    a21=nonlin(a21)
    a22=nonlin(a22)
    a23=nonlin(a23)
    a24=nonlin(a24)




    #calculate layer3 [nodes] based on layer2 and layer3
    a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

    #normalize the layer3
    a31=nonlin(a31)




    #backprobagation
    #error = output - realoutput
    err=0-a31 # 0 here is the desired output for inputs [111]->[0]


    #update weights between layer1 and layer2
    weighta11a21=weighta11a21+(a11*weighta21a31*a21*a31)*err
    weighta12a21=weighta12a21+(a12*weighta21a31*a21*a31)*err
    weighta13a21=weighta13a21+(a13*weighta21a31*a21*a31)*err



    weighta11a22=weighta11a22+(a11*weighta22a31*a22*a31)*err
    weighta12a22=weighta12a22+(a12*weighta22a31*a22*a31)*err
    weighta13a22=weighta13a22+(a13*weighta22a31*a22*a31)*err



    weighta11a23=weighta11a23+(a11*weighta23a31*a23*a31)*err
    weighta12a23=weighta12a23+(a12*weighta23a31*a23*a31)*err
    weighta13a23=weighta13a23+(a13*weighta23a31*a23*a31)*err



    weighta11a24=weighta11a24+(a11*weighta24a31*a24*a31)*err
    weighta12a24=weighta12a24+(a12*weighta24a31*a24*a31)*err
    weighta13a24=weighta13a24+(a13*weighta24a31*a24*a31)*err

    #update weights between layer2 and layer 3
    weighta21a31=weighta21a31+(a21*a31)*err
    weighta22a31=weighta22a31+(a22*a31)*err
    weighta23a31=weighta23a31+(a23*a31)*err
    weighta24a31=weighta24a31+(a24*a31)*err


    #end backprobagation

    # ********************************end training1********************************

    ##print("********************************training2********************************")
    # ********************************training2********************************

    #let suppose the following inputs [1,1,1]
    a11=0
    a12=1
    a13=1


    #clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

    a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
    a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
    a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
    a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


    #normalize the layer2
    a21=nonlin(a21)
    a22=nonlin(a22)
    a23=nonlin(a23)
    a24=nonlin(a24)




    #calculate layer3 [nodes] based on layer2 and layer3
    a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

    #normalize the layer3
    a31=nonlin(a31)


    #backprobagation
    #error = output - realoutput
    err=1-a31  # 0 here is the desired output for inputs [101]->[1]


    #update weights between layer1 and layer2
    weighta11a21=weighta11a21+(a11*weighta21a31*a21*a31)*err
    weighta12a21=weighta12a21+(a12*weighta21a31*a21*a31)*err
    weighta13a21=weighta13a21+(a13*weighta21a31*a21*a31)*err



    weighta11a22=weighta11a22+(a11*weighta22a31*a22*a31)*err
    weighta12a22=weighta12a22+(a12*weighta22a31*a22*a31)*err
    weighta13a22=weighta13a22+(a13*weighta22a31*a22*a31)*err



    weighta11a23=weighta11a23+(a11*weighta23a31*a23*a31)*err
    weighta12a23=weighta12a23+(a12*weighta23a31*a23*a31)*err
    weighta13a23=weighta13a23+(a13*weighta23a31*a23*a31)*err



    weighta11a24=weighta11a24+(a11*weighta24a31*a24*a31)*err
    weighta12a24=weighta12a24+(a12*weighta24a31*a24*a31)*err
    weighta13a24=weighta13a24+(a13*weighta24a31*a24*a31)*err

    #update weights between layer2 and layer 3
    weighta21a31=weighta21a31+(a21*a31)*err
    weighta22a31=weighta22a31+(a22*a31)*err
    weighta23a31=weighta23a31+(a23*a31)*err
    weighta24a31=weighta24a31+(a24*a31)*err




    #end backprobagation



    # ********************************end training2********************************


    ##print("********************************training3********************************")
    # ********************************training3********************************

    #let suppose the following inputs [1,1,1]
    a11=1
    a12=0
    a13=1


    #clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

    a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
    a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
    a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
    a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


    #normalize the layer2
    a21=nonlin(a21)
    a22=nonlin(a22)
    a23=nonlin(a23)
    a24=nonlin(a24)




    #calculate layer3 [nodes] based on layer2 and layer3
    a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

    #normalize the layer3
    a31=nonlin(a31)




    #backprobagation
    #error = output - realoutput
    err=1-a31  # 0 here is the desired output for inputs [111]->[0]


    #update weights between layer1 and layer2
    weighta11a21=weighta11a21+(a11*weighta21a31*a21*a31)*err
    weighta12a21=weighta12a21+(a12*weighta21a31*a21*a31)*err
    weighta13a21=weighta13a21+(a13*weighta21a31*a21*a31)*err



    weighta11a22=weighta11a22+(a11*weighta22a31*a22*a31)*err
    weighta12a22=weighta12a22+(a12*weighta22a31*a22*a31)*err
    weighta13a22=weighta13a22+(a13*weighta22a31*a22*a31)*err



    weighta11a23=weighta11a23+(a11*weighta23a31*a23*a31)*err
    weighta12a23=weighta12a23+(a12*weighta23a31*a23*a31)*err
    weighta13a23=weighta13a23+(a13*weighta23a31*a23*a31)*err



    weighta11a24=weighta11a24+(a11*weighta24a31*a24*a31)*err
    weighta12a24=weighta12a24+(a12*weighta24a31*a24*a31)*err
    weighta13a24=weighta13a24+(a13*weighta24a31*a24*a31)*err

    #update weights between layer2 and layer 3
    weighta21a31=weighta21a31+(a21*a31)*err
    weighta22a31=weighta22a31+(a22*a31)*err
    weighta23a31=weighta23a31+(a23*a31)*err
    weighta24a31=weighta24a31+(a24*a31)*err



    #end backprobagation




    # ********************************end training3********************************


    ##print("********************************training4********************************")
    # ********************************training4********************************

    #let suppose the following inputs [1,1,1]
    a11=1
    a12=1
    a13=1


    #clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

    a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
    a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
    a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
    a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


    #normalize the layer2
    a21=nonlin(a21)
    a22=nonlin(a22)
    a23=nonlin(a23)
    a24=nonlin(a24)




    #calculate layer3 [nodes] based on layer2 and layer3
    a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

    #normalize the layer3
    a31=nonlin(a31)


    #backprobagation
    #error = output - realoutput
    err=0-a31  # 0 here is the desired output for inputs [111]->[0]

    #update weights between layer1 and layer2
    weighta11a21=weighta11a21+(a11*weighta21a31*a21*a31)*err
    weighta12a21=weighta12a21+(a12*weighta21a31*a21*a31)*err
    weighta13a21=weighta13a21+(a13*weighta21a31*a21*a31)*err



    weighta11a22=weighta11a22+(a11*weighta22a31*a22*a31)*err
    weighta12a22=weighta12a22+(a12*weighta22a31*a22*a31)*err
    weighta13a22=weighta13a22+(a13*weighta22a31*a22*a31)*err



    weighta11a23=weighta11a23+(a11*weighta23a31*a23*a31)*err
    weighta12a23=weighta12a23+(a12*weighta23a31*a23*a31)*err
    weighta13a23=weighta13a23+(a13*weighta23a31*a23*a31)*err



    weighta11a24=weighta11a24+(a11*weighta24a31*a24*a31)*err
    weighta12a24=weighta12a24+(a12*weighta24a31*a24*a31)*err
    weighta13a24=weighta13a24+(a13*weighta24a31*a24*a31)*err

    #update weights between layer2 and layer 3
    weighta21a31=weighta21a31+(a21*a31)*err
    weighta22a31=weighta22a31+(a22*a31)*err
    weighta23a31=weighta23a31+(a23*a31)*err
    weighta24a31=weighta24a31+(a24*a31)*err





    #end backprobagation



    # ********************************end training4********************************



#end
'''

[111]->[0]
[101]->[1]
[011]->[1]
[001]->[0]

'''


#let suppose the following inputs [1,1,1]
a11=1
a12=1
a13=1


#clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


#normalize the layer2
a21=nonlin(a21)
a22=nonlin(a22)
a23=nonlin(a23)
a24=nonlin(a24)




#calculate layer3 [nodes] based on layer2 and layer3
a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

#normalize the layer3
a31=nonlin(a31)

if(a31>1):
    a31=1

print(" ----------------------------------------------------- ")
print("a31 =",a31 , " print desired output =",0)




'''

[111]->[0]
[101]->[1]
[011]->[1]
[001]->[0]

'''

# ********************************end training1********************************

print("********************************training2********************************")
# ********************************training2********************************

#let suppose the following inputs [1,1,1]
a11=1
a12=0
a13=1


#clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


#normalize the layer2
a21=nonlin(a21)
a22=nonlin(a22)
a23=nonlin(a23)
a24=nonlin(a24)




#calculate layer3 [nodes] based on layer2 and layer3
a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

#normalize the layer3
a31=nonlin(a31)
if(a31>1):
    a31=1



print(" ----------------------------------------------------- ")
print("a31 =",a31 , " print desired output =",1)

'''

[111]->[0]
[101]->[1]
[011]->[1]
[001]->[0]

'''



# ********************************end training2********************************


print("********************************training3********************************")
# ********************************training3********************************

#let suppose the following inputs [1,1,1]
a11=0
a12=1
a13=1


#clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


#normalize the layer2
a21=nonlin(a21)
a22=nonlin(a22)
a23=nonlin(a23)
a24=nonlin(a24)




#calculate layer3 [nodes] based on layer2 and layer3
a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

#normalize the layer3
a31=nonlin(a31)
if(a31>1):
    a31=1


print(" ----------------------------------------------------- ")
print("a31 =",a31 , " print desired output =",1)




'''

[111]->[0]
[101]->[1]
[011]->[1]
[001]->[0]

'''
# ********************************end training3********************************


print("********************************training4********************************")
# ********************************training4********************************

#let suppose the following inputs [1,1,1]
a11=0
a12=0
a13=1


#clculate layer2 [nodes] based on inputs and weights between layer1 and layer2

a21=weighta11a21*a11+weighta12a21*a12+weighta13a21*a13
a22=weighta11a22*a11+weighta12a22*a12+weighta13a22*a13
a23=weighta11a23*a11+weighta12a23*a12+weighta13a23*a13
a24=weighta11a24*a11+weighta12a24*a12+weighta13a24*a13


#normalize the layer2
a21=nonlin(a21)
a22=nonlin(a22)
a23=nonlin(a23)
a24=nonlin(a24)




#calculate layer3 [nodes] based on layer2 and layer3
a31=weighta21a31*a21+weighta22a31*a22+weighta23a31*a23+weighta24a31*a24

#normalize the layer3
a31=nonlin(a31)
if(a31>1):
    a31=1



print(" ----------------------------------------------------- ")
print("a31 =",a31 , " print desired output =",0)
