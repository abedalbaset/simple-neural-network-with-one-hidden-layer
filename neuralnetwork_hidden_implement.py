import numpy as np


#ini
buffsize=3
numpurofoutputs=1
numperofinputs=buffsize
numberofnhiddenlayer=2*buffsize
numberoftraining=1000
learningrate=1


inputs=[
[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]]


outputs=[
0,
1,
1,
0
]


numperoftestspertrial=len(outputs)


weightsa=[]
weightsb=[]


for inicnumtests in range(0,numberofnhiddenlayer):
    tmparr=[]
    for inic in range(0,numperofinputs):
        tmparr.append(np.random.random())
    weightsa.append(tmparr)

for inic in range(0,numberofnhiddenlayer):
    weightsb.append(np.random.random())


def nonlin(x, deriv=False):
  if(deriv==True):
     return (x*(1-x))
  return 1/(1+np.exp(-x))
#end ini


for numperoftrainingcounter in range(0,numberoftraining):
    for numoftestspertrainingc in range(0,numperoftestspertrial):
        #forward process
        nhidden=[]
        #clculate numbofnhidden
        for nc in range(0,numberofnhiddenlayer):
            cl=0
            for wc in range(0,numperofinputs):
                cl=cl+weightsa[nc][wc]*inputs[numoftestspertrainingc][wc]
            cl=nonlin(cl)
            nhidden.append(cl)

        outputpredict=0
        for nc in range(0,numberofnhiddenlayer):
            outputpredict=outputpredict+weightsb[nc]*nhidden[nc]

        #print("outputpredict =",outputpredict)
        #print("test outputs[numoftestspertrainingc] =",outputs[numoftestspertrainingc])
        outputpredict=nonlin(outputpredict)
        #print("outputpredict =",outputpredict)
        Err=(outputs[numoftestspertrainingc]-outputpredict)*learningrate


        #backprobagation

        #update weightsa
        for nc in range(0,numberofnhiddenlayer):
            cl=0
            for wc in range(0,numperofinputs):
                weightsa[nc][wc]=weightsa[nc][wc]+Err*(inputs[numoftestspertrainingc][wc]*weightsb[nc]*nhidden[nc]*outputpredict)


        for nc in range(0,numberofnhiddenlayer):
            weightsb[nc]=weightsb[nc]+Err*outputs[numoftestspertrainingc]*nhidden[nc]



        #update weightsb


        #end backprobagation



#end forward process




#verfied inputs
for numoftestspertrainingc in range(0,numperoftestspertrial):
    #forward process
    nhidden=[]
    #clculate numbofnhidden
    for nc in range(0,numberofnhiddenlayer):
        cl=0
        for wc in range(0,numperofinputs):
            cl=cl+weightsa[nc][wc]*inputs[numoftestspertrainingc][wc]
        cl=nonlin(cl)
        nhidden.append(cl)

    outputpredict=0
    for nc in range(0,numberofnhiddenlayer):
        outputpredict=outputpredict+weightsb[nc]*nhidden[nc]
    outputpredict=nonlin(outputpredict)
    print("outputpredict =",outputpredict," outputs[numoftestspertrainingc]=",outputs[numoftestspertrainingc])







#end
