import random
import numpy as np

# preordine arbore
#


class Chromosome:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.fitness = float("inf")
        self.normX=None
        self.normY=None

        self.functions = ['-', '+','*']
        self.terminals=range(len(self.x[0]))


        # Marimea arborelui
        self.depthMax=10
        self.repr=[]

        self.normalize()
        self.grow()
        self.calculateFitness()

    def normalize(self):
        self.normX=np.linalg.norm(self.x)
        self.x=self.x/self.normX

        self.normY=np.linalg.norm(self.y)
        self.y=self.y/self.normY

    # rez fin
    def getOutput(self,input):
        rez,_=self.eval(input)
        return rez*self.normY

    # init tree
    def grow(self,currentLevel=0):
        if currentLevel==self.depthMax:
            val=random.choice(self.terminals)
            self.repr.append(val)
        elif random.uniform(0,1)<=0.5:
            val=random.choice(self.functions)
            self.repr.append(val)
            self.grow(currentLevel+1)
            self.grow(currentLevel+1)
        else:
            val=random.choice(self.terminals)
            self.repr.append(val)

    # rez arbore
    def eval(self,input,poz=0):
        if self.repr[poz] in self.terminals:
            return input[self.repr[poz]],poz
        else:
            pozOp=poz
            left,poz=self.eval(input,poz+1)
            right,poz=self.eval(input,poz+1)

            if self.repr[pozOp]=='-':
                return left-right,poz
            elif self.repr[pozOp]=='+':
                return left+right,poz
            elif self.repr[pozOp]=='*':
                return left*right,poz

    def calculateFitness(self):
        sum = 0

        for i in range(len(self.x)):
            crtEval, _ = self.eval(self.x[i])
            # self.y[i][0]
            crtErr = abs(crtEval - self.y[i][0]) ** 2
            sum += crtErr

        self.fitness = sum

    # schimba o operatie random
    # sau
    # schimba un indice cu altul
    def mutate(self):
        poz=random.randint(0,len(self.repr)-1)

        if self.repr[poz] in self.functions:
            func=random.choice(self.functions)
            self.repr[poz]=func
        else:
            term=random.choice(self.terminals)
            self.repr[poz]=term

    # helper for new generation
    # ia o parte din mama si o parte din tata
    def traverse(self,pos):
        if self.repr[pos] in self.terminals:
            return pos + 1
        else:
            pos = self.traverse(pos + 1)
            pos = self.traverse(pos)
            return pos
