import random

from Chromosome import Chromosome


class Population:
    def __init__(self,x,y,dimPop,nrEpochs):
        self.x=x
        self.y=y
        self.dimPop=dimPop
        self.nrEpochs=nrEpochs
        self.chromosomes=[]

        self.createPopulation()

    def createPopulation(self):
        for i in range(self.dimPop):
            self.chromosomes.append(Chromosome(self.x,self.y))

    # parinte cu tournir
    # cel mai bun din parte de pop
    def pickParent(self):
        chr=random.sample(self.chromosomes,self.dimPop//10)

        bestChromosome = chr[0]

        for chromosome in chr:
            if chromosome.fitness<bestChromosome.fitness:
                bestChromosome=chromosome

        return bestChromosome


    # create new cromosome
    # plus mutations
    def getOffspring(self,m,f):
        chromosome=Chromosome(self.x,self.y)
        chromosome.repr=[]

        startM = random.randrange(len(m.repr))
        endM = m.traverse(startM)
        startF = random.randrange(len(f.repr))
        endF = f.traverse(startF)

        for i in range(0, startM):
            chromosome.repr.append(m.repr[i])
        for i in range(startF, endF):
            chromosome.repr.append(f.repr[i])
        for i in range(endM, len(m.repr)):
            chromosome.repr.append(m.repr[i])

        if random.uniform(0,1)<=0.5:
            chromosome.mutate()

        chromosome.calculateFitness()

        return chromosome

    #
    def getBestChromosome(self):
        bestChromosome = self.chromosomes[0]

        for chromosome in self.chromosomes:
            if chromosome.fitness<bestChromosome.fitness:
                bestChromosome=chromosome

        return bestChromosome

    def getWorstChromosome(self):
        bestChromosome = self.chromosomes[0]

        for chromosome in self.chromosomes:
            if chromosome.fitness>bestChromosome.fitness:
                bestChromosome=chromosome

        return bestChromosome

    # training
    # ia cel mai prost si inlocuieste cu new pop daca e mai bun
    def findSolution(self):
        count = 0
        bestChromosome = self.chromosomes[0]

        for i in range(self.nrEpochs):

            if count % 5 == 0:
                print("Epoch: ", count)


            localBestChromosome = self.getBestChromosome()

            if localBestChromosome.fitness < bestChromosome.fitness:
                bestChromosome = localBestChromosome

            worst=self.getWorstChromosome()

            print(bestChromosome.fitness)
            print(worst.fitness)
            print()

            male=self.pickParent()
            female=self.pickParent()
            offspring=self.getOffspring(male,female)

            index=self.chromosomes.index(worst)

            if worst.fitness>offspring.fitness:
                self.chromosomes[index]=offspring

            count = count + 1

        return bestChromosome

