from spgates import  spAND, spOR, spXOR, spXNOR, spNAND, spNOR, spNOT
import random
import matplotlib.pyplot as plot

from copy import deepcopy
from numpy import var, mean

class Element:
    def __init__(self, type, input, output):
        self.type = type  # index in ElementTypes
        self.input = input  # array of indexes in SignalsTable
        self.output = output  # index in SignalsTable

class SimulationModel:
    def __init__(self, signalsTable, elementsTable, topInputs):
        self.signalsTable = signalsTable
        self.elementsTable = elementsTable
        self.topInputs = topInputs  # contains signals indexes that are top inputs

    def runSimulation(self, inputValueList):
        self.setTopInputs(inputValueList)

        for element in self.elementsTable:
            self.process(element)

    def process(self, element):
        output = 0
        if element.type == 'NOT':
            output = spNOT(self.signalsTable[element.input[0]])
        else:
            inputValueList = self.getElementInputValueList(element)
            if element.type == 'AND':
                output = spAND(inputValueList)
            elif element.type == 'OR':
                output = spOR(inputValueList)
            elif element.type == 'XOR':
                output = spXOR(inputValueList)
            elif element.type == 'NAND':
                output = spNAND(inputValueList)
            elif element.type == 'NOR':
                output = spNOR(inputValueList)
            elif element.type == 'XNOR':
                output = spXNOR(inputValueList)
            else:
                print("unsupported")
        self.signalsTable[element.output] = output

    def getElementInputValueList(self, element):
        tempInputValueList = []
        for inputIndex in element.input:
            signal = self.signalsTable[inputIndex]
            tempInputValueList.append(signal)
        return tempInputValueList

    def setTopInputs(self, inputValueList):
        inputValueIndex = 0
        for signalIndex in self.topInputs:
            self.signalsTable[signalIndex] = inputValueList[inputValueIndex]
            inputValueIndex += 1

    def findTopInputs(self):
        signalMarkingTable = [0] * len(self.signalsTable)
        for element in self.elementsTable:
            outputIndex = element.output
            signalMarkingTable[outputIndex] = 1
        for signalIndex in range(len(signalMarkingTable)):
            if signalMarkingTable[signalIndex] == 0:
                self.topInputs.append(signalIndex)

    def sortElements(self):
        sortedElementsTable = []
        markedInputs = self.topInputs[:]

        while len(sortedElementsTable) != len(self.elementsTable):
            for element in self.elementsTable:
                if element in sortedElementsTable:
                    continue
                if self.isElementInLevel(element, markedInputs):
                    sortedElementsTable.append(element)
                    markedInputs.append(element.output)

        self.elementsTable = sortedElementsTable

    def isElementInLevel(self, element, markedInputs):
        inputList = element.input
        for input in inputList:
            if input not in markedInputs:
                return False
        return True

    def countTopInputs(self):
        return len(self.topInputs)

    def resetSignals(self):
        for index in range(len(self.signalsTable)):
            self.signalsTable[index] = 0

class Loader:
    def __init__(self, fileName):
        self.fileName = fileName
        self.signalDictionary = {}
        self.signalsTableSize = 0

    def loadFile(self):
        elemTable = []
        topInputsTable = []

        circuitFile = open(fileName)
        for line in circuitFile:
            stringList = line.split()

            if stringList[0] == "TLPINPUTS":
                topInputsTable = self.createTopInputs(stringList[1:])
                continue

            self.addSignals(stringList[1:])
            elemTable.append(self.createElement(stringList))
        circuitFile.close()

        signalsTable = [0] * self.signalsTableSize
        return SimulationModel(signalsTable, elemTable, topInputsTable)

    def createTopInputs(self, stringList):
        tempSignalsTable = []
        for signalName in stringList:
            signalIndex = self.signalsTableSize
            self.signalsTableSize += 1
            self.signalDictionary.update({signalName: signalIndex})
            tempSignalsTable.append(signalIndex)
        return tempSignalsTable

    def addSignals(self, stringList):
        for signalName in stringList:
            if signalName in self.signalDictionary:
                continue
            else:
                signalIndex = self.signalsTableSize
                self.signalsTableSize += 1
                self.signalDictionary.update({signalName: signalIndex})

    def createElement(self, stringList):
        elemType = stringList[0]
        elemOutput = self.signalDictionary.get(stringList[1])
        elemIndex = []
        for signalName in stringList[2:]:
            elemIndex.append(self.signalDictionary.get(signalName))

        return Element(elemType, elemIndex, elemOutput)

    def printFile(self):
        print(self.fileName)
        circuitFile = open(self.fileName)
        for line in circuitFile:
            print(line, end="")
        circuitFile.close()
        print("\n")

    def printSimulationModel(self, simulationModel):
        print("SimulationModel")
        for element in simulationModel.elementsTable:
            outputName = self.getSignalName(element.output)
            inputNameString = ""
            for inputIndex in element.input:
                inputNameString += self.getSignalName(inputIndex) + " "
            print(element.type + " " + outputName + " " + inputNameString)

    def getSignalName(self, signalIndex):
        for key, value in self.signalDictionary.items():
            if signalIndex == value:
                return key
        return "signalName doesn't exist"

def createSimulationModel(fileName):
    loader = Loader(fileName)
    simulationModel = loader.loadFile()

    if not simulationModel.topInputs:
        simulationModel.findTopInputs()

    simulationModel.sortElements()

    loader.printFile()
    loader.printSimulationModel(simulationModel)
    return simulationModel

class Individual:
    def __init__(self):
        self.workload = []
        self.score = -1

class Population:
    def __init__(self):
        self.individuals = []
        self.bestIndividual = []

def runGeneticAlgorithm(simulationModel, populationSize, workloadSize,
                        maxGenerations, mutateRate):
    global executionCount
    print("\nRUNNING Execution", executionCount,
          ":", maxGenerations, "Generations")

    generations = []
    topInputsCount = simulationModel.countTopInputs()
    population = seedPopulation(populationSize, workloadSize, topInputsCount)

    for i in range(maxGenerations):
        #print("------------------------------")
        #print('\nGeneration:\t', (i+1))
        measurePopulation(simulationModel, population)
        generations.append(population)

        parents = selectParents(population)
        newPopulation = crossoverParents(parents, populationSize, workloadSize)
        newPopulation = mutatePopulation(newPopulation, mutateRate)
        population = newPopulation

    print('FINISHED Execution', executionCount)
    return generations

def seedPopulation(populationSize, workloadSize, topInputsCount):
    population = Population()
    for i in range(populationSize):
        individual = Individual()
        individual.workload = generateRandomWorkload(topInputsCount, workloadSize)
        population.individuals.append(individual)
    return population

def generateRandomWorkload(inputsNumber, length):
    workload = []
    for row in range(length):
        row = []
        for column in range(inputsNumber):
            row.append(round(random.uniform(0, 1)))
        workload.append(row)
    return workload

def measurePopulation(simulationModel, population):
    for individual in population.individuals:
        switchesCounter = 0
        simulationModel.resetSignals()

        for input in individual.workload:
            signalsBefore = deepcopy(simulationModel.signalsTable)
            simulationModel.runSimulation(input)
            signalsAfter = deepcopy(simulationModel.signalsTable)
            switchesCounter += countSwitches(signalsBefore, signalsAfter)

        individual.score = switchesCounter

    '''
    print("\n-----Measure Population----")
    printPopulation(population)
    '''

def countSwitches(signalsBefore, signalsAfter):
    switches = 0
    for signalIndex in range(len(signalsBefore)):
        if signalIndex in simulationModel.topInputs:
            continue
        if signalsBefore[signalIndex] != signalsAfter[signalIndex]:
            switches += 1
    return switches

def selectParents(population):
    best = sbest = -1
    besti = sbesti = -1
    for individualIndex in range(len(population.individuals)):
        individualScore = population.individuals[individualIndex].score
        if individualScore > best:
            sbest = best
            sbesti = besti
            best = individualScore
            besti = individualIndex
        elif individualScore >= sbest:
            sbest = individualScore
            sbesti = individualIndex

    population.bestIndividual = population.individuals[besti]

    parent1 = population.individuals[besti]
    parent2 = population.individuals[sbesti]
    '''
    print("\n-------Select Parents------")
    printIndividual(parent1)
    printIndividual(parent2)
    '''
    return [parent1, parent2]

def crossoverParents(parents, populationSize, workloadSize):
    newPopulation = Population()
    newPopulation.individuals.append(parents[0])
    newPopulation.individuals.append(parents[1])
    # print("\n----Crossover Parents----")

    while len(newPopulation.individuals) < populationSize:
        crossoverLine = random.randint(0, workloadSize-1)
        parentChoice = random.randint(0, 1)
        parent1 = parents[parentChoice]
        parent2 = parents[abs(parentChoice - 1)]
        '''
        print("CrossoverLine: ", crossoverLine)
        printIndividual(parent1)
        printIndividual(parent2)
        '''

        individual = Individual()
        individual.workload = generateChildWorkload(parent1, parent2,
                                            crossoverLine, workloadSize)
        newPopulation.individuals.append(individual)

    # printPopulation(newPopulation)
    return newPopulation

def generateChildWorkload(parent1, parent2, crossoverLine, workloadSize):
    childWorkload = []
    for i in range(workloadSize):
        if i <= crossoverLine:
            childWorkload.append(parent1.workload[i])
        else:
            childWorkload.append(parent2.workload[i])
    return childWorkload

def mutatePopulation(population, mutateRate):
    mutatedPopulation = Population()
    mutatedPopulation.individuals.append(population.individuals[0])
    mutatedPopulation.individuals.append(population.individuals[1])

    for individualIndex in range(2, len(population.individuals)):
        individual = population.individuals[individualIndex]

        mutatedIndividual = Individual()
        for input in individual.workload:
            mutatedInput = []

            for value in input:
                probability = random.random()
                if probability <= mutateRate:
                    value = abs(value-1)
                mutatedInput.append(value)

            mutatedIndividual.workload.append(mutatedInput)

        mutatedPopulation.individuals.append(mutatedIndividual)
    '''
    print("\n----Mutate Population----")
    printPopulation(mutatedPopulation)
    '''
    return mutatedPopulation

def printPopulation(population):
    print("Printing Populations's Individuals")
    for individual in population.individuals:
        printIndividual(individual)

def printIndividual(individual):
    print("Individual's score: ", individual.score)
    for row in individual.workload:
        print(row)

def getExecutionScores(execution):
    executionScores =[]
    for population in execution:
        executionScores.append(population.bestIndividual.score)
    return executionScores

def printExecutionResults(execution, executionScores, executionCount):
    print('\n---------------  EXECUTION', executionCount, '---------------')
    print('Generation scores:\n', executionScores)
    print('\nBest individual:')
    for row in execution[-1].bestIndividual.workload:
        print(row)

def getBestIndividual(executionsList):
    bestIndividual = []
    bestScore = 0
    for execution in executionsList:
        if execution[-1].bestIndividual.score > bestScore:
            bestIndividual = execution[-1].bestIndividual
    return bestIndividual

def createBestWorkloadFile(bestIndividual):
    file = open("bestWorkload.txt", "w")
    file.write("Score: " + str(bestIndividual.score) + "\n")
    file.write("Workload:\n")
    for row in bestIndividual.workload:
        file.write(str(row))
        file.write("\n")
    file.close()

if __name__ == '__main__':
    L = 2                 # workload size
    N = 30              # population size
    m = 0.05          # mutation rate
    G = 100           # max number of generations
    fileName = "circuitFile.txt"
    # testbench()

    simulationModel = createSimulationModel(fileName)

    executionCount = 1
    execution1 = runGeneticAlgorithm(simulationModel, N, L, G, m)
    executionCount = 2
    execution2 = runGeneticAlgorithm(simulationModel, N, L, G, m)
    executionCount = 3
    execution3 = runGeneticAlgorithm(simulationModel, N, L, G, m)

    executionScores1 = getExecutionScores(execution1)
    executionScores2 = getExecutionScores(execution2)
    executionScores3 = getExecutionScores(execution3)

    printExecutionResults(execution1, executionScores1, 1)
    printExecutionResults(execution2, executionScores2, 2)
    printExecutionResults(execution3, executionScores3, 3)

    plot.plot(executionScores1, label='Execution 1')
    plot.plot(executionScores2, label='Execution 2')
    plot.plot(executionScores3, label='Execution 3')

    plot.xlabel('Generation #')
    plot.ylabel('Generation score')
    plot.legend(loc='best')
    plot.show()

    bestIndividual = getBestIndividual([execution1, execution2, execution3])
    createBestWorkloadFile(bestIndividual)
'''
    L = 2
    N = 3
    m = 0.05
    G = 3
'''