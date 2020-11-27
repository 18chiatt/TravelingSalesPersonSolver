#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def greedy(self, time_allowance=60.0):
        start_time = time.time()
        cities = self._scenario.getCities()
        costs = getCostMatrix(cities)
        bestCost = np.inf
        bestTour = []
        print(cities)

        for i in range(len(cities)):
            if i % 10 == 0:  # only check for out of time bounds every 10 cities, should save some time doing
                # floating point operations and whatnot
                currTime = time.time()
                if currTime - start_time > time_allowance:
                    break
            cost, tour = greedyTour(i, costs)
            if cost < bestCost:
                bestTour = tour
                bestCost = cost

        citySolution = []
        for i in range(len(bestTour)):
            citySolution.append(cities[bestTour[i]])



        end_time = time.time()
        results = {}
        results['cost'] = bestCost
        results['time'] = end_time - start_time
        results['count'] = 1
        results['soln'] = TSPSolution(citySolution)
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    def greedySimple(self):
        cities = self._scenario.getCities()
        costs = getCostMatrix(cities)
        bestCost = np.inf
        bestTour = []

        for i in range(len(cities)):
            cost, tour = greedyTour(i, costs)
            if cost < bestCost:
                bestTour = tour
                bestCost = cost

        return bestCost, bestTour


    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    def branchAndBound(self, time_allowance=60.0):
        max = 0
        total =0
        pruned = 0
        solutions = 0
        startTime = time.time()
        cities = self._scenario.getCities()
        numCities = len(cities)
        #create BSSF via Greedy
        bestCost, bestTour = self.greedySimple()
        unreducedInitial = getCostMatrix(cities)

        # get initial cost matrix and reduce it into minimum cost and number of nodes left
        firstCost, lowerBound = reduce(unreducedInitial,0)
        initialPath = [0]
        firstSolution = BranchSolution(firstCost,lowerBound,initialPath)
        initialPriority = getPriority(firstSolution)
        heap = []
        heapq.heappush(heap, (initialPriority, firstSolution))

        while startTime - time.time() < time_allowance and len(heap) != 0:
            #do an iteration
            ignore, toExpand = heapq.heappop(heap)
            if toExpand.lowerBound > bestCost:
                pruned +=1
                continue


            newSolutions = expand(toExpand)
            total += len(newSolutions)
            for solution in newSolutions:
                if len(solution.tour) == numCities and solution.lowerBound < bestCost:
                    bestCost, bestTour = solution.lowerBound, solution.tour
                    solutions +=1
                    continue
                if len(solution.tour) == numCities and solution.lowerBound > bestCost:
                    continue
                if solution.lowerBound > bestCost:
                    pruned +=1
                else:
                    heapq.heappush(heap, (getPriority(solution) + random.random(),solution))
                    if(len(heap) > max):
                        max = len(heap)



        # solution has a partial path, minimum bound,


        ## ---return results

        citySolution = []
        for i in range(len(bestTour)):
            citySolution.append(cities[bestTour[i]])

        end_time = time.time()
        results = {}
        results['cost'] = bestCost
        results['time'] = end_time - startTime
        results['count'] = 1
        results['soln'] = TSPSolution(citySolution)
        results['max'] = max
        results['total'] = total
        results['pruned'] = pruned

        return results

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        pass


def getCostMatrix(cities):
    numCities = len(cities)
    costs = np.zeros((numCities, numCities))
    for i, origin in enumerate(cities):
        for j, dest in enumerate(cities):
            costs[i, j] = origin.costTo(dest)
    return costs


def greedyTour(index, costs):
    costs = np.copy(costs)
    numCities = len(costs)
    tour = [None for _ in range(numCities)]
    tour[0] = index
    currNode = index
    cost = 0
    indexInTour = 1

    while (indexInTour < numCities):
        minCost = np.inf
        nextNode = None
        for j in range(numCities):
            if (costs[currNode, j] < minCost):
                minCost = costs[currNode, j]
                nextNode = j
        if minCost == np.inf:
            return np.inf, None
        tour[indexInTour] = nextNode
        cost += minCost
        visit(costs,currNode,nextNode)
        currNode = nextNode
        indexInTour+=1

    cost += costs[currNode, index]




    return cost, tour

def visit(costs,currNode,nextNode):
    for j in range(len(costs)):
        costs[currNode,j] = np.inf
        costs[j,nextNode] = np.inf
    costs[nextNode,currNode] = np.inf
    return

def createAndVisit(olderCosts,currNode,nextNode):
    costs = np.copy(olderCosts)
    for j in range(len(costs)):
        costs[currNode,j] = np.inf
        costs[j,nextNode] = np.inf
    costs[nextNode,currNode] = np.inf
    return costs

def expand(solution):
    currentTour = solution.tour
    currentCity = currentTour[-1]
    costMatrix = solution.costMatrix
    lowerBound = solution.lowerBound

    numCities = len(costMatrix)
    newSolutions = [None for _ in range(numCities)]
    for i in range(numCities):
        base = lowerBound + costMatrix[currentCity,i]
        unFilteredCost = createAndVisit(costMatrix,currentCity,i)
        newCost, newLowerBound = reduce(unFilteredCost,base)
        newTour = currentTour + [i]
        currNewSolution = BranchSolution(newCost,newLowerBound,newTour)
        newSolutions[i] = currNewSolution

    return newSolutions





def reduce(costs, initialLowerBound): # return new costs, new lower bound
    lowerBound = initialLowerBound
    for i in range(len(costs)):
        lowest = np.inf
        for j in range(len(costs)):
            if costs[i,j] < lowest:
                lowest = costs[i,j]
        if lowest > 0 and lowest != np.inf:
            costs[i] = costs[i] - lowest
            lowerBound += lowest
    for j in range( len(costs)):
        lowest = np.inf
        for i in range(len(costs)):
            if costs[i,j] < lowest:
                lowest = costs[i,j]
        if lowest > 0 and lowest != np.inf:
            costs[:,j] = costs[:,j] - lowest
            lowerBound +=lowest
    return costs,lowerBound

def getPriority(solution):
    numVisited = len(solution.tour)
    discount = (.8 ** numVisited)
    return solution.lowerBound * discount

class BranchSolution:
    def __init__(self,costMatrix,lowerBound,tour):
        self.costMatrix = costMatrix
        self.lowerBound = lowerBound
        self.tour = tour

