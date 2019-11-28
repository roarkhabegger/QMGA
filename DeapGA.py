#DeapGA.py File used for running variational method genetic alg

#Libraries used in code
import random
import numpy as np
import sys
import argparse
from argparse import RawTextHelpFormatter
import importlib

#Using DEAP python Library
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from time import time

def Main():
    t0 = time()
#Parse arguments for prob, ngen, npop, percKeep
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("prob",type=str,default='QD',
                        help="Which Problem File?\n")
    parser.add_argument("numGens",type=int,default=10,
                        help="How Many Generations to do?\n")
    parser.add_argument("numInds",type=int,default=20,
                        help="How big of a population?\n")
    parser.add_argument("percKeep",type=float,default=10,
                        help="What percentage of the population survives?\n")

    args = parser.parse_args()

#Store input values
    ngen = args.numGens
    npop = args.numInds
    if args.percKeep >=1:
        print("Can't keep >= 100% of the population!")
        return
    nKeep = int(npop*args.percKeep)

#Open problem file and import it
    #NOTE:: different users need to redefine this mypath definition
    mypath = r"/mnt/c/Users/Roark Habegger/OneDrive"
    mypath += r"/Documents/UNC/Senior Year/PHYS 521/GA Project"
    probFile = args.prob
    sys.path.insert(0,mypath)
    p = importlib.import_module(probFile)

#Get problem variables and functions
    npar = p.NumPar
    E = p.Func
    feasible = p.ParBounds
    dist = p.DistOutOfBounds

#Create framework for toolbox
    creator.create("FitnessMax",base.Fitness,weights = (1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

#create toolbox elements
    toolbox = base.Toolbox()
    toolbox.register("attr_float",random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n = npar)
    toolbox.register("population", tools.initRepeat, list,toolbox.individual)

    #Only include crossover if 2 parameters to crossover
    if npar>=2:
        toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta = 10, low = 0,
                      up = 1, indpb = 0.4)
    toolbox.register("select", tools.selBest,fit_attr='fitness')
    toolbox.register("evaluate", E)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible,0.0,dist))

#define statistics objec
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)

#What to print at end of generation and record in logbook
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness"

#Initial population creation
    pop = toolbox.population(n=npop)

#used to return best individual
    hof = tools.HallOfFame(1)

#RUN Mu+Lambda algorithm
    result,log = algorithms.eaMuPlusLambda(pop,toolbox,nKeep,npop-nKeep,
                 0.0,1.0 ,ngen, stats = stats, verbose = True, halloffame = hof)
    #Print best individual after ngens
    print(hof)
    print(time()-t0)
#Allow problem to plot/output important statistical info
    p.plotter(hof)

#Run the above function
Main()
