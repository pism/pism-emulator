import numpy as np
import pandas as pd
import seaborn as sns
import random
import pylab as plt
from pismemulator.utils import param_keys_dict as keys_dict
from pismemulator.utils import kl_divergence
from joblib import parallel, delayed

# Function that plots the posterior distributions of randomly selected ensembles with a fixed number of members
def plot_random_groups(df,vars,models,num_groups=10,per_group=50,seed=8675309):
    # df        : data frame of all models for true avg
    # vars      : list of variable (column) names
    # models    : list of available models (0,1,2 ...)
    # num_groups: number of ensembles
    # per_group : number of emulators per ensemble

    sns.set_theme(palette='colorblind')
    rows = int(np.ceil(len(vars)/2))
    cols = 2

    fig, axes = plt.subplots(rows,cols,figsize=(11.69,17.44))

    groups = []

    random.seed(seed)
    for i in range(num_groups):
        groups.append(random.sample(sorted(models),per_group))
    
    foo = 0
    bar = 0
    for i in range(rows):
        for j in range (cols):            
            var = vars[foo]
            # TODO define histogram of avg
            sns.kdeplot(data=df, x=var,ax=axes[i,j],color='black')
            foo += 1
            print(var)
            for group in groups:
                temp = df[df['Model'].isin(group)]
                sns.kdeplot(data=temp,x=var,ax=axes[i,j],color='green',alpha = .5) # remove y axis values, add x axis label
                axes[i,j].set(xlabel=keys_dict[var],yticklabels=[],ylabel=None)

# Function that plots the aggregate posterior distributions from any number of ensembles of emulators
def plot_posteriors(dfs,vars,labels=None):
    # dfs   : any number of data frames containing posterior samples
    # vars  : list of the names of variables contained in the dfs
    # labels: list of names for each df for graphing 
    
    sns.set_theme(palette='colorblind')
    rows = int(np.ceil(len(vars)/2))
    cols = 2
    named = True
    if labels is None:
        named = False  # Check if user passed in df labels
        print("debug")
        
    fig, axes = plt.subplots(rows,cols,figsize=(11.69,17.44))
    
    bar = 0
    for df in dfs:
        foo = 0
        for i in range(rows):
            for j in range (cols):            
                var = vars[foo]
                if named:
                    sns.kdeplot(data=df, x=var,ax=axes[int(i),int(j)],label=labels[bar])
                else:
                    sns.kdeplot(data=df, x=var,ax=axes[int(i),int(j)],label=bar)
                foo += 1
                axes[int(i),int(j)].legend()
        bar += 1
        print(bar)
    plt.legend()
    
def kl_divergences(df,vars,models,num_groups=10,per_group=50):
    # this was designed to calculate the kl divergences of random sub-ensembles from a larger "true" ensemble
    # df : data frame of all models for true avg
    # vars: list of variable (column) names
    # models: list of available models (0,1,2 ...)
    # num_groups: number of ensembles
    # per_group: number of emulators per ensemble

    divs = {}
    groups = []

    random.seed(8675309)
    for i in range(num_groups):
        groups.append(random.sample(sorted(models),per_group))
    
    bar = 0
    for i in range(len(vars)):
        kl_average = 0
        var = vars[i]
        p = np.histogram(df[var], bins=30,density=True)[0]
        for group in groups:
            temp = df[df['Model'].isin(group)]
            q = np.histogram(temp[var],bins=30,density=True)[0]
            kl_average += np.abs(kl_divergence(p,q))
        kl_average  = kl_average / num_groups
        divs[var] = kl_average
    return divs

def kl_deviation(df,vars,models):
    # This was designed to calculate the "kl deviation" of a single ensemble of emulators
    # In essence, we calculate the average kl divergence of each individual emulator with the ensemble average
    divs = {}

    for i in range(len(vars)):
        kl_average = 0
        var = vars[i]
        p = np.histogram(df[var], bins=30,density=True)[0]
        for model in models:
            temp = df[df['Model']==model]
            q = np.histogram(temp[var],bins=30,density=True)[0]
            kl_average += np.abs(kl_divergence(p,q))
        kl_average  = kl_average / int(len(models))
        divs[var] = kl_average
    return divs