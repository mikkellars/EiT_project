"""
Hyperparameter optimization.
"""

import os
import sys
sys.path.append(os.getcwd())

import random
from bayes_opt import BayesianOptimization
from vision.utils.trainer import train_model


def random_search(model, criterion, dls, opt, n_classes:int, name:str,
                  log_path:str, lrs=(1e-4, 1e-2), wds=(1e-4, 0.4), epochs:int=10,
                  n_iters:int=25):
    
    best_acc = 0.0
    for i in range(n_iters):
        lr = random.uniform(lrs[0], lrs[1])
        wd = random.uniform(wds[0], wds[1])

        def fit_with(lr:float, wd:float, log_path:str, epochs:int, n_classes:int, name:str, criterion, dls, model):
            optimizer = opt(model.parameters(), lr=lr, wd=wd)
            name += f'{lr}_{wd}'
            model, acc, loss = train_model(model=model, criterion=criterion, dls=dls,
                                           opt=opt, n_classes=n_classes, name=name,
                                           log_path=log_path, epochs=epochs, verbose=False)
            print(f'Current accuracy: {acc*100:.2f} Current loss: {loss:.4f}')
            return acc

        acc = fit_with(lr=lr, wd=wd, log_path=log_path, epochs=epochs, n_classes=n_classes, criterion=criterion,
                       dls=dls, model=model, opt=optimizer, name=name)

        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_wd = wd

    print(f'The highest accuracy is {best_acc} with learning rate {best_lr} and weight decay {best_wd}.')
    
    return best_acc, best_lr, best_wd
    
    
def bayesian_search(model, criterion, dls, opt, n_classes:int, name:str,
                    log_path:str, lrs=(1e-4, 1e-2), wds=(1e-4, 0.4), epochs:int=10,
                    n_iters:int=25):
    """Bayesian optimization for Hyperparameter search.

    Args:
        model ([type]): [description]
        criterion ([type]): [description]
        dls ([type]): [description]
        opt ([type]): [description]
        n_classes (int): [description]
        name (str): [description]
        log_path (str): [description]
        lrs (tuple, optional): [description]. Defaults to (1e-4, 1e-2).
        wds (tuple, optional): [description]. Defaults to (1e-4, 0.4).
        epochs (int, optional): [description]. Defaults to 10.
        n_iters (int, optional): [description]. Defaults to 25.
    """

    def fit_with(lr:float, wd:float, model, criterion, dls, opt, name, log_path, epochs):
        optim = opt(model.parameters(), lr=lr, wd=wd)
        name += f'{lr}_{wd}'
        model, acc, loss = train_model(model=model, criterion=criterion, dls=dls,
                                        opt=opt, n_classes=n_classes, name=name,
                                        log_path=log_path, epochs=epochs, verbose=False)
        print(f'Current accuracy: {acc*100:.2f} Current loss: {loss:.4f}')
        return acc
    
    pbounds = {'lr': , 'wd': }

    optimizer = BayesianOptimization(
        f=fit_with,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize()

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)
    
    return optimizer.max