# -*- coding: utf-8 -*-

import time
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from contextlib import closing
import numpy as np
from scipy.optimize import minimize


class Resource(object):
    """Resources container for the worker load"""
    def __init__(self, cost, model, x_data, y_data=None):
        self.cost_function = cost
        self.model = model
        self.x_data = x_data
        self.y_data = y_data


def worker_initialize(cost, model, x_data, y_data):
    global resource
    resource = Resource(cost, model, x_data, y_data)


def worker_compute(params):
    if not resource.model.set_parameters(params):
        return np.NaN
    res = resource.cost_function.cost(np.real(resource.model(resource.x_data)), resource.y_data)
    return res


def worker_gradient(params):
    resource.model.set_parameters(params)
    xout = resource.model.feed_through(resource.x_data, True)
    resource.model.backprop(resource.cost_function.gradient(xout, resource.y_data))
    return resource.model.get_gradients()


class CMA(object):
    """Implements the GA using CMA library"""
    def __init__(self, parallel=False):
        super(CMA, self).__init__()
        if parallel:
            self.num_cores = mp.cpu_count()
        else:
            self.num_cores = 1
        print('CMA on %d cpu(s) enabled' % self.num_cores)

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, popsize=None, maxiter=None, use_grad=False):
        """The training algorithm"""

        initsol = np.real(model.get_parameters())
        args = {'bounds': model.get_bounds(),
                'tolfun': tolfun,
                'verb_log': 0}
        sigma = np.max(model.get_bounds()[1])*0.1

        if popsize is not None:
            args['popsize'] = popsize

        if maxiter is not None:
            args['maxiter'] = maxiter

        grad = None
        if use_grad:
            grad = worker_gradient
          
        es = CMAEvolutionStrategy(initsol, sigma, args)
        if self.num_cores > 1:
            with closing(mp.Pool(self.num_cores, initializer=worker_initialize,
                                 initargs=(cost, model, x_data, y_data))) as pool:
                while not es.stop():
                    f_values, solutions = [], []
                    while len(solutions) < es.popsize:
                        x = es.ask(es.popsize-len(solutions), gradf=grad)
                        curr_fit = pool.map_async(worker_compute, x).get()
                        for value, solution in zip(curr_fit,x):
                            if not np.isnan(value):
                                solutions.append(solution)
                                f_values.append(value)
                    es.tell(solutions, f_values)
                    es.disp()
                pool.terminate()
        else:
            worker_initialize(cost, model, x_data, y_data)
            while not es.stop():
                f_values, solutions = [], []
                while len(solutions) < es.popsize:
                    curr_fit = x = np.NaN
                    while np.isnan(curr_fit):
                        x = es.ask(1, gradf=grad)[0]
                        curr_fit = worker_compute(x)
                    solutions.append(x)
                    f_values.append(curr_fit)
                es.tell(solutions, f_values)
                es.disp()
        print(es.result)

        model.set_parameters(es.result[0])
        return es.result[0]

    @property
    def num_cores(self):
        return self._num_cores

    @num_cores.setter
    def num_cores(self, cores):
        if cores > mp.cpu_count():
            print('CMA: the number of requested CPU is larger than cpu_count.')
        elif cores <= 0:
            raise AssertionError('CMA: the requested number of CPU is <= 0')
        self._num_cores = cores


class SGD(object):
    """ Stochastic gradient descent 
        
        maxiter: Iterations
        batch_size: Batch size
        lr: learning rate
        momentum: Momentum
        Nesterov: Nesterov momentum
        Noise: Gaussian noise 
        decay: Learning rate decay rate
      
    """
    
    
    def train(self, cost, model, x_data, y_data=None, maxiter=100, batch_size=0, lr=0.001, decay=0, momentum=0,nesterov=False, noise=0, log_step=100):
        oldG = np.zeros(model.get_parameters().shape)
        
        # Generate batches
        RE = 0
        if(batch_size > 0):
            BS = x_data.shape[1] / batch_size
            if(x_data.shape[1] % batch_size > 0):
                RE = 1
        else:
            BS = 1
            batch_size = x_data.shape[1]
            
        # Switch on/off noise
        nF = 0
        if noise > 0 :
            nF = 1
            
        t0 = time.time()
        
        # Loop over epoches
        for i in range(0, maxiter):
            
            # Loop over batches
            for b in range(0, BS+RE):
                #print("bs: ",b*batch_size, "be:",(b+1)*batch_size)
                
                data_x = x_data[:,b*batch_size:(b+1)*batch_size]
                data_y = y_data[:,b*batch_size:(b+1)*batch_size]
                
                Xout = model.feed_through(data_x, True)
                C = cost.cost(Xout,data_y)
                model.backprop(cost.gradient(Xout,data_y))
            
                W = model.get_parameters()
            
                # Nesterov update 
                if(nesterov==True):
                    model.set_parameters(W-oldG)
            
                # Get gradients
                G = model.get_gradients()
        
                # Adjust weights (with momentum)
                U = lr*G + momentum*oldG + nF*np.random.normal(0, lr/(1+i)**noise, oldG.shape)
                oldG = U
            
                W = W - U
            
                # Set gradients
                model.set_parameters(W)
            
            # Decay learning rate
            lr = lr*(1-decay)
            
            if i % log_step == 0:
                #print("Batch C: ",C)
                #C = cost.cost(model.feed_through(x_data, True),y_data)
                print("Iteration %d in %.2f(s), cost = %f" % (i,time.time()-t0, C))
            
        print("Cost: ",C)    
        print("Sol: ",W)
        print("Time: %d s" % (time.time()-t0))
        return W


class BFGS(object):
    """Implements the BFGS method"""

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, maxiter=100):
        """The training algorithm"""
        x0 = np.real(model.get_parameters())
        worker_initialize(cost, model, x_data, y_data)
        bounds = [ (model.get_bounds()[0][i],model.get_bounds()[1][i]) for i in range(model.size())]
        res = minimize(worker_compute, x0, jac=worker_gradient,
                       bounds=bounds, options = {'gtol': tolfun,
                                                 'disp': True, 'maxiter': maxiter})
        print(res)
        model.set_parameters(res.x)
        return res.x
