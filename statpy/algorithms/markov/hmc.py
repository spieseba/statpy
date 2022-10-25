import numpy as np

# HMC
def HMC(N_sample, n_lf, dt_lf,guidanceH,gradU):
    sample = np.array([])
    k = 0 # sample counter
    # perform proposals until target sample size N_sample achieved
    q = np.random.normal(loc=0.0,scale=1.0,size=1)
    while ( k < N_sample ):
        p = np.random.normal(loc=0.0,scale=1.0,size=q.shape) # auxiliary momenta
        E1 = guidanceH(q,p) # evaluate guidance Hamiltonian  
        q,p = leapfrog(q,p,n_lf,dt_lf,gradU) # leapfrog to evolve q,p
        E2 = guidanceH(q,p) # compute guidance Hamiltonian after evolution
        # accept new proposal with probability min(1,exp(-(E2-E1)))
        Ediff = E2 - E1
        if ( (Ediff <= 0) or (np.random.uniform() < np.exp(-Ediff)) ):
            sample = np.append(sample,q)
            k += 1
        else:
            continue
    return sample


# leapfrog integrator to evolve according to Hamilton's e.o.m.; number of steps = n_lf, step size = dt
def leapfrog(q,p,n_lf,dt,gradU):
    p -= 0.5*dt * gradU(q) # initial half step: p(dt/2) = p(0) - dt/2 * dS
    # perform (n-1) steps in q & p
    for i in range(n_lf - 1):
        q += dt * p
        p -= dt * gradU(q)
    q += dt * p     # n-th step for q
    p -= 0.5*dt * gradU(q) # final half step for p
    return q,p

        

