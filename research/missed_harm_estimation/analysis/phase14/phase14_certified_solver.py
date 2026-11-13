#!/usr/bin/env python3
"""Phase 14: certified optimizer repair of ONLY the Phase 11 restricted MultiPPI GLS allocation.

This is not the MultiPPI author's SOCP/SDP or confidence interval; no real
safety-outcome labels are used. A solver's success flag is not a certificate.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import phase11_reference_unmodified as p
import phase13_methods as m


def repaired_optimize(Sigma, costs, *, gold_min=0.12, target_relative_gap=0.005,
                      max_fw=200, use_original=True, original_optimizer=None):
    """Solve the original restricted continuous allocation; certify convex gap.

    Feasible: x>=0, sum x=1, sum x[0:4]>=gold_min.
    Criterion: gap / objective <= target_relative_gap.
    On failure, return certified=False with explicit certificate; caller MUST
    reject or mark the run as uncertified, not silently treat it as optimized.
    """
    Sigma=np.asarray(Sigma,float);costs=np.asarray(costs,float)
    if Sigma.shape!=(3,3) or costs.shape!=(7,) or not np.all(np.isfinite(Sigma)) or not np.all(np.isfinite(costs)) or np.any(costs<=0) or not (0<gold_min<1) or not (0<target_relative_gap<1):
        raise ValueError('Invalid covariance, costs, or constraint')
    if np.linalg.eigvalsh(Sigma).min()<=0:
        raise ValueError('Pilot covariance must be positive definite')
    A=np.array(p.multi_matrices(Sigma))/costs[:,None,None]
    a=np.array([1.,0,0]);R=1e-8*np.eye(3)

    def both(x):
        info=np.einsum('j,jab->ab',x,A)+R
        v=np.linalg.solve(info,a)
        return float(a@v),-np.einsum('a,jab,b->j',v,A,v)

    def cert(x):
        return m.multi_optimality_certificate(Sigma,costs,x,gold_min=gold_min)

    def good(x):
        return bool(np.all(np.isfinite(x)) and np.all(x>=-1e-9) and
                    abs(x.sum()-1)<1e-7 and x[:4].sum()>=gold_min-1e-8)

    original=None;original_cert=None; original_success=False
    if use_original:
        x0, original_success=(original_optimizer or p.optimize_multi)(Sigma,costs)
        if good(x0):
            original=x0.copy();original_cert=cert(x0)
    starts=[np.array([1.,0,0,0,0,0,0]),
            np.array([gold_min,0,0,0,1-gold_min,0,0]),
            np.array([0.4,0.1,0.05,0.2,0.1,0.05,0.1])]
    if original is not None:starts.insert(0,original)
    constraints=[
        {'type':'eq','fun':lambda x:sum(x)-1,'jac':lambda x:np.ones(7)},
        {'type':'ineq','fun':lambda x:sum(x[:4])-gold_min,
         'jac':lambda x:np.array([1.,1.,1.,1.,0.,0.,0.])}
    ]
    candidates=[original] if original is not None else []
    # Optimizing the analytic gradient repairs spurious SLSQP success flags.
    for start in starts:
        opt=minimize(lambda x:both(x)[0],start,
                     jac=lambda x:both(x)[1],method='SLSQP',bounds=[(0,1)]*7,
                     constraints=constraints,options={'maxiter':550,'ftol':1e-13})
        if good(opt.x):candidates.append(opt.x)
    if not candidates:raise RuntimeError('No feasible allocation found')
    x=min(candidates,key=lambda x:both(x)[0]).copy()
    iterations=0
    for step in range(max_fw):
        report=cert(x)
        if report['relative_gap_upper']<=target_relative_gap:break
        f,g=both(x)
        best_gold=int(np.argmin(g[:4]));best_proxy=int(np.argmin(g[4:])+4)
        vertex=np.zeros(7)
        if g[best_gold]<=g[best_proxy]:vertex[best_gold]=1.
        else:vertex[best_gold]=gold_min;vertex[best_proxy]=1.-gold_min
        direction=vertex-x
        trial=minimize_scalar(lambda alpha:both(x+alpha*direction)[0],
                     method='bounded',bounds=(0,1),options={'xatol':1e-12,'maxiter':120})
        if trial.fun < f-1e-14:
            x=x+float(trial.x)*direction
            iterations+=1
        if step%12==0 or trial.fun >= f-1e-14:
            opt=minimize(lambda z:both(z)[0],x,
                    jac=lambda z:both(z)[1],method='SLSQP',bounds=[(0,1)]*7,
                    constraints=constraints,options={'maxiter':400,'ftol':1e-13})
            if good(opt.x) and both(opt.x)[0]<both(x)[0]-1e-14:x=opt.x
            if trial.fun >= f-1e-14 and cert(x)['relative_gap_upper']>target_relative_gap:
                # Fail visibly rather than inventing convergence.
                break
    after=cert(x)
    if not good(x):raise AssertionError('Optimizer returned infeasible allocation')
    if original_cert is not None and after['objective'] > original_cert['objective']*(1+1e-9):
        raise AssertionError('Repaired allocation objective increased')
    return {
        'fractions':x,
        'certified':bool(after['relative_gap_upper']<=target_relative_gap),
        'after_certificate':after,
        'original_certificate':original_cert,
        'original_solver_success':bool(original_success),
        'frank_wolfe_steps':iterations,
        'target_relative_gap':float(target_relative_gap),
        'scope':'RESTRICTED_CONTINUOUS_PILOT_GLS_OBJECTIVE_NOT_AUTHOR_MULTI_PPI'
    }
