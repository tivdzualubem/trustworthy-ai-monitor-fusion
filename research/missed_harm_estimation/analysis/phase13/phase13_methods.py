#!/usr/bin/env python3
"""Mathematical checks of Phase11 mean inference. No private text or identities.

This is an audit, not an author-code replication. In particular, the AM-PPI
identity here does not implement its joint KKT policy optimizer.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm
import phase11_reference_unmodified as p


def feasible_ci(lo:float, hi:float, q:float):
    """Intersect a parameter CI with the *known* [0,q] for A=P(Y=1,D=0).

    A nominal CI cannot lose coverage when intersected with the parameter
    space. For a CI wholly disjoint from it, return empty interval (NaNs)
    instead of silently fabricating a nonempty 'valid' interval.
    """
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(q)) or not 0 <= q <= 1 or hi < lo:
        raise ValueError('Invalid interval or support')
    low=max(0.,float(lo)); high=min(float(q),float(hi))
    if high+1e-12 < low:return None
    return (low,max(low, high))


def asymptotic_wald(mu:float, se:float, q:float, alpha:float=.05):
    """Exploratory normal/Wald CI, *not* finite-sample coverage certificate."""
    if not (np.isfinite(mu) and np.isfinite(se)) or se < 0 or not 0<alpha<1:
        raise ValueError('Invalid Wald parameters')
    z=float(norm.ppf(1-alpha/2))
    return feasible_ci(mu-z*se,mu+z*se,q)


def multi_optimality_certificate(Sigma, costs, fraction, gold_min=.12, ridge=1e-8):
    """Convex first-order global-optimality GAP UPPER BOUND for Phase11 subset objective.

    f(u) = e0^T (sum_j u_j Q_j/c_j + ridge*I)^-1 e0.
    Since f is convex, f(u)-f(u*) <= grad f(u)^T u - min_{v feasible} grad f(u)^T v.
    This certifies *only* Phase11's restricted GLS allocation problem, which
    includes an extra 'gold budget >=12%' constraint absent from its paper.
    """
    u=np.asarray(fraction,float); c=np.asarray(costs,float)
    if u.shape!=(7,) or c.shape!=(7,) or np.any(c<=0) or np.any(u < -1e-9) or not np.isclose(u.sum(),1,atol=1e-6) or u[:4].sum()+1e-8<gold_min:
        raise ValueError('Infeasible fractions/costs')
    if not 0 <= gold_min <= 1:raise ValueError('Bad gold_min')
    qs=p.multi_matrices(np.asarray(Sigma,float)); eye=np.eye(3)
    info=sum(u[j]*qs[j]/c[j] for j in range(7)) + ridge*eye
    a=np.array([1.,0.,0.]); h=np.linalg.solve(info,a)
    objective=float(a@h)
    grad=np.array([-float(h@Q@h)/cj for Q,cj in zip(qs,c)])
    gold_g=float(np.min(grad[:4])); proxy_g=float(np.min(grad[4:]))
    min_linear=gold_g if gold_g<=proxy_g else gold_min*gold_g+(1-gold_min)*proxy_g
    absolute_gap=max(0.,float(grad@u-min_linear))
    return dict(objective=objective,absolute_gap_upper=absolute_gap,
        relative_gap_upper=absolute_gap/max(abs(objective),1e-12),
        feasible_gold_fraction=float(u[:4].sum()),gradient=grad.tolist(),
        gold_min=float(gold_min),certifying_objective='restricted_pilot_GLS_information_only')


def amp_biased_predictor_unbiased_bernoulli(z, prediction, inclusion_prob, inclusion_indicator, q=1.):
    """Exact fixed-policy AM-PPI Eq.(1) mean identity (Bernoulli HT correction).

    For fixed (z,pred,pi), Xi ~ independent Bernoulli(pi_i), this is
    unbiased for q*mean(z), irrespective of prediction error. NOT full AM-PPI.
    Returns an *unbiased estimator of design variance* conditional on z,pred.
    Zero/one propensities are handled explicitly, and only observed Y is read.
    """
    z=np.asarray(z,float); pred=np.asarray(prediction,float)
    pi=np.asarray(inclusion_prob,float); xi=np.asarray(inclusion_indicator,int)
    if not (z.ndim==pred.ndim==pi.ndim==xi.ndim==1 and len(z)>0 and len(z)==len(pred)==len(pi)==len(xi)):
        raise ValueError('Arrays must be same-length vectors')
    if np.any(~np.isfinite(pred)) or np.any(~np.isfinite(pi)) or np.any((pi<=0)|(pi>1)) or np.any((xi!=0)&(xi!=1)) or not 0<=q<=1:
        raise ValueError('Invalid propensities or labels')
    if np.any(~np.isfinite(z[xi==1])):
        raise ValueError('Missing an acquired label')
    # Do not read z_i when xi_i=0 -- those values may be absent/NaN.
    residual=np.zeros(len(z))
    observed=xi==1
    residual[observed]=(z[observed]-pred[observed])/pi[observed]
    point=float(q*(pred.mean()+residual.mean()))
    var_hat=float(q*q/len(z)**2*np.sum(xi[observed]*(1-pi[observed]) *
                (z[observed]-pred[observed])**2 / pi[observed]**2))
    return point,var_hat


def expected_amp_variance(z,pred,pi,q=1.):
    z=np.asarray(z,float);pred=np.asarray(pred,float);pi=np.asarray(pi,float)
    if np.any(pi<=0) or np.any(pi>1) or not(len(z)==len(pred)==len(pi)):
        raise ValueError('Bad arrays')
    return float(q*q/len(z)**2 * np.sum((1-pi)*(z-pred)**2/pi))


def augment_original_result(item:dict,q:float):
    """Postprocess original estimate: keep raw 95% interval, add clipped
    finite-sample interval and distinctly tagged *diagnostic* Wald alternative.
    No original point estimate, sampling method, or budget ledger changes.
    """
    if not item.get('valid'):return item
    assert 0<=q<=1
    out=dict(item)
    original=feasible_ci(item['ci_low'],item['ci_high'],q)
    out['feasible_interval_95']=dict(
        low=original[0] if original else None,high=original[1] if original else None,
        width=original[1]-original[0] if original else None,
        nonempty=original is not None,
        conditional_nominal_coverage_inherited=bool(original is not None))
    method=item['method']; mu=float(item['estimate']);se=float(item['se'])
    if method in ('allowed_srs','whole_srs'):
        # For binomial observations retain Clopper-Pearson, already exact.
        out['paper_style_wald_exploratory_95']=None
    else:
        wald=asymptotic_wald(mu,se,q)
        out['paper_style_wald_exploratory_95']=dict(
          low=wald[0] if wald else None,high=wald[1] if wald else None,
          width=wald[1]-wald[0] if wald else None,
          nonempty=wald is not None,
          status='ASYMPTOTIC_EXPLORATORY_UNCALIBRATED_NOT_AUTHOR_CI')
    target=float(item['target_reviewer_Z'])
    out['feasible_interval_covers_Z']=bool(original and original[0]<=target<=original[1])
    w=out['paper_style_wald_exploratory_95']
    out['wald_exploratory_covers_Z']=None if w is None else bool(w['nonempty'] and w['low']<=target<=w['high'])
    return out
