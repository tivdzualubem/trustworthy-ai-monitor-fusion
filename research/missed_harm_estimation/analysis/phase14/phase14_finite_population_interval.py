#!/usr/bin/env python3
"""Exact conservative design interval for a finite allowed-response population.

Samples MUST be a true simple random sample WITHOUT replacement among all
allowed cases in a predeclared fixed frame. The interval estimates prevalence
of a FIXED REVIEWER PROTOCOL endpoint Z; a qualified reviewer decision does
NOT automatically equal latent policy harmfulness Y. No new data collected.
"""
from __future__ import annotations
from scipy.stats import hypergeom


def exact_allowed_srswor_interval(frame_N:int, allowed_M:int, sample_n:int,
                                 harmful_x:int, uncertain_u:int=0, alpha:float=0.05):
    vals=(frame_N,allowed_M,sample_n,harmful_x,uncertain_u)
    if any(not isinstance(v,int) or isinstance(v,bool) for v in vals):
        raise ValueError('Counts must be integers')
    if not (frame_N>=allowed_M>=sample_n>0 and 0<=harmful_x<=sample_n
            and 0<=uncertain_u<=sample_n-harmful_x and 0<alpha<1):
        raise ValueError('Inconsistent frame, sample, or outcome counts')
    # Exact inversion with a finite-population hypergeometric law; these are
    # Clopper-Pearson-type central tail acceptance regions for unknown K.
    def accepted(K,x):
        return (float(hypergeom.sf(x-1,allowed_M,K,sample_n)) >= alpha/2-1e-12
            and float(hypergeom.cdf(x,allowed_M,K,sample_n)) >= alpha/2-1e-12)
    lows=[K for K in range(allowed_M+1) if accepted(K,harmful_x)]
    highs=[K for K in range(allowed_M+1) if accepted(K,harmful_x+uncertain_u)]
    if not lows or not highs:raise AssertionError('Hypergeometric inversion empty')
    Klo=min(lows);Khi=max(highs)
    return {'K_min':Klo,'K_max':Khi,
       'A_lower':Klo/frame_N,'A_upper':Khi/frame_N,
       'allowed_fraction':allowed_M/frame_N,
       'reference_endpoint':'FIXED_REVIEWER_PROTOCOL_Z_NOT_LATENT_Y',
       'design':'SRSWOR_OF_ALLOWED_FIXED_FRAME',
       'unresolved_treatment':'UNCERTAIN_COUNT_AS_0_FOR_LOWER_AND_1_FOR_UPPER',
       'conditional_assumptions':'KNOWN_N_M_UNIFORM_WOR_SAMPLE_FROZEN_REFERENCE_LABEL_PROTOCOL'}
