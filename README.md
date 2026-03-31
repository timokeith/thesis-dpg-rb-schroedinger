# thesis-dpg-rb-schroedinger
# Model order reduction for space-time discontinuous Petrov-Galerkin approximations of the Schrödinger equation

**Author:** Timo Keith

**Institution:** University of Münster, Germany 

**Date:** March, 2026

## Overview
This repository contains the code and experiments for my Master's thesis, "Model order reduction for space-time discontinuous Petrov-Galerkin approximations of the Schrödinger equation". The code implements:
* the primal DPG method for the stationary Schrödinger equation in 2D using NGSolve
* the ultraweak space-time DPG method for the time-dependent Schrödinger equation with and without potential in 1D using NGSolve
* a greedy reduced basis construction scheme for the stationary Schrödinger equation in 2D using pyMOR and NGSolve
* a greedy reduced basis construction scheme for the time-dependent Schrödinger equation in 1D using pyMOR and NGSolve

## Repository Structure
```text
thesis-dpg-rb-schroedinger/
├──     README.md                                           <-- This instruction manual
├──     requirements.txt                                    <-- Exact package versions
├──     dpg_stationary_schr/                                <-- Code for Chapter 3.2 (stationary Schrödinger equation primal DPG)
│       └── stationary_schrodinger_convergence_studies.py       <-- Complete source file generating convergence plots under uniform mesh refinement
│
├──  	dpg_spacetime_schr_free/                            <-- Code for Chapter 4.2.1 (time-dependent Schrödinger equation without potential ultraweak space-time DPG)
│       └── spacetime_schr_free_convergence_studies.py          <-- Complete source file generating convergence plots under uniform mesh refinement
│
│       dpg_spacetime_schr_potential/                       <-- Code for Chapter 4.2.2 (time-dependent Schrödinger equation with potential ultraweak space-time DPG)
│       └── spacetime_schrodinger_potential_convergence_studies.py		        <-- Complete source file generating convergence plots under uniform mesh refinement
│
├──     dpg_rb_stationary_schr/                             <-- Code for Chapter 5.5.1 (RB for primal DPG stationary Schrödinger equation)
│       ├── fom_stationary.py					                <-- Setup of DPG Model for reduction, full-order solves with NGSolve
│       ├── main_stationary.py					                <-- Main file to run: greedy basis construction, convergence plot generation, test set validation
│       ├── scenario_stationary.py				                <-- Setup of discretization, parametrization, orthonormalization product and Gramian
│       ├── rom_stationary.py					                <-- pyMOR reductor and error estimator for greedy construction
│       └── utils_stationary.py					                <-- Helper functions for output prints, error analysis, and convergence plot generation
│
└──     dpg_rb_spacetime_schr/                              <-- Code for Chapter 5.5.2 (RB for ultraweak DPG time-dependent Schrödinger equation)
        ├── complex_ngsolve_bindings.py				                <-- NGSolve bindings for pyMOR to correctly handle complex valued computations
        ├── fom_spacetime.py					                <-- Setup of DPG Model for reduction, full-order solves with NGSolve
        ├── main_spacetime.py					                <-- Main file to run: greedy basis construction, convergence plot generation, test set validation
        ├── scenario_spacetime.py				                <-- Setup of discretization, parametrization, orthonormalization product and Gramian
        ├── rom_spacetime.py					                <-- pyMOR reductor and error estimator for greedy construction
        └── utils_spacetime.py					                <-- Helper functions for scenario setup, output prints, error analysis, and convergence plot generation

```
## Installation and Requirements
This code was written using Python 3.12. To ensure reproducibility, please install the exact package versions listed in the requirements file.

## Citation
https://doi.org/10.5281/zenodo.19347602
