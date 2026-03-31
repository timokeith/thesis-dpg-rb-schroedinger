"""
Helper functions for global physics, performance error analysis, output formatting and convergence plots.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ngsolve as ng
from ngsolve import grad, IfPos

# =============================================================================
# Global Physics and Operators
# =============================================================================
def d_t(func): 
    """Temporal derivative (y-component gradient)"""
    return grad(func)[1]

def d_x(func): 
    """Spatial derivative (x-component gradient)"""
    return grad(func)[0]

def d_xx(func): 
    """Second spatial derivative (Hessian[0,0])"""
    return func.Operator("hesse")[0,0]

def Adj_A(func, V=0):
    """
    Global Adjoint Operator: A* v = i dt v - dxx v + V v
    """
    return 1j * d_t(func) - d_xx(func) + V * func

def reconstruct_potential(mu, pot_decomp):
    """Reconstructs the parameter-dependent potential V(mu)."""
    V_mu = 0
    for key, V_comp in pot_decomp:
        V_mu += mu.get(key)[0] * V_comp
    return V_mu

def solve_and_evaluate_sample(mu, fom, rom, reductor):
    """Executes ROM and FOM solves and computes all error estimators for a given parameter."""
    # ROM Solve
    t0 = time.time()
    sol_rom = rom.solve(mu)
    t_rom = time.time() - t0

    A_N = rom.operator.assemble(mu).matrix
    cond_N = np.linalg.cond(A_N) if A_N.size > 0 else 0.0

    sol_rec = reductor.reconstruct(sol_rom)

    # ROM Error Estimation
    rom_res = reductor.error_estimator.estimate_error(sol_rom, mu, rom)[0]
    
    # FOM Solve
    t0 = time.time()
    sol_fom, fom_res_arr = fom.solve(mu=mu, return_error_estimate=True)
    fom_res = fom_res_arr[0]
    t_fom = time.time() - t0

    return sol_rom, sol_rec, sol_fom, t_rom, t_fom, rom_res, fom_res, cond_N

# =============================================================================
# Extract NGSolve GridFunction from pyMOR vector
# =============================================================================
def pymor_to_gf(pymor_vec, space):
    """Convert a natively complex pyMOR vector to an NGSolve GridFunction."""
    gf = ng.GridFunction(space)
    gf.vec.data = pymor_vec.impl.vec
    return gf

# =====================================================================
# Characteristic function on [a,b] via IfPos
# =====================================================================
def char_func(var, a, b):
    """Indicator function"""
    return IfPos(var - a, 1.0, 0.0) * IfPos(b - var, 1.0, 0.0)
    
# =============================================================================
# Analysis and Validation
# =============================================================================
def compute_3_norms(gf, mesh):
    """Universal helper to extract L2, Skeleton, and Flux norms."""
    u_comp = gf.components[0]
    norm_u_L2 = np.sqrt(max(ng.Integrate(u_comp * ng.Conj(u_comp), mesh, order=20).real, 0.0))

    n = ng.specialcf.normal(mesh.dim)
    is_vert = ng.IfPos(n[0]**2 - n[1]**2, 1.0, 0.0)

    # Skeleton trace norm
    skel_cf = gf.components[1] * ng.Conj(gf.components[1])
    norm_skel = np.sqrt(abs(ng.Integrate(skel_cf * ng.dx(element_boundary=True, bonus_intorder=20), mesh)))

    # Flux norm on vertical edges
    flux_cf = is_vert * gf.components[2] * ng.Conj(gf.components[2])
    norm_flux = np.sqrt(abs(ng.Integrate(flux_cf * ng.dx(element_boundary=True, bonus_intorder=20), mesh)))

    return norm_u_L2, norm_skel, norm_flux

def analyze_result(gf_sol_fom, gf_error, t_rom, t_fom, rom_res,
                   fom_res, mesh, cond_N, print_samples=False):
    """
    Compares ROM vs FOM using L2 norms and Parameter-Dependent Graph Norms.
    """
    norm_u_L2, norm_skel, norm_flux = compute_3_norms(gf_sol_fom, mesh)
    abs_error_u_L2, abs_error_skel, abs_error_flux = compute_3_norms(gf_error, mesh)

    rel_error_u_L2 = (abs_error_u_L2 / norm_u_L2 * 100) if norm_u_L2 > 0 else 0.0
    rel_error_skel = (abs_error_skel / norm_skel * 100) if norm_skel > 0 else 0.0
    rel_error_flux = (abs_error_flux / norm_flux * 100) if norm_flux > 0 else 0.0

    rel_res = (rom_res / norm_u_L2) * 100
    error_estim_ratio = rom_res / abs_error_u_L2 if abs_error_u_L2 > 0 else 0.0

    res_ratio = rom_res / fom_res if fom_res > 0 else 0.0

    speedup = t_fom / t_rom if t_rom > 0 else 0.0

    if print_samples:
        _print_sample_stats(t_rom, t_fom, speedup, rom_res, rel_res, fom_res, res_ratio, 
                            abs_error_u_L2, rel_error_u_L2, abs_error_skel, rel_error_skel, abs_error_flux, rel_error_flux, error_estim_ratio, cond_N)

    return speedup, rel_res, res_ratio, abs_error_u_L2, rel_error_u_L2, \
        abs_error_skel, rel_error_skel, abs_error_flux, rel_error_flux, error_estim_ratio


def _print_sample_stats(t_rom, t_fom, speedup, rom_res, rel_res, fom_res, res_ratio, 
                            abs_error_u_L2, rel_error_u_L2, abs_error_skel, rel_error_skel, abs_error_flux, rel_error_flux, error_estim_ratio, cond_N):
    """
    Print detailed statistics for a single test sample.
    """
    print(f"   >> ROM Matrix Condition Number:              {cond_N:.2e}")
    print(f"   [Performance]")
    print(f"   >> ROM Time:                                 {t_rom:.4e}s")
    print(f"   >> FOM Time:                                 {t_fom:.4e}s")
    if speedup > 0:
        print(f"   >> Speedup:                                  {speedup:.1f}x")
    else: 
        print("   >> No speedup factor computable since ROM Time was saved as 0")
    print("-" * 40)
    print(f"   [Accuracy]")
    print(f"   >> Abs ROM-FOM L2-error:                     {abs_error_u_L2:.4e}")
    print(f"   >> Abs ROM-FOM q_plus L2-error:              {abs_error_skel:.4e}")
    print(f"   >> Abs ROM-FOM q_prime L2-error:             {abs_error_flux:.4e}")
    print("-" * 40)
    print(f"   >> Rel ROM-FOM L2-error:                     {rel_error_u_L2:.4f} %")
    print(f"   >> Rel ROM-FOM q_plus L2-error:              {rel_error_skel:.4f} %")
    print(f"   >> Rel ROM-FOM q_prime L2-error:             {rel_error_flux:.4f} %")
    print("-" * 40)
    print(f"   [Residuals]")
    print("-" * 40)
    print(f"   >> ROM Residual:                             {rom_res:.4e}")
    print(f"   >> Rel. ROM Residual:                        {rel_res:.4f} %")
    print(f"   >> FOM Residual:                             {fom_res:.4e}")
    print(f"   >> Res. Ratio (ROM Res. / FOM Res.):         {res_ratio:.4f}")
    print("-" * 40)
    print(f"   >> Estim. Ratio (ROM Resid. / L2 Error):     {error_estim_ratio:.4f}")
    print("-" * 40)

def print_summary(stats_speedup, stats_rom_res, stats_rel_res, stats_fom_res, 
                  stats_res_ratio, stats_abs_error_u_L2, stats_abs_error_skel, 
                  stats_abs_error_flux, stats_rel_error_u_L2, stats_rel_error_skel, 
                  stats_rel_error_flux, stats_error_estim_ratio, stats_cond_N, rom, n_test):
    """
    Prints a statistical summary of the validation run. Used to generate table values in thesis
    """
    print("\n" + "="*60)
    print(f"ROM Size: {rom.solution_space.dim}")
    print("="*60)
    print(f"      FINAL STATISTICS FOR {n_test} RANDOM TEST SAMPLES      ")
    print("="*60)
    if len(stats_speedup) == 0:
        print(f"  WARNING: No valid speedup measurements collected.")
    else:
        print(f"  Median Speedup:                               {np.median(stats_speedup):.1f}x")
        print(f"  Avg    Speedup:                               {np.mean(stats_speedup):.1f}x")
        print(f"  Min    Speedup:                               {np.min(stats_speedup):.1f}x")
        print(f"  Max    Speedup:                               {np.max(stats_speedup):.1f}x")
    print("-"*60)
    print(f"  Median Absolute L2-Error:                     {np.median(stats_abs_error_u_L2):.4e}")
    print(f"  Avg    Absolute L2-Error:                     {np.mean(stats_abs_error_u_L2):.4e}")
    print(f"  Min    Absolute L2-Error:                     {np.min(stats_abs_error_u_L2):.4e}")
    print(f"  Max    Absolute L2-Error:                     {np.max(stats_abs_error_u_L2):.4e}")
    print("-"*60)
    print(f"  Median Absolute qplus L2-Error:               {np.median(stats_abs_error_skel):.4e}")
    print(f"  Avg    Absolute qplus L2-Error:               {np.mean(stats_abs_error_skel):.4e}")
    print(f"  Min    Absolute qplus L2-Error:               {np.min(stats_abs_error_skel):.4e}")
    print(f"  Max    Absolute qplus L2-Error:               {np.max(stats_abs_error_skel):.4e}")
    print("-"*60)
    print(f"  Median Absolute qprime L2-Error:              {np.median(stats_abs_error_flux):.4e}")
    print(f"  Avg    Absolute qprime L2-Error:              {np.mean(stats_abs_error_flux):.4e}")
    print(f"  Min    Absolute qprime L2-Error:              {np.min(stats_abs_error_flux):.4e}")
    print(f"  Max    Absolute qprime L2-Error:              {np.max(stats_abs_error_flux):.4e}")
    print("-"*60)
    print(f"  Median Relative L2-Error:                     {np.median(stats_rel_error_u_L2):.4f} %")
    print(f"  Avg    Relative L2-Error:                     {np.mean(stats_rel_error_u_L2):.4f} %")
    print(f"  Min    Relative L2-Error:                     {np.min(stats_rel_error_u_L2):.4f} %")
    print(f"  Max    Relative L2-Error:                     {np.max(stats_rel_error_u_L2):.4f} %")
    print("-"*60)
    print(f"  Median Relative qplus L2-Error:               {np.median(stats_rel_error_skel):.4f} %")
    print(f"  Avg    Relative qplus L2-Error:               {np.mean(stats_rel_error_skel):.4f} %")
    print(f"  Min    Relative qplus L2-Error:               {np.min(stats_rel_error_skel):.4f} %")
    print(f"  Max    Relative qplus L2-Error:               {np.max(stats_rel_error_skel):.4f} %")
    print("-"*60)
    print(f"  Median Relative qprime L2-Error:              {np.median(stats_rel_error_flux):.4f} %")
    print(f"  Avg    Relative qprime L2-Error:              {np.mean(stats_rel_error_flux):.4f} %")
    print(f"  Min    Relative qprime L2-Error:              {np.min(stats_rel_error_flux):.4f} %")
    print(f"  Max    Relative qprime L2-Error:              {np.max(stats_rel_error_flux):.4f} %")
    print("-"*60)
    print(f"  Median ROM Residual:                          {np.median(stats_rom_res):.4e}")
    print(f"  Avg    ROM Residual:                          {np.mean(stats_rom_res):.4e}")
    print(f"  Min    ROM Residual:                          {np.min(stats_rom_res):.4e}")
    print(f"  Max    ROM Residual:                          {np.max(stats_rom_res):.4e}")
    print("-"*60)
    print(f"  Median Relative ROM Residual:                 {np.median(stats_rel_res):.4f} %")
    print(f"  Avg    Relative ROM Residual:                 {np.mean(stats_rel_res):.4f} %")
    print(f"  Min    Relative ROM Residual:                 {np.min(stats_rel_res):.4f} %")
    print(f"  Max    Relative ROM Residual:                 {np.max(stats_rel_res):.4f} %")
    print("-"*60)
    print(f"  Median FOM Residual:                          {np.median(stats_fom_res):.4e}")
    print(f"  Avg    FOM Residual:                          {np.mean(stats_fom_res):.4e}")
    print(f"  Min    FOM Residual:                          {np.min(stats_fom_res):.4e}")
    print(f"  Max    FOM Residual:                          {np.max(stats_fom_res):.4e}")
    print("-"*60)
    print(f"  Median ROM/FOM Residual Ratio:                {np.median(stats_res_ratio):.4f}")
    print(f"  Avg    ROM/FOM Residual Ratio:                {np.mean(stats_res_ratio):.4f}")
    print(f"  Min    ROM/FOM Residual Ratio:                {np.min(stats_res_ratio):.4f}")
    print(f"  Max    ROM/FOM Residual Ratio:                {np.max(stats_res_ratio):.4f}")
    print("-"*60)
    print(f"  Median Ratio ROM Resid. / ROM-FOM L2-Error:   {np.median(stats_error_estim_ratio):.4f}")
    print(f"  Avg    Ratio ROM Resid. / ROM-FOM L2-Error:   {np.mean(stats_error_estim_ratio):.4f}")
    print(f"  Min    Ratio ROM Resid. / ROM-FOM L2-Error:   {np.min(stats_error_estim_ratio):.4f}")
    print(f"  Max    Ratio ROM Resid. / ROM-FOM L2-Error:   {np.max(stats_error_estim_ratio):.4f}")
    print("-"*60)
    print(f"  Median ROM Matrix Condition Number:           {np.median(stats_cond_N):.2e}")
    print(f"  Avg    ROM Matrix Condition Number:           {np.mean(stats_cond_N):.2e}")
    print(f"  Min    ROM Matrix Condition Number:           {np.min(stats_cond_N):.2e}")
    print(f"  Max    ROM Matrix Condition Number:           {np.max(stats_cond_N):.2e}")


# =============================================================================
# True error computation for convergence analysis
# =============================================================================
def compute_greedy_true_errors(max_err_mus, fom, reductor, rom, X, mesh):
    """
    Compute errors along the greedy path.
    """
    
    full_basis = reductor.bases['RB']
    n_mus = len(max_err_mus)
    
    u_L2_errors, skel_errors, flux_errors = [], [], []
    fom_residuals = []
    cond_nrs = []
    
    print(f"\n--- Computing True Errors at {n_mus} Greedy-Selected Parameters ---")
    
    for basis_size, mu in enumerate(max_err_mus):        
        # Solve FOM
        sol_fom, fom_resid = fom.solve(mu, return_error_estimate = True)
        fom_residuals.append(fom_resid[0])

        # Reconstruct ROM solution using truncated basis
        if basis_size == 0:
            rom_sol_rec_vec = fom.solution_space.zeros()
        else:
            A_N = rom.operator.assemble(mu).matrix[:basis_size, :basis_size]
            cond_N = np.linalg.cond(A_N)
            cond_nrs.append(cond_N)
            f_N = rom.rhs.as_range_array(mu).to_numpy().flatten()[:basis_size]
            u_i = np.linalg.solve(A_N, f_N)

            rom_sol_rec_vec = full_basis[:basis_size].lincomb(u_i)
            gf_rom = pymor_to_gf(rom_sol_rec_vec.vectors[0], X)

        gf_rom = pymor_to_gf(rom_sol_rec_vec.vectors[0], X)

        # Compute Error Field 
        gf_fom = pymor_to_gf(sol_fom.vectors[0], X)
        
        # Create explicit difference GridFunction
        gf_err = ng.GridFunction(X)
        gf_err.vec.data = gf_fom.vec - gf_rom.vec
        err_u_L2, err_skel, err_flux = compute_3_norms(gf_err, mesh)
        
        u_L2_errors.append(err_u_L2)
        skel_errors.append(err_skel)
        flux_errors.append(err_flux)
      
    return u_L2_errors, skel_errors, flux_errors, fom_residuals, cond_nrs

# =============================================================================
# Additional visualizations of selected parameters from greedy algorithm
# =============================================================================
def plot_greedy_parameters(mus, ranges, filename=None):
    """Plot greedy-selected parameters in the parameter domain."""
        
    param_names = list(ranges.keys())
    N = len(param_names)
    
    vals = {name: [mu[name][0] for mu in mus] for name in param_names}
    order = np.arange(len(mus))
    
    # =========================================================================
    # 2D Case: Standard Single Scatter Plot
    # =========================================================================
    if N == 2:
        p1_name, p2_name = param_names
        p1_vals = vals[p1_name]
        p2_vals = vals[p2_name]
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        
        sc = ax.scatter(p1_vals, p2_vals, c=order, cmap='viridis', 
                        s=40, edgecolors='none', linewidths=0.5, zorder=2)
        
        ax.set_xlabel(r'$\mu_1$', labelpad=10)
        ax.set_ylabel(r'$\mu_2$', labelpad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Selection order', labelpad=15)
        plt.tight_layout(pad=0.5)
    
    # =========================================================================
    # >= 3D Case: Scatterplot Matrix (Pairplot) 
    # =========================================================================
    elif N >= 3:
        # Create an N x N grid of subplots
        fig, axes = plt.subplots(N, N, figsize=(3 * N, 3 * N))

        for i in range(N):
            for j in range(N):
                ax = axes[i, j]
                p_i = param_names[i]
                p_j = param_names[j]

                if i == j:
                    # Diagonal: 1D Histogram showing sampling density
                    ax.hist(vals[p_i], bins=15, color='gray', alpha=0.7)
                    ax.set_yticks([])  # Hide y-ticks for histograms to keep it clean
                else:
                    # Off-diagonal: 2D Projection scatter plots
                    sc = ax.scatter(vals[p_j], vals[p_i], c=order, cmap='viridis',
                                    s=30, edgecolors='none', alpha=0.8)

                if i == N - 1:
                    ax.set_xlabel(fr'$\mu_{j+1}$ ({p_j})')
                else:
                    ax.set_xticklabels([]) # Hide inner x-ticks

                if j == 0:
                    ax.set_ylabel(fr'$\mu_{i+1}$ ({p_i})')
                else:
                    if i != j: ax.set_yticklabels([]) # Hide inner y-ticks

        # Add one shared colorbar to the side of the matrix
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
        cbar.set_label('Selection order', labelpad=15)
    
    else:
        print(f"Skipping plot: Cannot plot {N}D parameters.")
        return

    # Save logic
    if filename:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.4)
    
    plt.close(fig)