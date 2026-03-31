"""
Helper functions for geometry, error analysis, output formatting, and convergence plots.
"""

import numpy as np
import ngsolve as ng
from ngsolve import x, y, grad, IfPos

# =============================================================================
# 1. GEOMETRY HELPERS
# =============================================================================
def RectChar(xmin, xmax, ymin, ymax):
    """Coefficient function: 1.0 inside rectangle, 0.0 outside."""
    return  IfPos(x-xmin, 1, 0) * IfPos(xmax-x, 1, 0) * \
            IfPos(y-ymin, 1, 0) * IfPos(ymax-y, 1, 0)

# =============================================================================
# 2. ANALYSIS AND VALIDATION
# =============================================================================
def analyze_result(u_rec, u_fom, t_rom, t_fom, rom_res, fom_res, X, mesh, print_samples=False):
    """
    Compares ROM reconstruction with FOM solution and computes error norms.
    """
    n = ng.specialcf.normal(mesh.dim)
    
    # Setup GridFunctions
    gf_fom = ng.GridFunction(X)
    gf_fom.vec.data = u_fom.vectors[0].real_part.impl.vec

    error = u_fom - u_rec
    gf_error = ng.GridFunction(X)
    gf_error.vec.data = error.vectors[0].real_part.impl.vec

    # Compute FOM solution norms (H1 for u, Interface L2 for q)
    u_fom_comp, q_fom_comp = gf_fom.components
    norm_u_fom_sq = ng.Integrate(u_fom_comp**2 + grad(u_fom_comp)**2, mesh)
    norm_q_fom_sq = abs(ng.Integrate((ng.InnerProduct(q_fom_comp, n))**2 * ng.dx(element_boundary=True, bonus_intorder=20), mesh))

    # Absolute error norms
    err_u, err_q = gf_error.components
    abs_u_err_sq = ng.Integrate(err_u**2 + grad(err_u)**2, mesh)
    abs_q_err_sq = abs(ng.Integrate((ng.InnerProduct(err_q, n))**2 * ng.dx(element_boundary=True, bonus_intorder=20), mesh))
    
    abs_u_error = np.sqrt(abs_u_err_sq)
    abs_q_error = np.sqrt(abs_q_err_sq)
    norm_u_fom = np.sqrt(norm_u_fom_sq)
    norm_q_fom = np.sqrt(norm_q_fom_sq)

    # Relative error norms
    rel_u_error = (abs_u_error / norm_u_fom) * 100 if norm_u_fom > 0 else 0.0
    rel_q_error = (abs_q_error / norm_q_fom) * 100 if norm_q_fom_sq > 0 else 0.0

    # Relative residual, residual ratio and error estimator to true u-error ratio
    rel_res = (rom_res / norm_u_fom) * 100 if norm_u_fom > 0 else 0.0
    res_ratio = rom_res / fom_res if fom_res > 0 else 0.0
    error_estim_ratio = rom_res / abs_u_error if abs_u_error > 0 else 0.0

    speedup = t_fom / t_rom if t_rom > 0 else 0

    if print_samples:
        _print_sample_stats(t_rom, t_fom, speedup, rom_res, rel_res, fom_res, 
                     res_ratio, abs_u_error, rel_u_error, abs_q_error, 
                     rel_q_error, norm_u_fom, error_estim_ratio)

    return speedup, rel_res, res_ratio, abs_u_error, rel_u_error, abs_q_error, rel_q_error, error_estim_ratio

def _print_sample_stats(t_rom, t_fom, speedup, rom_res, rel_res, fom_res, 
                 res_ratio, abs_u_error, rel_u_error, abs_q_error, rel_q_error, 
                 norm_u_fom, error_estim_ratio):
    """
    Print detailed statistics for a single test sample.
    """
    print(f"   [Performance]")
    print(f"   >> ROM Time:                                 {t_rom:.4e}s")
    print(f"   >> FOM Time:                                 {t_fom:.4e}s")
    if speedup > 0:
        print(f"   >> Speedup:                                  {speedup:.1f}x")
    else: 
        print("   >> No speedup factor computable since ROM Time was saved as 0")
    print("-" * 40)
    print(f"   [Accuracy]")
    print(f"   >> Abs Vol H1 Error (u comp):                {abs_u_error:.4e}")
    print(f"   >> Abs Interface L2 Error (q comp):          {abs_q_error:.4e}")
    print("-" * 40)
    print(f"   >> H1-Norm of FOM Solution u:                {norm_u_fom:.4e}")
    print("-" * 40)
    print(f"   >> Rel Vol H1 Error (u comp):                {rel_u_error:.4f} %")
    print(f"   >> Rel Interface L2 Error (q comp):          {rel_q_error:.4f} %")
    print("-" * 40)
    print(f"   [Residuals]")
    print("-" * 40)
    print(f"   >> ROM Residual:                             {rom_res:.4e}")
    print(f"   >> Rel ROM Residual (wrt u norm):            {rel_res:.4f} %")
    print(f"   >> FOM Residual:                             {fom_res:.4e}")
    print(f"   >> Ratio (ROM Res. / FOM Res.):              {res_ratio:.4f}")
    print(f"   >> Ratio (ROM Res. / True u-Error):          {error_estim_ratio:.4f}")
    print("-" * 40)

def print_summary(stats_speedup, stats_rom_res, stats_rel_res, stats_fom_res, 
                stats_res_ratio, stats_abs_u_error, stats_abs_q_error,
                stats_rel_u_error, stats_rel_q_error,
                stats_error_estim_ratio, rom, n_test):
    """
    Prints a statistical summary of the validation run.
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
    print(f"  Median Absolute u-Error:                      {np.median(stats_abs_u_error):.4e}")
    print(f"  Avg    Absolute u-Error:                      {np.mean(stats_abs_u_error):.4e}")
    print(f"  Min    Absolute u-Error:                      {np.min(stats_abs_u_error):.4e}")
    print(f"  Max    Absolute u-Error:                      {np.max(stats_abs_u_error):.4e}")
    print("-"*60)
    print(f"  Median Absolute q-Error:                      {np.median(stats_abs_q_error):.4e}")
    print(f"  Avg    Absolute q-Error:                      {np.mean(stats_abs_q_error):.4e}")
    print(f"  Min    Absolute q-Error:                      {np.min(stats_abs_q_error):.4e}")
    print(f"  Max    Absolute q-Error:                      {np.max(stats_abs_q_error):.4e}")
    print("-"*60)
    print(f"  Median Relative u-Error:                      {np.median(stats_rel_u_error):.4f} %")
    print(f"  Avg    Relative u-Error:                      {np.mean(stats_rel_u_error):.4f} %")
    print(f"  Min    Relative u-Error:                      {np.min(stats_rel_u_error):.4f} %")
    print(f"  Max    Relative u-Error:                      {np.max(stats_rel_u_error):.4f} %")
    print("-"*60)
    print(f"  Median Relative q-Error:                      {np.median(stats_rel_q_error):.4f} %")
    print(f"  Avg    Relative q-Error:                      {np.mean(stats_rel_q_error):.4f} %")
    print(f"  Min    Relative q-Error:                      {np.min(stats_rel_q_error):.4f} %")
    print(f"  Max    Relative q-Error:                      {np.max(stats_rel_q_error):.4f} %")
    print("-"*60)
    print(f"  Median ROM Residual:                          {np.median(stats_rom_res):.4e}")
    print(f"  Avg    ROM Residual:                          {np.mean(stats_rom_res):.4e}")
    print(f"  Min    ROM Residual:                          {np.min(stats_rom_res):.4e}")
    print(f"  Max    ROM Residual:                          {np.max(stats_rom_res):.4e}")
    print("-"*60)
    print(f"  Median Relative Residual:                     {np.median(stats_rel_res):.4f} %")
    print(f"  Avg    Relative Residual:                     {np.mean(stats_rel_res):.4f} %")
    print(f"  Min    Relative Residual:                     {np.min(stats_rel_res):.4f} %")
    print(f"  Max    Relative Residual:                     {np.max(stats_rel_res):.4f} %")
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
    print(f"  Median Ratio ROM Resid. / True u-Error:       {np.median(stats_error_estim_ratio):.4f}")
    print(f"  Avg    Ratio ROM Resid. / True u-Error:       {np.mean(stats_error_estim_ratio):.4f}")
    print(f"  Min    Ratio ROM Resid. / True u-Error:       {np.min(stats_error_estim_ratio):.4f}")
    print(f"  Max    Ratio ROM Resid. / True u-Error:       {np.max(stats_error_estim_ratio):.4f}")


# =============================================================================
# 3. TRUE ERROR COMPUTATION FOR CONVERGENCE ANALYSIS
# =============================================================================
def compute_greedy_true_errors(max_err_mus, fom, reductor, rom, X, mesh):
    """
    Compute true X-norm errors, Interface errors, FOM residuals, and ROM condition numbers.
    """
    
    full_basis = reductor.bases['RB']
    n_mus = len(max_err_mus)
    u_errs, flux_errs, pure_fom_res, cond_nums = [], [], [], []
    
    print(f"\n--- Computing True Errors at {n_mus} Greedy-Selected Parameters ---")
    n = ng.specialcf.normal(mesh.dim)
    
    for i, mu in enumerate(max_err_mus):
        basis_size = i
        
        # Solve FOM once
        u_fom, fom_resid = fom.solve(mu, return_error_estimate=True)
        pure_fom_res.append(fom_resid[0])
        
        if basis_size == 0:
            # No basis: ROM solution is zero
            gf_fom = ng.GridFunction(X)
            gf_fom.vec.data = u_fom.vectors[0].real_part.impl.vec
            u_comp, q_comp = gf_fom.components
            
            err_u_sq = ng.Integrate(u_comp**2 + grad(u_comp)**2, mesh)
            err_flux_sq = abs(ng.Integrate((ng.InnerProduct(q_comp, n))**2 * ng.dx(element_boundary=True, bonus_intorder=20), mesh))
            
            u_errs.append(np.sqrt(err_u_sq))
            flux_errs.append(np.sqrt(err_flux_sq))
            cond_nums.append(1.0) # placeholder for N=0
            print(f"  Iteration {i}: N=0, Volume H1-error = {u_errs[-1]:.4e}, Flux error = {flux_errs[-1]:.4e}")
            continue
        
        # Assemble full reduced system
        A_N = rom.operator.assemble(mu).matrix
        f_N = rom.rhs.as_range_array(mu).to_numpy().flatten()
        
        # Restrict to first `basis_size` components
        A_i = A_N[:basis_size, :basis_size]
        f_i = f_N[:basis_size]
        
        # Compute condition number
        cond_nums.append(np.linalg.cond(A_i))
        
        # Solve restricted reduced system
        u_i = np.linalg.solve(A_i, f_i)
        
        # Reconstruct using first i basis vectors
        u_rec = full_basis[:basis_size].lincomb(u_i)
        
        # Compute norms of error
        error = u_fom - u_rec
        gf_error = ng.GridFunction(X)
        gf_error.vec.data = error.vectors[0].real_part.impl.vec
        
        err_u, err_q = gf_error.components
        err_u_sq = ng.Integrate(err_u**2 + grad(err_u)**2, mesh)
        err_flux_sq = abs(ng.Integrate((ng.InnerProduct(err_q, n))**2 * ng.dx(element_boundary=True, bonus_intorder=20), mesh))
        
        u_errs.append(np.sqrt(err_u_sq))
        flux_errs.append(np.sqrt(err_flux_sq))
        
        print(f"  Iteration {i}: N={basis_size}, Vol error = {u_errs[-1]:.4e}, Flux error = {flux_errs[-1]:.4e}, Cond = {cond_nums[-1]:.2e}")
    
    return u_errs, flux_errs, pure_fom_res, cond_nums