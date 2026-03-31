"""
Main script: greedy RB construction and validation for primal DPG for stationary Schrödinger equation with parametrized potential
"""

import time
import os
import matplotlib.pyplot as plt
import ngsolve as ng

from pymor.core.defaults import set_defaults
from pymor.algorithms.greedy import rb_greedy

from scenario_stationary import setup_scenario
from rom_stationary import DPGProjectedReductor
from utils_stationary import analyze_result, print_summary, compute_greedy_true_errors

if __name__ == "__main__":
    print("="*60)
    print("      Primal DPG with RB (Stationary Schrödinger with parametrized potential)      ")
    print("="*60)

    # =========================================================================
    # 1. SETUP AND INITIALIZE
    # =========================================================================
    # user_p = int(input("Polynomial Order p (e.g. 2 [default]): ").strip() or "2")
    # user_h = float(input("Mesh size h (e.g. 0.02 [default]): ").strip() or "0.02")
    # n_training = int(input("Size of training set (e.g. 100 [default]): ").strip() or "100")
    # max_ext = int(input("Maximal number of extensions for greedy algorithm (e.g. 25 [default]): ").strip() or "25")
    # compute_true_errs = input("Compute true ROM-FOM errors after greedy generation (for convergence plot) (True [default] / False): ").strip().lower() != "false"
    # n_test = int(input("Size of test set (e.g. 20 [default]): ").strip() or "20")
    # print_samples = input("Print stats for every single test parameter (False [default] / True): ").strip().lower() == "true"

    user_p, user_h, n_training, max_ext, compute_true_errs, n_test, print_samples = 2, 0.011, 10000, 100, True, 200, False  # default configurations employed for thesis results
    plotSolutions = False # turn on to plot every 10th solution in test set validation loop after greedy construction
    fom, X, mesh, param_ranges, product, pot_regions, pot_key = setup_scenario(user_p, user_h)
    reductor = DPGProjectedReductor(fom, product=product)

    param_space = fom.parameters.space(param_ranges)
    print(f"\n--- Sampling Training Set ({n_training} samples) ---")
    training_set = param_space.sample_randomly(n_training)

    # =========================================================================
    # 2. GREEDY SEARCH
    # =========================================================================
    # Relax GS tolerance to ignore non-deterministic solver noise (approx 1e-12) 
    # prevents duplicate snapshots. Default is 1e-13.
    set_defaults({'pymor.algorithms.gram_schmidt.gram_schmidt.rtol': 1e-11})

    with ng.TaskManager():
        greedy_data = rb_greedy(
            fom, reductor, training_set, 
            max_extensions=max_ext, 
            atol=1e-5, 
            use_error_estimator=True,
            extension_params={'method': 'gram_schmidt'}
        )

    rom = greedy_data['rom']
    print(f"Final ROM Size: {rom.solution_space.dim}")

    max_err_mus = greedy_data['max_err_mus'][:-1]
    
    # Optionally compute true errors along greedy path
    if compute_true_errs:
        u_errs, flux_errs, pure_fom_res, cond_nums = compute_greedy_true_errors(max_err_mus, fom, reductor, rom, X, mesh)
    # =========================================================================
    # 3. VISUALIZATION: CONVERGENCE & CONDITIONING
    # =========================================================================
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 11,
        'legend.title_fontsize': 11,
        'legend.labelspacing': 0.3,
        'legend.handletextpad': 0.4,
        'legend.borderpad': 0.3
    })
    
    max_errs = greedy_data['max_errs'][:-1]
    print("\nGenerating Convergence and Conditioning Plots...")

    # --- Plot 1: Convergence ---
    plt.figure(figsize=(6, 4.5))

    # Residual Estimator from Greedy Algorithm
    plt.semilogy(range(1, len(max_errs)), max_errs[1:], color='black', linestyle='-', marker='o',
                linewidth=1.5, markersize=4, zorder=5, label=r"ROM Residual")
    
    if compute_true_errs:
        plt.semilogy(range(1, len(u_errs)), u_errs[1:], color='tab:blue', linestyle='-', marker='s', 
                    linewidth=1.5, markersize=4, alpha=0.9, zorder=4, label=r"Volume $H^1$-Error ($u$)")
        plt.semilogy(range(1, len(flux_errs)), flux_errs[1:], color='tab:green', linestyle='-', marker='v',
                    linewidth=1.5, markersize=4, alpha=0.9, zorder=4, label=r"Interface Flux $L^2$-Error ($q\cdot n$)")
        plt.semilogy(range(1, len(pure_fom_res)), pure_fom_res[1:], color='gray', linestyle='-', marker='o', 
                    linewidth=1.5, markersize=4, alpha=0.9, zorder=3, label=r"FOM Residual")

    plt.xlabel(r"Reduced Basis Size ($N$)")
    plt.ylabel(r"Error and Residual Norms")
    plt.grid(True, which="major", ls="--", alpha=0.6)
    plt.legend(framealpha=0.4)
    
    # Save Convergence Plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"Convergence_stationary_dpg_rb_h{user_h}_p{user_p}_maxExt{max_ext}.pdf"
    full_path = os.path.join(script_dir, filename)
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches='tight')

    # --- Plot 2: Condition Number ---
    if compute_true_errs:
        plt.figure(figsize=(6, 4.5))
        plt.semilogy(range(1, len(cond_nums)), cond_nums[1:], color='tab:blue', linestyle='-', marker='o',
                    linewidth=1.5, markersize=4, label=r"ROM Matrix Condition Number")
        plt.xlabel(r"Reduced Basis Size ($N$)")
        plt.ylabel(r"Condition Number")
        plt.grid(True, which="major", ls="--", alpha=0.6)
        plt.legend(framealpha=0.7)
        
        # Save Conditioning Plot
        filename_cond = f"Conditioning_stationary_dpg_rb_h{user_h}_p{user_p}_maxExt{max_ext}.pdf"
        full_path_cond = os.path.join(script_dir, filename_cond)
        plt.tight_layout()
        plt.savefig(full_path_cond, bbox_inches='tight')
    
    # =================================================================
    # 4 AUTOMATED TEST LOOP
    # =================================================================
    print("\n" + "="*60)
    print(f"      RUNNING VALIDATION ON {n_test} RANDOM SAMPLES      ")
    print("="*60)
    
    test_set = param_space.sample_randomly(n_test)
    
    stats_speedup = []
    stats_rom_res = []
    stats_rel_res = []
    stats_fom_res = []
    stats_res_ratio = []
    stats_abs_u_error = []
    stats_abs_q_error = []
    stats_rel_u_error = []
    stats_rel_q_error = []
    stats_error_estim_ratio = []

    for k, mu in enumerate(test_set):
        print(f"\n--- Sample {k+1}/{n_test} : {mu} ---")

        # ROM Solve
        t0 = time.time()
        u_rom = rom.solve(mu)
        t_rom = time.time() - t0
        # Reconstruct ROM solution to full space
        u_rec = reductor.reconstruct(u_rom)

        # Error Estimation (ROM Residual)
        rom_res = reductor.assemble_error_estimator().estimate_error(u_rom, mu, rom)[0]
        stats_rom_res.append(rom_res)
        
        # FOM Solve with Residual Check
        t0 = time.time()
        u_fom, fom_res = fom.solve(mu = mu, return_error_estimate=True)
        t_fom = time.time() - t0
        fom_res = fom_res[0]  # Extract scalar from array

        if plotSolutions and k%10 == 0:
            import netgen.gui
            # Plotting RB Solution for this sample
            gfu_rom = ng.GridFunction(X)
            gfu_rom.vec.data = u_rec.vectors[0].real_part.impl.vec
            ng.Draw(gfu_rom.components[0], mesh, f"Sol_Test_{k+1}")
     
        # Diagnostics
        (speedup, rel_res, res_ratio, abs_u_error, 
        rel_u_error, abs_q_error, rel_q_error, error_estim_ratio) = analyze_result(u_rec, u_fom, t_rom, t_fom, 
                               rom_res, fom_res, X, mesh, print_samples = print_samples)
        if speedup > 0:
            stats_speedup.append(speedup)
        stats_rel_res.append(rel_res)
        stats_fom_res.append(fom_res)
        stats_res_ratio.append(res_ratio)
        stats_abs_u_error.append(abs_u_error)
        stats_abs_q_error.append(abs_q_error)
        stats_rel_u_error.append(rel_u_error)
        stats_rel_q_error.append(rel_q_error)
        stats_error_estim_ratio.append(error_estim_ratio)

    print_summary(stats_speedup, stats_rom_res, stats_rel_res, stats_fom_res, 
                  stats_res_ratio, stats_abs_u_error, stats_abs_q_error, 
                  stats_rel_u_error, stats_rel_q_error, stats_error_estim_ratio, rom, n_test)

    # =========================================================================
    # 5. INTERACTIVE MODE
    # =========================================================================
    print("="*60)
    print(f"      Interactive Visualization      ")
    print("="*60)
    
    while True: 
        print("\nInput parameter to solve for and plot (or 'q' to quit).")
        print("Format: 'v1 v2 v3 v4' (e.g., 100 1 1 100)")

        inp = input(">> ").strip()
        if inp.lower() == 'q': break

        try:
            vals = [float(x) for x in inp.split()]
            if len(vals) != 4: raise ValueError
        except ValueError:
            print("   [Error] Invalid input. Please enter 4 numbers.")
            continue
 
        mu = fom.parameters.parse({'blocks': vals})
        print(f"   Solving for {mu} ...")

        # ROM Solve & Reconstruction
        t0 = time.perf_counter()
        u_rom = rom.solve(mu)
        t_rom = time.perf_counter() - t0
        u_rec = reductor.reconstruct(u_rom)
        
        # Error Estimation
        rom_res = reductor.assemble_error_estimator().estimate_error(u_rom, mu, rom)[0]
        
        # FOM Verify
        print("   Running FOM Verify...", end="", flush=True)
        t0 = time.perf_counter()
        u_fom, fom_res = fom.solve(mu = mu, return_error_estimate=True)
        t_fom = time.perf_counter() - t0
        print(" Done.")
        fom_res = fom_res[0]  # Extract scalar from array

        (speedup, rel_res, res_ratio, abs_u_error, 
        rel_u_error, abs_q_error, rel_q_error, error_estim_ratio) = analyze_result(
            u_rec, u_fom, t_rom, t_fom, rom_res, fom_res, X, mesh, print_samples=True)
        
        if plotSolutions:
            # Plot
            gfu = ng.GridFunction(X)
            gfu.vec.data = u_rec.vectors[0].real_part.impl.vec
            ng.Draw(gfu.components[0], mesh, "Interactive_Sol")
    
