"""
Main script: greedy RB construction and validation for the ultraweak space-time DPG formulation of the Schrödinger equation with parametrized potential.

Uses hybrid approach of computing FOM solution using static condensation, 
while reducing the coercive normal equations via standard Galerkin projections. 
"""
import os
import matplotlib.pyplot as plt
import ngsolve as ng

from pymor.core.defaults import set_defaults
from pymor.algorithms.greedy import rb_greedy
from pymor.core.logger import set_log_levels

from scenario_spacetime import setup_scenario
from rom_spacetime import DPGProjectedReductor
from utils_spacetime import *

if __name__ == "__main__":
    print("="*60)
    print("      Ultraweak space-time DPG with RB for time-dependent Schrödinger equation with parametrized potential     ")
    print("="*60)
    # Suppress all PyMOR outputs globally
    # set_log_levels({'pymor': 'CRITICAL'}) # used for convergence plot generations to minimize outputs
    # =========================================================================
    # 1. SETUP AND INITIALIZE
    # =========================================================================
    plt.rcParams.update({
            'font.size': 14,         # Controls the default size for plt.annotate()
            'axes.labelsize': 16,    # Controls xlabel and ylabel
            'xtick.labelsize': 14,   # Controls the numbers on the x-axis
            'ytick.labelsize': 14,   # Controls the numbers on the y-axis
            'legend.fontsize': 11,    # Controls the legend 
            'legend.title_fontsize': 11,  # Forces the title to match!
            'legend.labelspacing': 0.3,  # Vertical space between entries
            'legend.handletextpad': 0.4, # Space between the line/marker and the text
            'legend.borderpad': 0.3      # Space between the text and the box edge
        })
    # user_p = int(input("Polynomial Order p (e.g. 3 [default]): ").strip() or "3")
    # user_dp = int(input("Increased test space polynomial order dp (e.g. 1 [default]): ").strip() or "1")
    # user_h = float(input("Mesh size h (e.g. 0.02 [default]): ").strip() or "0.02")
    # n_training = int(input("Size of training set (e.g. 500 [default]): ").strip() or "500")
    # max_ext = int(input("Maximal number of extensions for greedy algorithm (e.g. 30 [default]): ").strip() or "30")
    # compute_true_errs = input("Compute true ROM-FOM errors after greedy generation (for convergence plot) (True [default] / False): ").strip().lower() != "false"
    # n_test = int(input("Size of test set (e.g. 20 [default]): ").strip() or "20")
    # print_samples = input("Print stats for every single test parameter (False [default] / True): ").strip().lower() == "true"


    active_scenario = int(input("Potential scenario from 1 [default] to 7: ").strip() or "1") # scenarios between 1 and 7
    visualizeTestConfigurations = False # turn on for plotting every 10th solution in test set validation loop after greedy construction

    user_p, user_dp, user_h, n_training, max_ext, compute_true_errs, n_test, print_samples =  3,1,1,100,3,True,2,True    # set configurations at once 3, 2, 0.011, 10000, 100, True, 200, False #
    for active_scenario in [1,2,3,4]: # select scenarios here to run
        fom, X, mesh, pot_decomp, ranges, V_mean, energy_product = setup_scenario(user_p, user_dp, user_h, active_scenario = active_scenario)
        param_space = fom.parameters.space(ranges)
        
        USE_L2_PRODUCT = True  
        reductor = DPGProjectedReductor(
            fom, 
            product=fom.products.get('L2') if USE_L2_PRODUCT else energy_product, 
            use_l2_product=USE_L2_PRODUCT
        )
        print(f"\n--- Sampling Greedy Training Set ({n_training} samples) ---")
        training_set = param_space.sample_randomly(n_training)

        # =========================================================================
        # 2. GREEDY SEARCH
        # =========================================================================
        # Relax GS tolerance to ignore non-deterministic solver noise (approx 1e-12) 
        # preventing duplicate snapshots. Default is 1e-13.
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

        # Parameter selection plot
        filename = f'GreedyParams_{active_scenario}_h{user_h}_p{user_p}_dp{user_dp}.pdf'
        mus = greedy_data['max_err_mus']
        plot_greedy_parameters(mus, ranges, filename=filename)

        # print("\nGenerating Convergence Plot...")
        # Optionally compute true errors along greedy path for final convergence plot
        if compute_true_errs:
            plot_mus = mus[:-1] # omit last value to not rescale plot in case of convergence and error drop at last iteration
            u_L2_errs, skel_errs, flux_errs, fom_res, cond_nrs = \
                compute_greedy_true_errors(plot_mus, fom, reductor, rom, X, mesh)

        # =========================================================================
        # 3. VISUALIZATION: CONVERGENCE
        # =========================================================================
        max_errs = greedy_data['max_errs'][:-1]

        plt.figure(figsize=(6, 4.5))

        # Mean Parameter Residual Estimator from Greedy Algorithm (maximal estimator parameters)
        plt.semilogy(range(1, len(max_errs)), max_errs[1:], color='black', linestyle='-', marker='o',
                    linewidth=1.5, markersize=4, zorder=5, label=r"Mean Param. ROM Residual") 
        
        if compute_true_errs:
            # ROM-FOM L2 Error of components
            plt.semilogy(range(1, len(u_L2_errs)), u_L2_errs[1:], color='tab:blue', linestyle='-', marker='s', 
                        linewidth=1.5, markersize=4, alpha=0.9, zorder = 4, label=r"Volume $L^2$-Error ($u$)")
            plt.semilogy(range(1, len(skel_errs)), skel_errs[1:], color='tab:red', linestyle='-', marker='^',
                        linewidth=1.5, markersize=4, alpha=0.9, zorder = 4, label=r"Skeleton $L^2$-Error ($q^+$)")
            plt.semilogy(range(1, len(flux_errs)), flux_errs[1:], color='tab:green', linestyle='-', marker='v',
                        linewidth=1.5, markersize=4, alpha=0.9, zorder = 4, label=r"Flux $L^2$-Error ($q'$)")
            
            # FOM Residuals w.r.t. exact parameter graph norm, not mean parameter graph norm!
            plt.semilogy(range(1, len(fom_res)), fom_res[1:], color='gray', linestyle='-', marker='o', 
                        linewidth=1.5, markersize=4, alpha=0.9, zorder = 3, label=r"FOM Residual")

        plt.xlabel(r"Reduced Basis Size ($N$)")
        plt.ylabel(r"Error and Residual Norms")
        plt.grid(True, which="major", ls="--", alpha=0.6)
        plt.legend(framealpha=0.4)
        
        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"Convergence_spacetime_dpg_rb_{active_scenario}_h{user_h}_p{user_p}_dp{user_dp}_maxExt{max_ext}.pdf"
        full_path = os.path.join(script_dir, filename)
        plt.tight_layout()
        plt.savefig(full_path, bbox_inches='tight')
        # print(f"Saved plot to: {full_path}")

        if compute_true_errs:
            plt.figure(figsize=(6, 4.5))
            plt.semilogy(range(len(cond_nrs)), cond_nrs, color='tab:blue', linestyle='-', marker='o',
                        linewidth=1.5, markersize=4, label=r"ROM Matrix Condition Number")
            plt.xlabel(r"Reduced Basis Size ($N$)")
            plt.ylabel(r"Condition Number")
            plt.grid(True, which="major", ls="--", alpha=0.6)
            plt.legend(framealpha = 0.7)
            
            filename_cond = f"Conditioning_spacetime_dpg_rb_{active_scenario}_h{user_h}_p{user_p}_dp{user_dp}_maxExt{max_ext}.pdf"
            full_path_cond = os.path.join(script_dir, filename_cond)
            plt.tight_layout()
            plt.savefig(full_path_cond, bbox_inches='tight')
            # print(f"Saved conditioning plot to: {full_path_cond}")

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

        stats_abs_error_u_L2 = []
        stats_abs_error_skel = []
        stats_abs_error_flux = []

        stats_rel_error_u_L2 = []
        stats_rel_error_skel = []
        stats_rel_error_flux = []

        stats_error_estim_ratio = []
        stats_cond_N = []

        for k, mu in enumerate(test_set):
            # print(f"\n--- Sample {k+1}/{n_test} : {mu} ---")

            sol_rom, sol_rec, sol_fom, t_rom, t_fom, rom_res, fom_res, cond_N = \
                solve_and_evaluate_sample(mu, fom, rom, reductor)
            
            if cond_N > 0:
                stats_cond_N.append(cond_N)

            # Plotting RB Solution for this sample
            gf_sol_rom = pymor_to_gf(sol_rec.vectors[0], X)
            u_comp_rom = gf_sol_rom.components[0]

            # FOM Solution (u_fom)
            gf_sol_fom = pymor_to_gf(sol_fom.vectors[0], X)
            u_comp_fom = gf_sol_fom.components[0]

            # Error (error)
            gf_error = ng.GridFunction(X)
            gf_error.vec.data = gf_sol_fom.vec - gf_sol_rom.vec
            u_comp_error = gf_error.components[0]

            if visualizeTestConfigurations and k%10 == 0: 
                # Execute User-Requested Plots
                print("   Updating Netgen GUI...")
                
                mu_label = "_".join(f"{v:.0f}" for name in ranges.keys() for v in mu[name])

                # Numerical solution
                ng.Draw(ng.Norm(u_comp_rom), mesh, f"u_rom_magnitude_test_{k+1}_{mu_label}")
                ng.Draw(u_comp_rom.real, mesh, f"u_rom_real_test_{k+1}_{mu_label}")
                ng.Draw(u_comp_rom.imag, mesh, f"u_rom_imag_test_{k+1}_{mu_label}")

                # Exact solution
                ng.Draw(ng.Norm(u_comp_fom), mesh, f"u_fom_magnitude_test_{k+1}_{mu_label}")
                ng.Draw(u_comp_fom.real, mesh, f"u_fom_real_test_{k+1}_{mu_label}")
                ng.Draw(u_comp_fom.imag, mesh, f"u_fom_imag_test_{k+1}_{mu_label}")

                # Exact error
                ng.Draw(ng.Norm(u_comp_error), mesh, f"error_magnitude_test_{k+1}_{mu_label}")
                ng.Draw(u_comp_error.real, mesh, f"error_real_test_{k+1}_{mu_label}")
                ng.Draw(u_comp_error.imag, mesh, f"error_imag_test_{k+1}_{mu_label}")
        
            # Diagnostics
            (speedup, rel_res, res_ratio, abs_error_u_L2, rel_error_u_L2, \
                abs_error_skel, rel_error_skel, abs_error_flux, rel_error_flux, error_estim_ratio) \
                = analyze_result(gf_sol_fom, gf_error, t_rom, t_fom, rom_res, fom_res, mesh, cond_N, print_samples=print_samples)
            
            if speedup > 0: stats_speedup.append(speedup)

            stats_abs_error_u_L2.append(abs_error_u_L2)
            stats_abs_error_skel.append(abs_error_skel)
            stats_abs_error_flux.append(abs_error_flux)
            stats_rel_error_u_L2.append(rel_error_u_L2)
            stats_rel_error_skel.append(rel_error_skel)
            stats_rel_error_flux.append(rel_error_flux)

            stats_rel_res.append(rel_res)
            stats_fom_res.append(fom_res)
            stats_rom_res.append(rom_res)
            stats_error_estim_ratio.append(error_estim_ratio)
            stats_res_ratio.append(res_ratio)
            
        print_summary(stats_speedup, stats_rom_res, stats_rel_res, stats_fom_res, 
                    stats_res_ratio, stats_abs_error_u_L2, stats_abs_error_skel, 
                    stats_abs_error_flux, stats_rel_error_u_L2, stats_rel_error_skel, 
                    stats_rel_error_flux, stats_error_estim_ratio, stats_cond_N, rom, n_test)