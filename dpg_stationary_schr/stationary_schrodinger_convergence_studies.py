import os
import ngsolve as ng
ng.ngsglobals.msg_level = 0
from ngsolve import dx, x, y, grad, sin, cos, pi, IfPos, sqrt
from ngsolve.meshes import MakeStructured2DMesh
import matplotlib.pyplot as plt

# List of N values
N_values = [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

run_configs = [
    (1, 1, 'red','v', r'$p=1, \ \Delta p=1$'),
    (1, 2, 'darkred', 'o', r'$p=1, \ \Delta p=2$'),
    (1, 4, 'blue','x', r'$p=1, \ \Delta p=4$'),
    (2, 1, 'green','v', r'$p=2, \ \Delta p=1$'),
    (2, 2, 'darkgreen','o', r'$p=2, \ \Delta p=2$'),
    (2, 4, 'black', 'x',r'$p=2, \ \Delta p=4$'),
]
printResults = False    # print results while running loop
showPlots = False   # show plots at highest resolution for each configuration before continuing the loop

for case in [1,2]:  # cases 1 and 2 are included in the thesis
    stored_results = {}
    for p, dp, color, marker, label in run_configs:     
        stored_results[(p, dp)] = {'N': [], 'err': [], 'estim': [], 'resid': [], 'flux_err': []}
        print("\n" + "#"*60)
        print(f"STARTING SIMULATIONS FOR CASE {case}: POLYNOMIAL ORDER p={p}, dp={dp}  (Max N={N_values[-1]})")
        print("#"*60)

        for current_n in N_values:
            print("\n" + "="*60)
            print(f"Running case {case}: p={p}, dp={dp}, N={current_n}  ...")
            print("="*60)
            # ------------------------------------------------------------------------------
            # 1. Mesh, Parameters and Spaces
            # ------------------------------------------------------------------------------
            mesh = MakeStructured2DMesh(nx=current_n, ny=current_n, quads=False)

            U = ng.H1(mesh, order=p+1, dirichlet='.*')
            Q = ng.HDiv(mesh, order=p, orderinner=0)
            Y = ng.L2(mesh, order=p + dp)
            X = U * Q
            XY = X * Y

            n = ng.specialcf.normal(mesh.dim)
            u, q, e = XY.TrialFunction()
            w, r, v = XY.TestFunction()
            qn, rn = ng.InnerProduct(q,n), ng.InnerProduct(r,n)

            # ------------------------------------------------------------------------------
            # 2. Exact Solution RHS definition
            # ------------------------------------------------------------------------------              
            # First case included in the thesis
            if case == 1: 
                V_pot = 100000.0 * IfPos(y - 0.3*x - 0.37, 1.0, 0.0)
                u_exact = sin(pi*x) * sin(pi*y)

            # Second test case included in the thesis
            elif case == 2: 
                dist = sqrt((x-0.5)**2 + (y-0.5)**2)
                V_pot = 50000.0 / (dist + 1e-12)**0.4
                u_exact = x*(1-x)*y*(1-y)

            elif case == 3:
                V_pot = 10000.0 * sin(20*pi*x) * sin(20*pi*y)
                u_exact = sin(pi*x) * sin(pi*y)

            elif case == 4:
                V_pot = 50000.0 * IfPos(sin(5*pi*x)*cos(5*pi*y), 1.0, 0.0)
                u_exact = x*(1-x)*y*(1-y)
            
            elif case == 5:
                V_pot = -19.0 * IfPos(y - 0.3*x - 0.37, 1.0, 0.0)
                u_exact = sin(pi*x) * sin(pi*y)

            elif case == 6:
                dist = sqrt((x-0.5)**2 + (y-0.5)**2)
                V_pot = 1.0 / (dist + 1e-12)**0.4
                u_exact = x*(1-x)*y*(1-y)

            elif case == 7:
                V_pot = -18.0 * sin(10*pi*x)**2 * sin(10*pi*y)**2
                u_exact = sin(pi*x) * sin(pi*y)

            elif case == 8:
                V_pot = IfPos(sin(4*pi*x)*cos(4*pi*y), -19.0, 5.0)
                u_exact = x*(1-x)*y*(1-y)
            
            elif case == 9:
                V_pot = 100.0 * IfPos(y - 0.3*x - 0.37, 1.0, 0.0)
                u_exact = sin(pi*x) * sin(pi*y)
            
            elif case == 10:
                V_pot = -18.0 * sin(15*pi*x)**2 * sin(15*pi*y)**2
                u_exact = sin(pi*x) * sin(pi*y)

            # -Delta u + V u = f
            rhs_func = -(u_exact.Diff(x).Diff(x) + u_exact.Diff(y).Diff(y)) + V_pot * u_exact

            # ------------------------------------------------------------------------------
            # 3. Assembly
            # ------------------------------------------------------------------------------
            # We assemble the full matrix A = [ R_Y  B ]
            #                                 [ B^T  0 ]
            # condense=True for static condensation
            print("Assembling system...")
            A = ng.BilinearForm(XY, symmetric=True, condense=True)
            bonus_intorder_pot = 20 

            # Block R_Y: (e, v)_Y
            A += (e * v + grad(e) * grad(v)) * dx

            # Block B: b((u,q), v)
            A += (grad(u) * grad(v)) * dx + (qn * v) * dx(element_boundary=True)
            A += (V_pot * u * v) *dx(bonus_intorder=bonus_intorder_pot)

            # Block B^T: b((w,r), e)^T
            A += (grad(e) * grad(w)) * dx + (e * rn) * dx(element_boundary=True)
            A += (V_pot * e * w) * dx(bonus_intorder=bonus_intorder_pot)

            bonus_intorder_f = 20
            f = ng.LinearForm(XY)
            f += rhs_func * v * dx(bonus_intorder=bonus_intorder_f)
            
            # ------------------------------------------------------------------------------
            # 4. Solve with Non-Homogeneous BCs
            # ------------------------------------------------------------------------------
            with ng.TaskManager():
                A.Assemble()
                f.Assemble()   
                gfu = ng.GridFunction(XY)
                print(f"\nAssembling done.")
                print("-"*40)

                print("Solving with static condensation...")
                # Set Dirichlet BC
                gfu.components[0].Set(u_exact, ng.BND)

                # Compute residual from Dirichlet boundary lift
                r = f.vec.CreateVector()
                A.Apply(gfu.vec, r)

                rhs = f.vec.CreateVector()
                rhs.data = f.vec - r

                # Static Condensation      
                res = rhs.CreateVector()
                res.data = rhs
                res.data += A.harmonic_extension_trans * rhs

                SchurInv = A.mat.Inverse(freedofs=XY.FreeDofs(coupling=True), inverse='sparsecholesky')

                correction = gfu.vec.CreateVector()
                correction.data = SchurInv * res
                correction.data += A.harmonic_extension * correction
                correction.data += A.inner_solve * rhs

                # Add correction
                gfu.vec.data += correction
                print("\nSolve done.")
                # ------------------------------------------------------------------------------
                # Post-Processing and Diagnostics
                # ------------------------------------------------------------------------------
                u_sol, q_sol, e_sol = gfu.components[0], gfu.components[1], gfu.components[2]

                if printResults:
                    print("="*40)
                    print(f"RESULTS: Case {case}, p = {p}, dp = {dp}, {current_n}x{current_n} mesh")
                    print("="*40)

                # 1. Primary variable u
                error = u_sol - u_exact
                grad_u_exact = ng.CF((u_exact.Diff(x), u_exact.Diff(y)))
                grad_error = grad(u_sol) - grad_u_exact
                
                H1_error = sqrt(ng.Integrate(error * error + grad_error * grad_error, mesh, order=25))
                H1_exact = sqrt(ng.Integrate(u_exact * u_exact + grad_u_exact * grad_u_exact, mesh, order=25))

                rel_H1_error = H1_error/H1_exact if H1_exact > 0 else 0.0
                if printResults:
                    print(f"\n1. PRIMARY VARIABLE u:")
                    print(f"   ||u - u_exact||_H1 = {H1_error:.4e}")
                    print(f"   Relative error:     {rel_H1_error:.4e}")

                # 2. Error estimator
                n_e = sqrt(ng.Integrate(e_sol * e_sol + grad(e_sol) * grad(e_sol), mesh, order = 25))
                if printResults:
                    print(f"\n2. ERROR ESTIMATOR:")
                    print(f"   ||e||_Y = {n_e:.4e}")
                    print(f"   Effectivity index: {n_e/H1_error:.4f}")

                # 3. Flux interface variable q \cdot n
                grad_u_exact = ng.CF((u_exact.Diff(x), u_exact.Diff(y)))
                exact_flux = -ng.InnerProduct(grad_u_exact, n)
                num_flux = ng.InnerProduct(q_sol, n)

                err_func_flux = (num_flux - exact_flux)**2
                flux_error = sqrt(abs(ng.Integrate(err_func_flux * dx(element_boundary=True, bonus_intorder=20), mesh)))

                exact_func_flux = exact_flux ** 2
                flux_exact = sqrt(abs(ng.Integrate(exact_func_flux * dx(element_boundary=True, bonus_intorder=20), mesh)))

                rel_flux_error = flux_error / flux_exact if flux_exact > 0 else 0.0
                if printResults:
                    print(f"\n3. INTERFACE FLUX q:")
                    print(f"   ||q·n - grad(u_exact)·n||_L2(E_h) = {flux_error:.4e}")
                    print(f"   Relative flux error:              {rel_flux_error:.4e}")

                # 4. Monitor residual to detect conditioning issues
                # Compute ||Au_h - f|| / ||f||
                final_resid = f.vec.CreateVector()
                A.Apply(gfu.vec, final_resid)
                final_resid.data -= f.vec

                # Only count free DOFs
                resid_norm_sq = 0.0
                f_norm_sq = 0.0
                for i in range(XY.ndof):
                    if XY.FreeDofs()[i]:
                        resid_norm_sq += abs(final_resid[i])**2
                        f_norm_sq += abs(f.vec[i])**2

                rel_resid = sqrt(resid_norm_sq / max(f_norm_sq, 1e-20))

                if printResults:
                    print(f"\n4. NUMERICAL RESIDUAL (conditioning):")
                    print(f"   ||A·u - f||          = {sqrt(resid_norm_sq):.4e}")
                    print(f"   ||A·u - f|| / ||f||  = {rel_resid:.4e}")

                # Store results
                stored_results[(p, dp)]['N'].append(current_n)
                stored_results[(p, dp)]['estim'].append(n_e)
                stored_results[(p, dp)]['resid'].append(rel_resid)
                stored_results[(p, dp)]['err'].append(H1_error)
                stored_results[(p, dp)]['flux_err'].append(rel_flux_error)

            # ------------------------------------------------------------------------------
            # Netgen GUI Visualization (only for the finest mesh)
            # ------------------------------------------------------------------------------
            if showPlots and current_n == N_values[-1]:
                print(f"\n[GUI] Visualizing results for case {case}, p={p}, dp={dp}, N={current_n}...")
                import netgen.gui
                
                # Numerical solution
                ng.Draw(ng.Norm(u_sol), mesh, "u_num_magnitude")
                ng.Draw(u_sol.real, mesh, "u_num_real")
                ng.Draw(u_sol.imag, mesh, "u_num_imag")

                # Exact solution
                ng.Draw(ng.Norm(u_exact), mesh, "u_exact_magnitude")
                ng.Draw(u_exact.real, mesh, "u_exact_real")
                ng.Draw(u_exact.imag, mesh, "u_exact_imag")

                # Exact error
                ng.Draw(ng.Norm(error), mesh, "error_magnitude")
                ng.Draw(error.real, mesh, "error_real")
                ng.Draw(error.imag, mesh, "error_imag")

                # Error representation component e
                ng.Draw(ng.Norm(e_sol), mesh, "error_estimator")
                
                input(">> Check Netgen GUI. Dont't close the GUI! Press <Enter> to continue to next configuration...")

    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------
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

    plt.figure(figsize=(6, 4.5), constrained_layout = True)
    N_end = stored_results[(2, 1)]['N'][-1] + 10

    for p, dp, color, marker, label in run_configs:
        data = stored_results[(p, dp)]

        N_arr = data['N']
        Err_arr = data['err']
        Estim_arr = data['estim']
 
        # 1. Plot L2-Error (solid line)
        plt.loglog(N_arr, Err_arr, f'-{marker}', color=color, 
                    lw=2, markersize=6, label=label)
    
        # 2. Plot Error Estimator (dashed line, same color, same marker)
        plt.loglog(N_arr, Estim_arr, f'--{marker}', color=color, 
                    lw=1.5, markersize=5, alpha=0.8)
        
        # 3. Plot Reference Slopes (only once)
        err_start = Err_arr[0]
        N_start = N_arr[0]
        if p == 1 and dp == 4:
            y_ref2 = [err_start, err_start * (N_end / N_start)**(-2.0)]
            plt.loglog([N_start, N_end], y_ref2, '--', color='gray', lw=1, alpha=0.7)
            plt.annotate(r'$h^2$', xy=(N_end, y_ref2[1]), color='gray', ha='left', va='center')

        if p == 2 and dp == 4:
            y_ref3 = [err_start, err_start * (N_end / N_start)**(-3.0)]
            plt.loglog([N_start, N_end], y_ref3, '--', color='gray', lw=1, alpha=0.7)
            plt.annotate(r'$h^3$', xy=(N_end, y_ref3[1]),  color='gray', ha='left', va='center')

    # Styling
    plt.xlabel(r"Mesh Resolution $N = 1/h$")
    plt.ylabel(r"$H^1$-Error / Error Estimator")
    leg = plt.legend(
    loc='upper right',
    title=r"Solid: $H^1$-error" "\n" r"Dashed: Error Estimator"
    )
    leg._legend_box.align = "left"
    plt.grid(True, which="major", linestyle='--', alpha=0.4)

    # Save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"H1_convergence_case_{case}_maxN_{N_values[-1]}.pdf"
    full_path_L2 = os.path.join(script_dir, filename)
    plt.savefig(full_path_L2, bbox_inches='tight')

    print("\n" + "="*80)
    print(f"Saved H1-error plot to: {full_path_L2}")
    #plt.show()

    # ------------------------------------------------------------------------------
    # Condition Number Plot (separate figure)
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(6, 4.5), constrained_layout = True)

    for p, dp, color, marker, label in run_configs:
        data = stored_results[(p, dp)]

        plt.loglog(data['N'], data['resid'], f'-{marker}', color=color, lw=2, 
                markersize=6, label=label)

    plt.xlabel(r"Mesh Eesolution $N = 1/h$")
    plt.ylabel(r"Relative Residual $\|Au_h - f\| \ / \ \|f\|$")
    plt.legend(loc='best')
    plt.grid(True, which="major", linestyle='--', alpha=0.4)

    filename_resid = f"Relative_residual_case_{case}_maxN_{N_values[-1]}.pdf"
    full_path_resid = os.path.join(script_dir, filename_resid)
    plt.savefig(full_path_resid, bbox_inches='tight')
    print("="*80)
    print(f"Saved residual plot to: {full_path_resid}")
    #plt.show()

    # ------------------------------------------------------------------------------
    # Flux Error Plot (separate figure)
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(6, 4.5), constrained_layout=True)

    for p, dp, color, marker, label in run_configs:
        data = stored_results[(p, dp)]
        if 'flux_err' in data and len(data['flux_err']) > 0:
            plt.loglog(data['N'], data['flux_err'], f'-{marker}', color=color, lw=1.5, 
                       markersize=6, alpha=0.8, label=label)

    plt.xlabel(r"Mesh Resolution $N = 1/h$")
    plt.ylabel(r"Relative Flux Error")
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.07))
    plt.grid(True, which="major", linestyle='--', alpha=0.4)

    plt.annotate(r"Dashed: $\|q\cdot n - \nabla u_{\mathrm{exact}}\cdot n\|_{L^2(\partial \Omega_h)} / \| \nabla u_{\mathrm{exact}}\cdot n \|_{L^2(\partial \Omega_h)}$", 
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=11, ha='left', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    filename_interface = f"Relative_Flux_error_case_{case}_maxN_{N_values[-1]}.pdf"
    full_path_interface = os.path.join(script_dir, filename_interface)
    plt.savefig(full_path_interface, bbox_inches='tight')
    print("="*80)
    print(f"Saved interface error plot to: {full_path_interface}")