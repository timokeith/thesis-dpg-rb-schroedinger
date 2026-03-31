import os
import ngsolve as ng
ng.ngsglobals.msg_level = 0
from ngsolve import dx, x, y, grad, exp, IfPos, sqrt, pi, sin
from ngsolve.meshes import MakeStructured2DMesh
import matplotlib.pyplot as plt

# List of N values
N_values = [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Run Configurations: (p, Delta_p, color, marker, label)
run_configs = [
    (3, 1, 'red', 'o', r'$p=3, \ \Delta p=1$'),
    (3, 2, 'blue', '^', r'$p=3, \ \Delta p=2$'),
    (4, 1, 'green', 'D', r'$p=4, \ \Delta p=1$'),
    (4, 2, 'black', 's', r'$p=4, \ \Delta p=2$')
]

plot_solutions = False  # turn on to plot highest resolutioon for each configuration before continuing the loop
qprime_order = 'reduced' # set in order to choose between paper results and full qprime order; options: 'full' or 'reduced'

for case in [1,2]:  # two cases in thesis
    stored_results = {}
    for p, dp, color, marker, label in run_configs:     
        stored_results[(p, dp)] = {'dofs': [], 'err': [], 'estim': [], 'resid': [], 'skel_err': [], 'flux_err': []}
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
            mesh = MakeStructured2DMesh(nx=current_n, ny=current_n)

            U = ng.L2(mesh, order=p-1, complex=True)
            Q_plus = ng.H1(mesh, order=p, orderinner=0, complex=True, dirichlet="bottom|left|right")
            
            if qprime_order == 'full':
                Q_prime = ng.FacetFESpace(mesh, order=p, complex=True)  
            elif qprime_order == 'reduced':
                Q_prime = ng.FacetFESpace(mesh, order=p-1, complex=True) 

            Y = ng.L2(mesh, order= p + dp, complex=True)

            # Product Space: u, q+, q', e
            X = U * Q_plus * Q_prime
            XY = X * Y

            # Trial: u, qplus, qprime, e
            u, qplus, qprime, e = XY.TrialFunction()
            # Test: w, qplus_test, qprime_test, v
            w, qplus_test, qprime_test, v = XY.TestFunction()

            # ------------------------------------------------------------------------------
            # 2. Exact Solution RHS definition
            # ------------------------------------------------------------------------------  
            beta = 2.5
            if case == 1:
                # Complex Gaussian (case (a) from Demkowicz et al. 2017)
                M_val, T0 = 1.5, 1.5
        
                D_term = T0**2 - 1j * beta * y
                u_exact = (M_val * T0 / sqrt(D_term)) * exp(-(x**2) / D_term)

            elif case == 2:
                # Wave packet (case (b) from Demkowicz et al. 2017)
                omega = 20.0  # change for alternative frequencies
                a0 = (2.0 / omega**2)**0.25
                
                # additionally added the phase term: exp(i * omega * r**2), 
                r_sq = x**2 + y**2
                u_exact = a0 * exp(-r_sq/omega**2) * exp(1j * omega * r_sq)     # this was changed to a0 * exp(-r_sq) * exp(1j * omega * r_sq) for the alternative envelope

            # Manufactured RHS via symbolic differentiation
            rhs_func = 1j * u_exact.Diff(y) - (beta/2) * u_exact.Diff(x).Diff(x) 

            # ------------------------------------------------------------------------------
            # 3. Forms and Operators
            # ------------------------------------------------------------------------------
            n = ng.specialcf.normal(mesh.dim)
            n_x, n_t = n[0], n[1]

            # Define edge type indicators
            is_vertical_edge = IfPos(n_x**2 - n_t**2, 1.0, 0.0)    # True when |n_x| > |n_t|
            is_horizontal_edge = IfPos(n_t**2 - n_x**2, 1.0, 0.0)  # True when |n_t| > |n_x|

            # Operators
            def d_t(func): return grad(func)[1]
            def d_x(func): return grad(func)[0]
            def d_xx(func): return func.Operator("hesse")[0,0]

            # The Adjoint Operator A* = i dt - (beta/2) dxx
            def Adj_A(func):
                return 1j * d_t(func) - (beta/2) * d_xx(func)

            # ------------------------------------------------------------------------------
            # 4. Assembly
            # ------------------------------------------------------------------------------
            # We assemble the full matrix A = [ R_Y  B ]
            #                                 [ B^H  0 ]
            print("Assembling system...")
            A = ng.BilinearForm(XY, hermitian=True, condense=True)

            # --- Block r_Y: (e, v)_Y  ---
            # Graph Norm Inner Product: (e, v) + (A* e, A* v)
            A += e * ng.Conj(v) * dx 
            A += Adj_A(e) * ng.Conj(Adj_A(v)) * dx

            # --- Block B: Physics b((u,q), v) ---
            # Corresponds to: (u, A* v) + <q, v>
            A += u * ng.Conj(Adj_A(v)) * dx

            # Boundary terms 
            # q+ term (time part + space part)
            term_B_1 = is_horizontal_edge * 1j * n_t * qplus * ng.Conj(v)
            term_B_2 = is_vertical_edge * (beta/2) * n_x * qplus * ng.Conj(d_x(v))
            # q' term (flux part)
            term_B_3 = is_vertical_edge * (beta/2) * n_x * qprime * ng.Conj(v)
            A += (term_B_1 + term_B_2 - term_B_3) * dx(element_boundary=True)

            # --- Block B^H: Conjugate transpose b((w, ...), e)^H ---
            A += Adj_A(e) * ng.Conj(w) * dx

            # Term 1 Transpose: Conjugate of i is -i
            term_B_1_H = is_horizontal_edge * (-1j) * n_t * ng.Conj(qplus_test) * e
            term_B_2_H = is_vertical_edge * (beta/2) * n_x * ng.Conj(qplus_test) * d_x(e) 

            term_B_3_H = is_vertical_edge * (beta/2) * n_x * ng.Conj(qprime_test) * e
            A += (term_B_1_H + term_B_2_H - term_B_3_H) * dx(element_boundary=True)

            bonus_intorder_f = 20
            f = ng.LinearForm(XY)
            f += rhs_func * ng.Conj(v) * dx(bonus_intorder=bonus_intorder_f)
            
            # ------------------------------------------------------------------------------
            # 5. Solve with Non-Homogeneous BCs
            # ------------------------------------------------------------------------------
            with ng.TaskManager():
                A.Assemble()

                # --- Exclude horizontal-edge Q_prime DOFs from coupling freedofs ---
                UDOFs, QplusDOFs, QprimeDOFs, YDOFs = U.ndof, Q_plus.ndof, Q_prime.ndof, Y.ndof
                qprime_offset = UDOFs + QplusDOFs   # Q_prime DOFs start here in XY
                
                horiz_qprime_dofs_XY = set()
                for edge in mesh.edges:
                    # Get edge vertex coordinates
                    verts = edge.vertices
                    pts = [mesh[v].point for v in verts]
                    if abs(pts[0][1] - pts[1][1]) < 1e-10: # horizontal edge
                        dofs = Q_prime.GetDofNrs(ng.NodeId(ng.EDGE, edge.nr))
                        for d in dofs:
                            if d >= 0: 
                                horiz_qprime_dofs_XY.add(qprime_offset + d)

                number_of_horiz_Qprime_DOFs = len(horiz_qprime_dofs_XY)
                net_QprimeDOFs = QprimeDOFs - number_of_horiz_Qprime_DOFs
                net_XDOFs = UDOFs + QplusDOFs + net_QprimeDOFs

                f.Assemble()   
                gfu = ng.GridFunction(XY)

                print(f"\nAssembling done.")
                print("-"*40)

                print("Solving with static condensation...")
                # Set Dirichlet BC: q+ = u_exact on bottom, left, right
                gfu.components[1].Set(u_exact, ng.BND)

                # Compute residual from Dirichlet boundary lift
                r = f.vec.CreateVector()
                A.Apply(gfu.vec, r)

                rhs = f.vec.CreateVector()
                rhs.data = f.vec - r

                # Static Condensation
                res = rhs.CreateVector()
                res.data = rhs
                res.data += A.harmonic_extension_trans * rhs

                coupling_freedofs = ng.BitArray(XY.ndof)
                coupling_freedofs[:] = XY.FreeDofs(coupling=True)
                for d in horiz_qprime_dofs_XY:
                    coupling_freedofs.Clear(d)

                SchurInv = A.mat.Inverse(freedofs=coupling_freedofs, inverse='pardiso')

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
                u_sol = gfu.components[0]
                qplus_sol = gfu.components[1]
                qprime_sol = gfu.components[2]
                e_sol = gfu.components[3]

                print("="*40)
                print(f"RESULTS: Case {case}, p = {p}, dp = {dp}, {current_n}x{current_n} mesh")
                print("="*40)

                print(f"U DOFs: {UDOFs}")
                print(f"Q_plus DOFs: {QplusDOFs}")
                print(f"Q_prime DOFs: {QprimeDOFs}")
                print(f"Net Q_prime DOFs: {net_QprimeDOFs}")
                print(f"Net X DOFs: {net_XDOFs}")
                print(f"Y DOFs: {YDOFs}")

                # 1. Primary variable u
                error = u_sol - u_exact
                L2_error = sqrt(ng.Integrate(error * ng.Conj(error), mesh, order=40).real)
                L2_exact = sqrt(ng.Integrate(u_exact * ng.Conj(u_exact), mesh, order=40).real)
                rel_L2_error = L2_error/L2_exact if L2_exact > 0 else 0.0
                print(f"\n1. PRIMARY VARIABLE u:")
                print(f"   ||u - u_exact||_L2 = {L2_error:.4e}")
                print(f"   Relative error:     {rel_L2_error:.4e}")

                # 2. Error estimator
                n_e = sqrt((ng.Integrate(e_sol * ng.Conj(e_sol), mesh, order = 40).real + 
                            ng.Integrate(Adj_A(e_sol) * ng.Conj(Adj_A(e_sol)), mesh, order = 40).real))
                print(f"\n2. ERROR ESTIMATOR:")
                print(f"   ||e||_Y = {n_e:.4e}")
                print(f"   Effectivity index: {n_e/L2_error:.4f}")

                # 3. Skeleton trace q+ (all element boundaries)
                err_func_skel = (qplus_sol - u_exact) * ng.Conj(qplus_sol - u_exact)
                skel_error = sqrt(abs(ng.Integrate(err_func_skel * dx(element_boundary=True, bonus_intorder=40), mesh)))

                exact_func_skel = u_exact * ng.Conj(u_exact)
                skel_exact = sqrt(abs(ng.Integrate(exact_func_skel * dx(element_boundary=True, bonus_intorder=40), mesh)))

                rel_skel_error = skel_error/skel_exact if skel_exact > 0 else 0.0
                print(f"\n3. SKELETON TRACE q+:")
                print(f"   ||q+ - u_exact||_L2(E_h^+) = {skel_error:.4e}")
                print(f"   Relative skeleton error:  {rel_skel_error:.4e}")

                # 4. Flux q' on vertical edges
                u_exact_dx = u_exact.Diff(x)

                err_func_flux = is_vertical_edge * (qprime_sol - u_exact_dx) * ng.Conj(qprime_sol - u_exact_dx)
                flux_error = sqrt(abs(ng.Integrate(err_func_flux * dx(element_boundary=True, bonus_intorder=40), mesh)))

                exact_func_flux = is_vertical_edge * u_exact_dx * ng.Conj(u_exact_dx)
                flux_exact = sqrt(abs(ng.Integrate(exact_func_flux * dx(element_boundary=True, bonus_intorder=40), mesh)))

                rel_flux_error = flux_error/flux_exact if flux_exact > 0 else 0.0
                print(f"\n4. FLUX q' (vertical edges):")
                print(f"   ||q' - partial_x u_exact||_L2(E_h') = {flux_error:.4e}")
                print(f"   Relative flux error: {rel_flux_error:.4e}")

                # 5. Monitor residual to detect conditioning issues
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

                print(f"\n5. NUMERICAL RESIDUAL (conditioning):")
                print(f"   ||A·u - f||          = {sqrt(resid_norm_sq):.4e}")
                print(f"   ||A·u - f|| / ||f||  = {rel_resid:.4e}")

                # Store results
                stored_results[(p, dp)]['dofs'].append(net_XDOFs)
                stored_results[(p, dp)]['estim'].append(n_e)
                stored_results[(p, dp)]['resid'].append(rel_resid)
                stored_results[(p, dp)]['err'].append(L2_error)
                stored_results[(p, dp)]['skel_err'].append(rel_skel_error)
                stored_results[(p, dp)]['flux_err'].append(rel_flux_error)

            # ------------------------------------------------------------------------------
            # Netgen GUI Visualization (only for the finest mesh) 
            # ------------------------------------------------------------------------------
            if plot_solutions and current_n == N_values[-1]:
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

                # Pause execution to inspect the GUI
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
    dofs_end = stored_results[(4, 1)]['dofs'][-1] + 100

    for p, dp, color, marker, label in run_configs:
        data = stored_results[(p, dp)]

        dofs_arr = data['dofs']
        Err_arr = data['err']
        Estim_arr = data['estim']
 
        # 1. Plot L2-Error (solid line)
        plt.loglog(dofs_arr, Err_arr, f'-{marker}', color=color, 
                    lw=2, markersize=6, label=label)
    
        # 2. Plot Error Estimator (dashed line, same color, same marker)
        plt.loglog(dofs_arr, Estim_arr, f'--{marker}', color=color, 
                    lw=1.5, markersize=5, alpha=0.8)
        
        # 3. Plot Reference Slopes (only once)
        err_start = Err_arr[0]
        dofs_start = dofs_arr[0]
        if p == 3 and dp == 1:
            y_ref2 = [err_start, err_start * (dofs_end / dofs_start)**(-1.0)]
            plt.loglog([dofs_start, dofs_end], y_ref2, '--', color='gray', lw=1, alpha=0.7)
            plt.annotate(r'$h^2$', xy=(dofs_end, y_ref2[1]), color='gray', ha='left', va='center')

            y_ref3 = [err_start, err_start * (dofs_end / dofs_start)**(-1.5)]
            plt.loglog([dofs_start, dofs_end], y_ref3, '--', color='gray', lw=1, alpha=0.7)
            plt.annotate(r'$h^3$', xy=(dofs_end, y_ref3[1]), color='gray', ha='left', va='center')

        if p == 4 and dp == 2:
            y_ref3 = [err_start, err_start * (dofs_end / dofs_start)**(-1.5)]
            plt.loglog([dofs_start, dofs_end], y_ref3, '--', color='gray', lw=1, alpha=0.7)
            plt.annotate(r'$h^3$', xy=(dofs_end, y_ref3[1]),  color='gray', ha='left', va='center')
        
            y_ref4 = [err_start, err_start * (dofs_end / dofs_start)**(-2.0)]
            plt.loglog([dofs_start, dofs_end], y_ref4, '--', color='gray', lw=1, alpha=0.7)
            plt.annotate(r'$h^4$', xy=(dofs_end, y_ref4[1]), color='gray', ha='left', va='center')

    # Styling
    plt.xlabel(r"Number of Trial Space DOFs $\sim 1/{h^2}$")
    plt.ylabel(r"$L^2$-Error / Error Estimator")
    leg = plt.legend(
    loc='upper right',
    title=r"Solid: $L^2$-error" "\n" r"Dashed: error estimator"
    )
    leg._legend_box.align = "left"
    plt.grid(True, which="major", linestyle='--', alpha=0.4)

    # Save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"L2_convergence_case_{case}_maxN_{N_values[-1]}_qprime_order_{qprime_order}.pdf"
    full_path_L2 = os.path.join(script_dir, filename)
    plt.savefig(full_path_L2, bbox_inches='tight')

    print("\n" + "="*80)
    print(f"Saved L2-error plot to: {full_path_L2}")
    #plt.show()

    # ------------------------------------------------------------------------------
    # Condition Number Plot (separate figure)
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(6, 4.5), constrained_layout = True)

    for p, dp, color, marker, label in run_configs:
        data = stored_results[(p, dp)]

        plt.loglog(data['dofs'], data['resid'], f'-{marker}', color=color, lw=2, 
                markersize=6, label=label)

    plt.xlabel(r"Number of Trial Space DOFs $\sim 1/{h^2}$")
    plt.ylabel(r"Relative Residual $\|Au_h - f\| \ / \ \|f\|$")
    plt.legend(loc='best')
    plt.grid(True, which="major", linestyle='--', alpha=0.4)

    filename_resid = f"Relative_residual_case_{case}_maxN_{N_values[-1]}_qprime_order_{qprime_order}.pdf"
    full_path_resid = os.path.join(script_dir, filename_resid)
    plt.savefig(full_path_resid, bbox_inches='tight')
    print("="*80)
    print(f"Saved residual plot to: {full_path_resid}")
    #plt.show()

    # ------------------------------------------------------------------------------
    # Skeleton (q+) and Flux (q') Error Plot
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(6, 4.5), constrained_layout=True)

    for p, dp, color, marker, label in run_configs:
        data = stored_results[(p, dp)]
        
        # Solid lines for q+ (skeleton) error
        if data['skel_err']:
            plt.loglog(data['dofs'], data['skel_err'], f'-{marker}', color=color, lw=2, 
                    markersize=6, label=label)
        
        # Dashed lines for q' (flux) error (same color, same marker)
        if data['flux_err']:
            plt.loglog(data['dofs'], data['flux_err'], f'--{marker}', color=color, lw=1.5, 
                    markersize=5, alpha=0.8)

    plt.xlabel(r"Number of Trial Space DOFs $\sim 1/{h^2}$")
    plt.ylabel(r"Relative Interface Variable Errors")
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.13))
   

    plt.grid(True, which="major", linestyle='--', alpha=0.4)

    # Add annotation explaining line styles
    plt.annotate(r"Solid: $\|q^+ - u_{\mathrm{exact}}\|_{L^2(E_h^+)} / \| u_{\mathrm{exact}} \|_{L^2(E_h^+)}$" + "\n" + r"Dashed: $\|q' - \partial_x u_{\mathrm{exact}}\|_{L^2(E_h')} / \| \partial_x u_{\mathrm{exact}} \|_{L^2(E_h')}$", 
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize = 11, ha='left', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    filename_interface = f"Relative_Interface_errors_case_{case}_maxN_{N_values[-1]}_qprime_order_{qprime_order}.pdf"
    full_path_interface = os.path.join(script_dir, filename_interface)
    plt.savefig(full_path_interface, bbox_inches='tight')
    print("="*80)
    print(f"Saved interface error plot to: {full_path_interface}")
    print("="*80+"\n")
    #plt.show()