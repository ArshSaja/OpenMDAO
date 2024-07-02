"""Define the NewtonSolver class."""


import numpy as np

from scipy.sparse import csc_matrix 
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI


class PreconditionedInexactNewton(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    _linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = 'NL: PINewton'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None
        
        # IN cache counter
        self._cache_residual_counter = 0
        self._cache_output_counter = 0

        # resdiual subspace projector
        self._p_residual = None
        # states subspace projector
        self._p_states = None

        # Slot for linesearch
        self.supports['linesearch'] = True
        self._preconditioned = False
        self._linesearch = BoundsEnforceLS()
        self._linesearchPrecond = BoundsEnforceLS()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('solve_subsystems', types=bool,
                             desc='Set to True to turn on sub-solvers (Hybrid Newton).')
        self.options.declare('max_sub_solves', types=int, default=50,
                             desc='Maximum number of subsystem solves.')
        self.options.declare('number_of_inexact_newton_steps', types=int, default=5,
                             desc='Maximum number of subsystem solves.')
        self.options.declare('precond_rtol', default=1e-2,
                             desc='relative error tolerance for the preconditioned stage')
        self.options.declare('cs_reconverge', types=bool, default=True,
                             desc='When True, when this driver solves under a complex step, nudge '
                             'the Solution vector by a small amount so that it reconverges.')
        self.options.declare('reraise_child_analysiserror', types=bool, default=False,
                             desc='When the option is true, a solver will reraise any '
                             'AnalysisError that arises during subsolve; when false, it will '
                             'continue solving.')

        self.supports['gradients'] = True
        self.supports['implicit_components'] = True

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)

        self._disallow_discrete_outputs()

        if not isinstance(self.options._dict['solve_subsystems']['val'], bool):
            msg = '{}: solve_subsystems must be set by the user.'
            raise ValueError(msg.format(self.msginfo))

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver
        self._linesearchPrecond._setup_solvers(system, self._depth + 1)
        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.linear_solver is not None:
            for s in self.linear_solver._assembled_jac_solver_iter():
                yield s

    def _set_solver_print(self, level=2, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        super()._set_solver_print(level=level, type_=type_)

        if self.linear_solver is not None and type_ != 'NL':
            self.linear_solver._set_solver_print(level=level, type_=type_)
        self._linesearchPrecond._set_solver_print(level=level, type_=type_)
        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_)

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))

        system = self._system()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._apply_nonlinear()
        finally:
            self._recording_iter.pop()

            # Enable local fd
            system._owns_approx_jac = approx_status

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        bool
            Flag for indicating child linerization
        """
        return (self.options['solve_subsystems'] and not system.under_complex_step
                and self._iter_count <= self.options['max_sub_solves'])

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.linear_solver is not None:
            self.linear_solver._linearize()
        self._linesearchPrecond._linearize()
        if self.linesearch is not None:
            self.linesearch._linearize()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        self._cache_residual_counter = 0
        self._cache_output_counter = 0

        self._preconditioned = False
        
        system = self._system()
        solve_subsystems = self.options['solve_subsystems'] and not system.under_complex_step

        n_rows = len(system._residuals.asarray(copy=True))
        n_cols = self.options['number_of_inexact_newton_steps']
        self._cache_residuals = np.asarray(np.zeros((n_rows, n_cols), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype))
        # IN cache outputs
        self._cache_states =  np.asarray(np.zeros((n_rows, n_cols), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype))


        if self.options['debug_print']:
            self._err_cache['inputs'] = system._inputs._copy_views()
            self._err_cache['outputs'] = system._outputs._copy_views()

        # Execute guess_nonlinear if specified and
        # we have not restarted from a saved point
        if not self._restarted and system._has_guess:
            system._guess_nonlinear()

        with Recording('Newton_subsolve', 0, self) as rec:

            if solve_subsystems and self._iter_count <= self.options['max_sub_solves']:

                self._solver_info.append_solver()

                # should call the subsystems solve before computing the first residual
                self._gs_iter()

                self._solver_info.pop()

            self._run_apply()
            norm = self._iter_get_norm()

            rec.abs = norm
            norm0 = norm if norm != 0.0 else 1.0
            rec.rel = norm / norm0

        return norm0, norm

    def _inexact_newton_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        self._solver_info.append_subsolver()
        do_subsolve = self.options['solve_subsystems'] and not system.under_complex_step and \
            (self._iter_count < self.options['max_sub_solves'])
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._dresiduals.set_vec(system._residuals)
            system._dresiduals *= -1.0
            my_asm_jac = self.linear_solver._assembled_jac

            system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
            if (my_asm_jac is not None and
                    system.linear_solver._assembled_jac is not my_asm_jac):
                my_asm_jac._update(system)

            self._linearize()

            self.linear_solver.solve('fwd')

            if self.linesearch and not system.under_complex_step:
                self.linesearch._do_subsolve = do_subsolve
                self.linesearch.solve()
            else:
                system._outputs += system._doutputs

            self._solver_info.pop()

            # Hybrid newton support.
            if do_subsolve:
                with Recording('Newton_subsolve', 0, self):
                    self._solver_info.append_solver()
                    self._gs_iter()
                    self._solver_info.pop()
        finally:
            # Enable local fd
            system._owns_approx_jac = approx_status

            if self._cache_residual_counter==0:
                self._run_apply()
                norm = self._iter_get_norm()
                self.abs0 = norm

    def _train_for_PIN(self):
        """
        This function train the nonlinear system for getting the reduce subspace
        """
        atol = self.options['atol']
        rtol = self.options['rtol']

        # IN cache residual
        system = self._system()

        ### Step 1 ###

        # run inexact newton steps for training
        # iter = 0
        if self._iter_count<self.options['number_of_inexact_newton_steps']:

            if system.comm.rank==0 and self._iter_count==0:
                print("Starting training PINewton")

            self._inexact_newton_iteration()

            # compute the residuals
            self._run_apply()

            # cahce the resd for training
            self.cache_residuals_pin()
            # cahce the outp for training
            self.cache_states_pin()
            
            # compute the norm and see if the system has already converged
            norm = self._iter_get_norm()

            abs_IN = norm
            rel_IN = norm / self.abs0 if self.abs0 != 0.0 else 1.0

            if abs_IN < atol or rel_IN < rtol:
                return 0
            
            # iter = iter + 1
            if self._iter_count==self.options['number_of_inexact_newton_steps']-1:
                self._trained_PIN = True
                if system.comm.rank==0:
                    print("Ending training PINewton")

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """

        rtol = self.options['precond_rtol']

        ### Step 1 ###
        
        # First, we run the solver sequentially
        # Call the training function
        if not self._trained_PIN:
            self._train_for_PIN()
            return None
        
        # self._cache_residuals = np.ndarray(self._cache_residuals)
        # self._cache_states = np.ndarray(self._cache_states)

        system = self._system()
        # self._solver_info.append_subsolver()
        do_subsolve = self.options['solve_subsystems'] and not system.under_complex_step and \
            (self._iter_count < self.options['max_sub_solves'])
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._dresiduals.set_vec(system._residuals)
            system._dresiduals *= -1.0
            my_asm_jac = self.linear_solver._assembled_jac

            system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
            if (my_asm_jac is not None and
                    system.linear_solver._assembled_jac is not my_asm_jac):
                my_asm_jac._update(system)

            self._linearize()
        
            # PIN starts

            # compute P and Q projectors
            self.minimize_variance()


            ### Step 2 ###

            # compute mean residual from the IN
            mean_residuals = self.compute_means(self._cache_residuals)

            # P^TP
            PTP= np.matmul(self._p_residual, self._p_residual.transpose())

            # resdiual subspace projector f(Y_j) = PP^T(F(Y_j)-F_mean) + F_mean
            init_residual = self._cache_residuals[:,-1]
            init_states = system._outputs
            projected_approx_residual_init = PTP.dot(init_residual-mean_residuals) + mean_residuals

            if not self._preconditioned:
                self.projected_approx_residual = PTP.dot(init_residual-mean_residuals) + mean_residuals
                self._preconditioned = True

            # converge the reduce nonlinear space
            iter=0
            while np.linalg.norm(self.projected_approx_residual) > rtol*np.linalg.norm(projected_approx_residual_init) and iter < self.options['maxiter']:
                # resdiual subspace projector f(Y_j) = PP^T(F(Y_j)-F_mean) + F_mean
                # projected_approx_residual = PTP.dot(init_residual-mean_residuals) + mean_residuals
                
                if iter==0 and system.comm.rank==0:
                    print("Starting PInewton preconditioning")
                # dimension reduced vector
                dimension_reduced_residual = self._p_residual.transpose().dot(self.projected_approx_residual)

                # compute the projected jacobian
                # matrix = self.linear_solver._assembled_jac
                my_asm_jac = system.linear_solver._assembled_jac

                system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
                if (my_asm_jac is not None and
                        system.linear_solver._assembled_jac is not my_asm_jac):
                    my_asm_jac._update(system)

                self._linearize()
                my_asm_jac = system._linear_solver._assembled_jac._int_mtx._matrix
                if my_asm_jac is not None:
                    if isinstance(my_asm_jac, csc_matrix):
                        jac_lup = my_asm_jac.todense()
                    elif isinstance(my_asm_jac, np.ndarray):
                        jac_lup = my_asm_jac
                else:
                    jac_lup = system.linear_solver._build_mtx()
            
                # P^T x Jac
                p_residual_csc = np.matmul(self._p_residual.transpose(),jac_lup)
                # P_tranpose_Jac =p_residual_csc.multiply(my_asm_jac)
                # Jp = P^T x Jac x Q
                Jac_p = np.matmul( p_residual_csc, self._p_states)

 
                Sub_sol = np.linalg.solve(Jac_p, -dimension_reduced_residual)
                # system._dresiduals.set_vec(np.asarray(dimension_reduced_residual))
               
                # print(Sub_sol)
                # Y(i+1) = Y(i) + QSp
                if self.linesearch and not system.under_complex_step:
                    system._doutputs.set_val(0)
                    system._doutputs += self._p_states.dot(Sub_sol)
                    self._linesearchPrecond._do_subsolve = do_subsolve
                    # BoundsEnforceLS.solve()
                    # self.linesearch._enforce_bounds(step=system._doutputs, alpha=1.0)
                    self._linesearchPrecond.solve()
                else:
                    system._outputs += self._p_states.dot(Sub_sol)
                 
                # self._solver_info.pop()
                # system._outputs.set_vec(init_states)
                # Hybrid newton support.
                if do_subsolve:
                    with Recording('Newton_subsolve', 0, self):
                        # self._solver_info.append_solver()
                        self._gs_iter()
                        # self._solver_info.pop()
                
                self._run_apply()

                init_residual = system._residuals.asarray()
                # system._dresiduals.set_vec(system._residuals)
                self.projected_approx_residual = PTP.dot(init_residual-mean_residuals) + mean_residuals

                iter = iter + 1
                if np.linalg.norm(self.projected_approx_residual) <= rtol*np.linalg.norm(projected_approx_residual_init) and iter < self.options['maxiter']:
                    if system.comm.rank==0:
                        print("PInewton preconditioned converged")
                        self._preconditioned=True
                elif np.linalg.norm(self.projected_approx_residual) > rtol*np.linalg.norm(projected_approx_residual_init) and iter >= self.options['maxiter']:
                    if system.comm.rank==0:
                        print("PInewton preconditioned did not converged")
                        msg = (f"Solver '{self.SOLVER}' preconditioned on system '{system.pathname}' stalled after "
                    f"{iter} iterations.")
                        self.report_failure(msg)
            
            # run the inexact newton
            self._inexact_newton_iteration()
            
        finally:
            # Enable local fd
            system._owns_approx_jac = approx_status
        
        

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self.linear_solver is not None:
            self.linear_solver._set_complex_step_mode(active)
            if self.linear_solver._assembled_jac is not None:
                self.linear_solver._assembled_jac.set_complex_step_mode(active)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super().cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()

    def use_relevance(self):
        """
        Return True if relevance should be active.

        Returns
        -------
        bool
            True if relevance should be active.
        """
        return False


## helper functions ##
# * First cahce the nonlinear reduced space for modified nonlinear system
# * minimize the variance
# * Project the nonlinear space
# * Do the same for the nonlinear solutions
# * The training step
# *   

    def cache_residuals_pin(self):
        """
        This function cache the nonlinear residuals from IN
        """
        system = self._system()
        
        self._cache_residuals[:,self._cache_residual_counter]=system._residuals.asarray(copy=True)
        self._cache_residual_counter =self._cache_residual_counter + 1

    def cache_states_pin(self):
        """
        This function cache the nonlinear solution from IN
        """

        system = self._system()
        
        self._cache_states[:,self._cache_output_counter]=system._outputs.asarray(copy=True)
        self._cache_output_counter =self._cache_output_counter+ 1

    def compute_means(self, matrix):
        """
        Compute the mean residual vector
        """
        ncols = len(matrix[0][:])
        matrix_column_mean = np.sum(matrix, axis=1)

        return matrix_column_mean/ncols  
    
    def compute_centered_solution(self, matrix, mean_solution):
        """
        Compute the mean residual vector
        """
        ncols = len(matrix[0][:])

        means_expanded = np.outer(mean_solution, np.ones(ncols))

        centered_matrix = matrix - means_expanded

        return centered_matrix  


    def minimize_variance(self):
        """
        This function minimizes the variance and compute the low frequncey subspace of the nonlinear system
        """
        system = self._system()

        # svd on residuals

        mean_residuals = self.compute_means(self._cache_residuals)
        mean_centered_residuals = self.compute_centered_solution(self._cache_residuals,mean_residuals)
        self._p_residual, _, _ = np.linalg.svd(mean_centered_residuals, full_matrices=True)

        # svd on states
        mean_states = self.compute_means(self._cache_states)
        mean_centered_states = self.compute_centered_solution(self._cache_states,mean_states)
        self._p_states, _, _ = np.linalg.svd(mean_centered_states, full_matrices=True)


