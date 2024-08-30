"""Define the NewtonSolver class."""


import numpy as np

from scipy.sparse import csc_matrix 
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS, ArmijoGoldsteinLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning, SolverWarning

class MultiPreconditionedNewton(NonlinearSolver):
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

    SOLVER = 'NL: MPNewton'

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
        self.linesearchPrecond = BoundsEnforceLS()
        self.linesearchPrecondSubLevel = BoundsEnforceLS()
        self.linesearchPostNewton = BoundsEnforceLS()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('solve_subsystems', types=bool,
                             desc='Set to True to turn on sub-solvers (Hybrid Newton).')
        self.options.declare('max_sub_solves', types=int, default=100,
                             desc='Maximum number of subsystem solves.')
        self.options.declare('level_1st_precond_max_sub_solves', types=int, default=15,
                             desc='Maximum number of first-level precondition solves.')
        self.options.declare('level_2nd_precond_max_sub_solves', types=int, default=10,
                             desc='Maximum number of second-level precondition solves.')
        self.options.declare('minimum_training_newton_iter', types=int, default=5,
                             desc='Minimum number of training iterations of Newton.')
        self.options.declare('minimum_training_1st_level_precond_iter', types=int, default=3,
                             desc='Maximum number of training iterations of inexacat newton.')

        self.options.declare('level_1st_precond_rtol', default=1e-2,
                             desc='relative error tolerance for the 1st level preconditioned stage')
        self.options.declare('level_2nd_precond_rtol', default=1e-2,
                             desc='relative error tolerance for the 2nd level preconditioned stage')
        self.options.declare('Newton_train_stall_tol', default=1e-2,
                             desc='Stall tolerance for the Newton training stage. When the Newton stalls for this tolerance and the training iteration is higher than minimum_training_newton_iter, precondtioning step will be activated.')
        self.options.declare('level_1st_precond_stall_tol', default=1e-1,
                             desc='Stall tolerance for the 1st level precond training stage. When the precondition stalls for this tolerance and the training iteration is higher than minimum_training_1st_level_precond_iter, 2nd-level precondtioning will be activated.')
        self.options.declare('Newton_train_stall_tol_type', default='rel', values=('abs', 'rel'),
                             desc='Specifies whether the absolute or relative norm of the '
                                  'residual is used for Newton stall detection in the training stage.')
        self.options.declare('solve_2nd_level_precond', types=bool,default=False,
                             desc='Set to True to turn on activate 2nd level preconditioning.')
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
        if self.linesearchPrecond is not None:
            self.linesearchPrecond._setup_solvers(system, self._depth + 1)
        if self.linesearchPrecondSubLevel is not None:
            self.linesearchPrecondSubLevel._setup_solvers(system, self._depth + 1)
        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)
        if self.linesearchPostNewton is not None:
            self.linesearchPostNewton._setup_solvers(system, self._depth + 1)

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
        if self.linesearchPrecond is not None:
            self.linesearchPrecond._set_solver_print(level=level, type_=type_)
        if self.linesearchPrecondSubLevel is not None:
            self.linesearchPrecondSubLevel._set_solver_print(level=level, type_=type_)
        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_)
            
        if self.linesearchPostNewton is not None:
            self.linesearchPostNewton._set_solver_print(level=level, type_=type_)

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

        system = self._system()
        return (self.options['solve_subsystems'] and not system.under_complex_step
                and self._iter_count <= self.options['max_sub_solves'])

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.linear_solver is not None:
            self.linear_solver._linearize()
        if self.linesearchPrecond is not None:
            self.linesearchPrecond._linearize()
        if self.linesearchPrecondSubLevel is not None:
            self.linesearchPrecondSubLevel._linearize()
        if self.linesearch is not None:
            self.linesearch._linearize()
        if self.linesearchPostNewton is not None:
            self.linesearchPostNewton._linearize()

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

        self._iter_count_precond = 0

        self._cache_residual_counter_precond = 0
        self._cache_output_counter_precond = 0

        self._preconditioned = False
        self._trained_PIN = False

        self._preconditioned_precond = False
        self._trained_PIN_precond = False

        system = self._system()
        solve_subsystems = self.options['solve_subsystems'] and not system.under_complex_step

        

        n_rows = len(system._residuals.asarray(copy=True))
        n_cols = self.options['minimum_training_newton_iter']
        n_cols_precond = self.options['minimum_training_1st_level_precond_iter']
        self._cache_residuals = np.array(np.zeros((n_rows, n_cols), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype))
        # IN cache outputs
        self._cache_states =  np.array(np.zeros((n_rows, n_cols), dtype=system._vectors["output"]["linear"].asarray(copy=True).dtype))

        self._cache_residuals_precond = np.asarray(np.zeros((n_rows, n_cols_precond), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype))
        # IN cache outputs
        self._cache_states_precond =  np.asarray(np.zeros((n_rows, n_cols_precond), dtype=system._vectors["output"]["linear"].asarray(copy=True).dtype))


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
            self.abs0 = norm
            self.train_stall_norm=norm
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

            if not self._preconditioned:
                if self.linesearch and not system.under_complex_step:
                    self.linesearch._do_subsolve = do_subsolve
                    self.linesearch.solve()
                else:
                    system._outputs += system._doutputs
            else:
                if self.linesearchPostNewton and not system.under_complex_step:
                    self.linesearchPostNewton._do_subsolve = do_subsolve
                    self.linesearchPostNewton.solve()
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
                init_residual = system._residuals.asarray(copy=True)
                # self._run_apply()
                # norm = self._iter_get_norm()
                

    def _train_for_PIN(self):
        """
        This function train the nonlinear system for getting the reduce subspace
        """
        atol = self.options['atol']
        rtol = self.options['rtol']

        # IN cache residual
        system = self._system()
        prefix = self._solver_info.prefix + self.SOLVER
        ### Step 1 ###

        # run inexact newton steps for training
        # iter = 0
        stall_limit = self.options['minimum_training_newton_iter']
        train_stall_tol = self.options['Newton_train_stall_tol']
        train_stall_tol_type = self.options['Newton_train_stall_tol_type']
        # norm = self._iter_get_norm()
        # abs = norm
        # if norm0 == 0:
        #     norm0 = 1
        # rel = norm / self._norm0
        
        stalled = False 
        # if self._iter_count<self.options['minimum_training_newton_iter']:

        if system.comm.rank==0 and self._iter_count==0:
            print(prefix+" starting Newton training ")

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
        
        # Check if convergence is stalled.
        if stall_limit > 0:
            norm_for_stall = rel_IN if train_stall_tol_type == 'rel' else abs_IN
            norm_diff = self.train_stall_norm - norm_for_stall
            if norm_diff <= train_stall_tol:
                if self._iter_count >= stall_limit-1:
                    stalled = True
            else:
                self.train_stall_norm = norm_for_stall

        # iter = iter + 1
        if self._iter_count>=self.options['minimum_training_newton_iter']-1 and stalled:
            self._trained_PIN = True
            if system.comm.rank==0:
                
                print(prefix +" ending Newton training ")
        if abs_IN < atol or rel_IN < rtol:
            self._trained_PIN = True
            self._preconditioned = True
            return 0


    def _train_for_PIN_precond(self, dim_red_resid, dim_red_states):
        """
        This function train the nonlinear system for getting the reduce subspace
        """
        atol = self.options['atol']
        rtol = self.options['rtol']

        # IN cache residual
        system = self._system()

        ### Step 1 ###

        prefix = self._solver_info.prefix + self.SOLVER
        # run inexact newton steps for training
        if self._iter_count_precond<self.options['minimum_training_1st_level_precond_iter']:

            # compute the residuals
            self._run_apply()

            # cahce the resd for training
            self.cache_residuals_pin_precond(dim_red_resid)
            # cahce the outp for training
            self.cache_states_pin_precond(dim_red_states)
            
            # compute the norm and see if the system has already converged
            norm = self._iter_get_norm()

            abs_IN = norm
            rel_IN = norm / self.abs0 if self.abs0 != 0.0 else 1.0

            self._iter_count_precond = self._iter_count_precond + 1

            if abs_IN < atol or rel_IN < rtol:
                self._preconditioned_precond = True
                self._trained_PIN_precond = True
                return 0

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """

        ### Step 1 ###
        
        # First, we run the newton solver sequentially
        # Call the training function
        if not self._trained_PIN:
            self._train_for_PIN()
            return None

        system = self._system()
        # self._solver_info.append_subsolver()
        do_subsolve = self.options['solve_subsystems'] and not system.under_complex_step and \
            (self._iter_count < self.options['max_sub_solves'])
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            if not self._preconditioned:
                # compute P and Q projectors
                self.minimize_variance()


                ### Step 2 ###
                # call the precondition convergence
                self._preconditioning_iter()
                # init_residual = system._residuals.asarray(copy=True)
            # run the inexact newton
            self._inexact_newton_iteration()
            init_residual = system._residuals.asarray(copy=True)
        finally:
            # Enable local fd
            system._owns_approx_jac = approx_status
        
    
    def _preconditioning_iter(self):
        rtol = self.options['level_1st_precond_rtol']
        stall_limit = self.options['minimum_training_1st_level_precond_iter']
        precond_stall_tol = self.options['level_1st_precond_stall_tol']
        system = self._system()
        # self._solver_info.append_subsolver()
        do_subsolve = self.options['solve_subsystems'] and not system.under_complex_step and \
            (self._iter_count < self.options['max_sub_solves'])
        do_sub_ln = self.linear_solver._linearize_children()

        

        # compute mean residual from the IN
        mean_residuals = self.compute_means(self._cache_residuals)

        # P^TP
        self.PTP= np.matmul(self._p_residual, self._p_residual.transpose())

        # resdiual subspace projector f(Y_j) = PP^T(F(Y_j)-F_mean) + F_mean

        init_residual = system._residuals.asarray(copy=True)
        
        # init_states = system._outputs
        projected_approx_residual_init = self.PTP.dot(init_residual-mean_residuals) + mean_residuals
        
        self.projected_approx_residual = projected_approx_residual_init
        # self._preconditioned = True
        # rel_IN = projected_approx_residual_init / projected_approx_residual_init if self.abs0 != 0.0 else 1.0
        # converge the reduce nonlinear space
        iter=0
        self._iter_count_precond = 0
        prefix = self._solver_info.prefix + self.SOLVER
        self.precond_stall_norm = 1.0
        stalled = False
        init_precond_norm = np.linalg.norm(projected_approx_residual_init)  if np.linalg.norm(projected_approx_residual_init) != 0.0 else 1
        while np.linalg.norm(self.projected_approx_residual) > rtol* init_precond_norm and iter < self.options['level_1st_precond_max_sub_solves']:
        
            if iter==0:
                if system.comm.rank==0:
                    print(prefix +" starting preconditioning")
                    if self.options['solve_2nd_level_precond']:
                        print(prefix +" starting 1st level precondition training ")       

            dimension_reduced_residual = self._p_residual.transpose().dot(self.projected_approx_residual)

            # Disable local fd
            approx_status = system._owns_approx_jac
            system._owns_approx_jac = False

            # compute the projected jacobian
            # matrix = self.linear_solver._assembled_jac
            my_asm_jac = system.linear_solver._assembled_jac

            system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
            if (my_asm_jac is not None and
                    system.linear_solver._assembled_jac is not my_asm_jac):
                my_asm_jac._update(system)

            self._linearize()
            # self.linear_solver.solve('fwd')
            my_asm_jac = system._linear_solver._assembled_jac._int_mtx._matrix
            if my_asm_jac is not None:
                if isinstance(my_asm_jac, csc_matrix):
                    jac_lup = my_asm_jac.todense()
                elif isinstance(my_asm_jac, np.ndarray):
                    jac_lup = my_asm_jac
            else:
                jac_lup = system.linear_solver._build_mtx()
        
            # P^T x Jac
            p_residual_csc = np.matmul(jac_lup,self._p_states)
            # Jp = P^T x Jac x Q
            Jac_p = np.matmul( self._p_residual.transpose(),p_residual_csc )


            Sub_sol = np.linalg.solve(Jac_p, -dimension_reduced_residual)

            
            if self.linesearchPrecond and not system.under_complex_step:
                system._doutputs.set_val(self._p_states.dot(Sub_sol))
                self.linesearchPrecond._do_subsolve = do_subsolve
                self.linesearchPrecond.solve()                    
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

            init_residual = system._residuals.asarray(copy=True)
            # system._dresiduals.set_vec(system._residuals)
            self.projected_approx_residual = self.PTP.dot(init_residual-mean_residuals) + mean_residuals
            
             # Check if convergence is stalled.
            if stall_limit > 0:
                norm_for_stall = np.linalg.norm(self.projected_approx_residual)/init_precond_norm
                norm_diff = self.precond_stall_norm - norm_for_stall 
                # print("stall dif: ",norm_for_stall , norm_diff, self.precond_stall_norm)
                if norm_diff <= precond_stall_tol:
                    if iter >= stall_limit-1:
                        # stalling. so end the end training
                        stalled = True
                else:
                    self.precond_stall_norm = norm_for_stall

            # Call the training function
            if not self._trained_PIN_precond and self.options['solve_2nd_level_precond']:
                self._train_for_PIN_precond(self.projected_approx_residual, system._outputs.asarray(copy=True))
                if stalled:
                    self._trained_PIN_precond = True
                    if system.comm.rank==0:
                        print(prefix +" ending 1st level precond training")

                # return None

            if not self._preconditioned_precond and self._trained_PIN_precond and self.options['solve_2nd_level_precond']:
                # compute P and Q projectors
                self.minimize_variance_precond()
                self._preconditioning_sub_iter()
                init_residual = system._residuals.asarray()
                self.projected_approx_residual = self.PTP.dot(init_residual-mean_residuals) + mean_residuals

            iter = iter + 1
            if np.linalg.norm(self.projected_approx_residual) <= rtol*init_precond_norm and iter <= self.options['level_1st_precond_max_sub_solves']:
                if system.comm.rank==0:
                    print(prefix +" 1st level preconditioned converged")

            elif iter >= self.options['level_1st_precond_max_sub_solves']:
                if system.comm.rank==0:
                    print(prefix +" 1st level precondition did not converged")
                    msg = (f"Solver '{self.SOLVER}' 1st level precondition on system '{system.pathname}' stalled after "
                f"{iter} iterations.")
                    # self.report_failure(msg)
                    issue_warning(msg, category=SolverWarning)
            system._owns_approx_jac = approx_status
        self._preconditioned=True
        self.del_cached_matrices()

            

    def _preconditioning_sub_iter(self):
        rtol = self.options['level_2nd_precond_rtol']
        system = self._system()
        # self._solver_info.append_subsolver()
        do_subsolve = self.options['solve_subsystems'] and not system.under_complex_step and \
            (self._iter_count < self.options['max_sub_solves'])
        do_sub_ln = self.linear_solver._linearize_children()

        

        # compute mean residual from the IN
        mean_residuals = self.compute_means(self._cache_residuals)
        mean_residuals_precond = self.compute_means(self._cache_residuals_precond)

        # P^TP
        PTPcond= np.matmul(self._p_residual_precond, self._p_residual_precond.transpose())

        # resdiual subspace projector f(Y_j) = PP^T(F(Y_j)-F_mean) + F_mean
        init_residual = self._cache_residuals_precond[:,-1]


        projected_approx_residual_init = PTPcond.dot(init_residual-mean_residuals_precond) + mean_residuals_precond
        self.projected_approx_residual_precond = projected_approx_residual_init

        prefix = self._solver_info.prefix + self.SOLVER
        # converge the reduce nonlinear space
        init_secondprecond_norm = np.linalg.norm(projected_approx_residual_init) if np.linalg.norm(projected_approx_residual_init) != 0.0 else 1
        iter=0
        while np.linalg.norm(self.projected_approx_residual_precond) > rtol* init_secondprecond_norm and iter < self.options['level_2nd_precond_max_sub_solves']:
            # resdiual subspace projector f(Y_j) = PP^T(F(Y_j)-F_mean) + F_mean
            # projected_approx_residual = PTP.dot(init_residual-mean_residuals) + mean_residuals
            
            if iter==0 and system.comm.rank==0:
                print(prefix + " starting 2nd level preconditioning")
            # dimension reduced vector

            

            dimension_reduced_residual = self._p_residual_precond.transpose().dot(self.projected_approx_residual_precond)

            # Disable local fd
            approx_status = system._owns_approx_jac
            system._owns_approx_jac = False

            # compute the projected jacobian
            # matrix = self.linear_solver._assembled_jac
            my_asm_jac = system.linear_solver._assembled_jac

            system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
            if (my_asm_jac is not None and
                    system.linear_solver._assembled_jac is not my_asm_jac):
                my_asm_jac._update(system)

            self._linearize()
            # self.linear_solver.solve('fwd')
            my_asm_jac = system._linear_solver._assembled_jac._int_mtx._matrix
            if my_asm_jac is not None:
                if isinstance(my_asm_jac, csc_matrix):
                    jac_lup = my_asm_jac.todense()
                elif isinstance(my_asm_jac, np.ndarray):
                    jac_lup = my_asm_jac
            else:
                jac_lup = system.linear_solver._build_mtx()

            Jac_p_precond= np.matmul(self.PTP,jac_lup)
            # P_tranpose_Jac =p_residual_csc.multiply(my_asm_jac)
            # Jp = P^T x Jac x Q
            # Jac_p = np.matmul( p_residual_csc, self._p_states)
            # compute the projected jacobian        
            # P^T x Jac
            p_residual_csc = np.matmul(self._p_residual_precond.transpose(),Jac_p_precond)
            # P_tranpose_Jac =p_residual_csc.multiply(my_asm_jac)
            # Jp = P^T x Jac x Q
            Jac_p = np.matmul( p_residual_csc, self._p_states_precond)


            Sub_sol = np.linalg.solve(Jac_p, -dimension_reduced_residual)
            # system._dresiduals.set_vec(np.asarray(dimension_reduced_residual))
        
        
            if self.linesearchPrecondSubLevel and not system.under_complex_step:
                system._doutputs.set_val(0)
                system._doutputs += self._p_states_precond.dot(Sub_sol)
                # system._outputs
                

                # if np.linalg.norm(self.projected_approx_residual) > self.options['precondlinesearch_robust_tol']*np.linalg.norm(projected_approx_residual_init):

                self.linesearchPrecondSubLevel._do_subsolve = do_subsolve
                self.linesearchPrecondSubLevel.solve()
            else:
            
                system._outputs += self._p_states_precond.dot(Sub_sol)

            # Call the training function
            # if not self._trained_PIN_precond:
            #     self._train_for_PIN_precond(dimension_reduced_residual, system._outputs.asarray(copy=True))
            #     # return None

            # if not self._preconditioned_precond:


            # self._solver_info.pop()
            # system._outputs.set_vec(init_states)
            # Hybrid newton support.
            if do_subsolve:
                with Recording('Newton_subsolve', 0, self):
                    # self._solver_info.append_solver()
                    self._gs_iter()
                    # self._solver_info.pop()
            
            self._run_apply()

            init_residual = system._residuals.asarray(copy=True)
            # mean_residuals = self.compute_means(self._cache_residuals)
            init_residual = self.PTP.dot(init_residual-mean_residuals) + mean_residuals
            
            # system._dresiduals.set_vec(system._residuals)
            self.projected_approx_residual_precond = PTPcond.dot(init_residual-mean_residuals_precond) + mean_residuals_precond

            iter = iter + 1
            if np.linalg.norm(self.projected_approx_residual_precond) <= rtol*init_secondprecond_norm and iter <= self.options['level_2nd_precond_max_sub_solves']:
                if system.comm.rank==0:
                    print(prefix +" 2nd level preconditioned converged")

            elif np.linalg.norm(self.projected_approx_residual_precond) > rtol*init_secondprecond_norm and iter >= self.options['level_2nd_precond_max_sub_solves']:
                if system.comm.rank==0:
                    print(prefix +" 2nd level preconditioned did not converged")
                    msg = (f"Solver '{self.SOLVER}' 2nd level on system '{system.pathname}' stalled after "
                f"{iter} iterations.")
                    # self.report_failure(msg)
                    issue_warning(msg, category=SolverWarning)
            system._owns_approx_jac = approx_status
        self._preconditioned_precond=True
        
            

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
        training_step=self.options['minimum_training_newton_iter']
        
        if self._cache_residual_counter<training_step:
            self._cache_residuals[:,self._cache_residual_counter]=system._residuals.asarray(copy=True)
            
        else:
            self._cache_residuals=np.concatenate([self._cache_residuals, np.zeros((len(self._cache_residuals[:,0]),1))], axis=1) 
            self._cache_residuals[:,self._cache_residual_counter]=system._residuals.asarray(copy=True)
        self._cache_residual_counter =self._cache_residual_counter + 1

    def cache_states_pin(self):
        """
        This function cache the nonlinear solution from IN
        """

        system = self._system()
        training_step=self.options['minimum_training_newton_iter']
        
        if self._cache_output_counter<training_step:
        
            self._cache_states[:,self._cache_output_counter]=system._outputs.asarray(copy=True)
            
        else:
            self._cache_states=np.concatenate([self._cache_states, np.zeros((len(self._cache_states[:,0]),1))], axis=1) 
            self._cache_states[:,self._cache_output_counter]=system._outputs.asarray(copy=True) 
        self._cache_output_counter =self._cache_output_counter+ 1

    def cache_residuals_pin_precond(self, dim_red_resid):
        """
        This function cache the nonlinear residuals from IN
        """
        system = self._system()
        
        self._cache_residuals_precond[:,self._cache_residual_counter_precond]=dim_red_resid
        self._cache_residual_counter_precond =self._cache_residual_counter_precond + 1

    def cache_states_pin_precond(self, dim_red_states):
        """
        This function cache the nonlinear solution from IN
        """

        system = self._system()
        
        self._cache_states_precond[:,self._cache_output_counter_precond]=dim_red_states
        self._cache_output_counter_precond =self._cache_output_counter_precond+ 1

    def compute_means(self, matrix):
        """
        Compute the mean residual vector
        """
        
        ncols = len(matrix[0,:])
        matrix_column_mean = np.sum(matrix, axis=1)

        return matrix_column_mean/ncols  
    
    def compute_centered_solution(self, matrix, mean_solution):
        """
        Compute the mean residual vector
        """
        ncols = len(matrix[0,:])
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
        p_residual, p_sing, _ = np.linalg.svd(mean_centered_residuals, full_matrices=True)
        

        # svd on states
        mean_states = self.compute_means(self._cache_states)
        mean_centered_states = self.compute_centered_solution(self._cache_states,mean_states)
        p_states, _, _ = np.linalg.svd(mean_centered_states, full_matrices=True)

    
        self._p_residual = p_residual

        self._p_states = p_states


    def minimize_variance_precond(self):
        """
        This function minimizes the variance and compute the low frequncey subspace of the nonlinear system
        """
        system = self._system()

        # svd on residuals

        mean_residuals = self.compute_means(self._cache_residuals_precond)
        mean_centered_residuals = self.compute_centered_solution(self._cache_residuals_precond,mean_residuals)
        p_residual_precond, _, _ = np.linalg.svd(mean_centered_residuals, full_matrices=True)

        

        # svd on states
        mean_states = self.compute_means(self._cache_states_precond)
        mean_centered_states = self.compute_centered_solution(self._cache_states_precond,mean_states)
        p_states_precond, _, _ = np.linalg.svd(mean_centered_states, full_matrices=True)

        self._p_residual_precond = p_residual_precond

        self._p_states_precond = p_states_precond
    
    def del_cached_matrices(self):

        del self._cache_residuals
        del self._cache_states

        del self._cache_residuals_precond
        del self._cache_states_precond

        del self.PTP
        # del self._p_states

        del self._p_residual
        del self._p_states

        del self._p_residual_precond
        del self._p_states_precond