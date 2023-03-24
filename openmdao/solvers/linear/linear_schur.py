"""Define the LinearBlockGS class."""

import sys
import numpy as np

from openmdao.core.constants import _UNDEFINED
from openmdao.solvers.solver import BlockLinearSolver
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS
from openmdao.utils.general_utils import ContainsAll
import scipy


class LinearSchur(BlockLinearSolver):
    """
    Linear block Gauss-Seidel solver.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _delta_d_n_1 : ndarray
        Cached change in the d_output vectors for the previous iteration. Only used if the
        aitken acceleration option is turned on.
    _theta_n_1 : float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on.
    """

    SOLVER = "LN: SCHUR"

    def __init__(self, mode_linear="rev", groupNames=["group1", "group2"], **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self._theta_n_1 = None
        self._delta_d_n_1 = None
        self._mode_linear = mode_linear
        self._groupNames = groupNames

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        # this solver does not iterate
        # self.options.undeclare("maxiter")
        # self.options.undeclare("err_on_non_converge")

        # self.options.undeclare("atol")
        # self.options.undeclare("rtol")
        self.options["maxiter"] = 1
        self.options.declare("use_aitken", types=bool, default=False, desc="set to True to use Aitken relaxation")
        self.options.declare("aitken_min_factor", default=0.1, desc="lower limit for Aitken relaxation factor")
        self.options.declare("aitken_max_factor", default=1.5, desc="upper limit for Aitken relaxation factor")
        self.options.declare("aitken_initial_factor", default=1.0, desc="initial value for Aitken relaxation factor")

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
        if self.options["use_aitken"]:
            if self._mode_linear == "fwd":
                self._delta_d_n_1 = self._system()._doutputs.asarray(copy=True)
            else:
                self._delta_d_n_1 = self._system()._dresiduals.asarray(copy=True)
            self._theta_n_1 = 1.0

        return super()._iter_initialize()

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """

        # if mode != self._mode_linear:
        #     raise ValueError(
        #         f"The solve function is called with {mode} mode. But the user defined the linear Schur solve to work in {self._mode_linear} mode"
        #     )
        system = self._system()
        mode = self._mode_linear
        # self._update_rhs_vec()

        use_aitken = self.options["use_aitken"]

        if use_aitken:
            aitken_min_factor = self.options["aitken_min_factor"]
            aitken_max_factor = self.options["aitken_max_factor"]

            # some variables that are used for Aitken's relaxation
            delta_d_n_1 = self._delta_d_n_1
            theta_n_1 = self._theta_n_1

            # store a copy of the outputs, used to compute the change in outputs later
            if self._mode_linear == "fwd":
                d_out_vec = system._doutputs
            else:
                d_out_vec = system._dresiduals

            d_n = d_out_vec.asarray(copy=True)
            delta_d_n = d_out_vec.asarray(copy=True)

        # take the subsystems
        subsys1, _ = system._subsystems_allprocs[self._groupNames[0]]
        subsys2, _ = system._subsystems_allprocs[self._groupNames[1]]

        # TODO this may not be the most general case. think about just solving for a subset
        subsys2_outputs = subsys2._doutputs
        subsys2_residuals = subsys2._dresiduals

        # list of variables we solve for here. this should include all variables in
        # subsys2 ideally because we dont do anything else for this subsystem here.
        vars_to_solve = [*subsys2_outputs.keys()]
        resd_to_solve = [*subsys2_residuals.keys()]

        # total size of the jacobian
        n_vars = 0
        for var in vars_to_solve:
            n_vars += subsys2_outputs[var].size

        # initialize the schur complement jacobian for these variables
        # TODO better way to get the dtype?
        schur_jac = np.zeros((n_vars, n_vars), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype)
        schur_rhs = np.zeros((n_vars), dtype=system._vectors["residual"]["linear"].asarray(copy=True).dtype)

        # backup the vectors we are working with
        rvec = system._vectors["residual"]["linear"]
        ovec = system._vectors["output"]["linear"]
        ivec = system._vectors["input"]["linear"]

        r_data = rvec.asarray(copy=True)
        o_data = ovec.asarray(copy=True)
        i_data = ivec.asarray(copy=True)

        if mode == "fwd":
            parent_offset = system._dresiduals._root_offset

            # must always do the transfer on all procs even if subsys not local
            # for subsys2 in subsystem_list:
            if self._rel_systems is not None and subsys2.pathname not in self._rel_systems:
                return
            # must always do the transfer on all procs even if subsys not local
            # system._transfer("linear", mode, subsys2.name)
            system._transfer("linear", mode, subsys2.name)

            if not subsys2._is_local:
                return

            # take the d_resdiuals for both of the subsys
            b_vec = subsys1._dresiduals
            off = b_vec._root_offset - parent_offset
            b_vec2 = subsys2._dresiduals
            off2 = b_vec2._root_offset - parent_offset

            # cache the rhs vector since we ll need this later
            subsys1_rhs = self._rhs_vec[off : off + len(b_vec)].copy()
            subsys2_rhs = self._rhs_vec[off2 : off2 + len(b_vec2)].copy()

            ########################
            #### schur_jacobian ####
            ########################

            ## Schur_Jac = D - C A^-1 B ##

            ovec.set_val(np.zeros(len(ovec)))

            for ii, var in enumerate(vars_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2
                ovec[f"{subsys2.name}.{var}"] = 1.0

                # transfer this seed to the first subsystem
                system._transfer("linear", mode, subsys1.name)

                # run the jac-vec computation in the first subsystem, this ll give us the B[:,{ii}] vector
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                subsys1._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                # amd then, by performing solve_linear we get A^-1 B[:,{ii}]
                subsys1._solve_linear(mode, ContainsAll())

                # do another mat-mult with the solution of this linear system, we want to get the final
                # jacobian using the schur method here, so we will need to do a bit more math

                # first negate the vector from the linear solve
                subsys1._vectors["output"]["linear"] *= -1.0

                # finally, set the seed of the variable to 1 as well to get the diagonal contribution
                # system._vectors["output"]["linear"][f"{subsys2.name}.{var}"]
                # this should already be at one since we perturbed it above!

                # transfer the outputs to inputs
                system._transfer("linear", mode)

                # run the apply linear. we do it on the complete system here
                # the result is the final jacobian for this using the schur complement method D[:,{ii}] - C A^-1 B[:,{ii}]
                scope_out, scope_in = system._get_matvec_scope()
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                system._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                # put this value into the jacobian.
                schur_jac[:, ii] = subsys2._vectors["residual"]["linear"].asarray()

                # set back the seed to zero for the next vector
                ovec[f"{subsys2.name}.{var}"] = 0.0

            # backup the vectors here
            rvec.set_val(r_data)
            ovec.set_val(o_data)

            # set the rhs vector
            b_vec.set_val(subsys1_rhs)

            ########################
            #### schur_jacobian ####
            ########################

            ################################
            #### Beg solve for subsys 2 ####
            ################################
            # now we work with the RHS
            subsys1._solve_linear(mode, ContainsAll())

            # first negate the vector from the linear solve
            subsys1._vectors["output"]["linear"] *= -1.0

            # set the inputs to be zero
            subsys2._dinputs.set_val(0.0)
            system._transfer("linear", "fwd", subsys2.name)

            # we do an apply linear on the subsys2 to get the negative part of the rhs
            scope_out, scope_in = system._get_matvec_scope(subsys2)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            subsys2._apply_linear(None, None, mode, scope_out, scope_in)

            # add the rhs vector with the resultant negative part
            b_vec2 += subsys2_rhs

            d_subsys2 = scipy.linalg.solve(schur_jac, subsys2._vectors["residual"]["linear"].asarray())

            # loop over the variables just to be safe with the ordering
            # subsys1._doutputs.set_val(subsys1_output)
            for ii, var in enumerate(vars_to_solve):
                system._doutputs[f"{subsys2.name}.{var}"] = d_subsys2[ii]

            ################################
            #### End solve for subsys 2 ####
            ################################

            ################################
            #### Beg solve for subsys 1 ####
            ################################

            # subsys1._doutputs.set_val(0.0)
            if self._rel_systems is not None and subsys1.pathname not in self._rel_systems:
                return
            # must always do the transfer on all procs even if subsys not local
            system._transfer("linear", mode, subsys1.name)

            if not subsys1._is_local:
                return

            scope_out, scope_in = system._get_matvec_scope(subsys1)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)

            if subsys1._iter_call_apply_linear():
                subsys1._apply_linear(None, None, mode, scope_out, scope_in)
                b_vec *= -1.0
                b_vec += self._rhs_vec[off : off + len(b_vec)]
            else:
                b_vec.set_val(self._rhs_vec[off : off + len(b_vec)])

            subsys1._solve_linear(mode, ContainsAll(), scope_out, scope_in)

            ################################
            #### End solve for subsys 1 ####
            ################################

        else:  # rev
            parent_offset = system._doutputs._root_offset

            # update the output of subsys2
            # system._transfer("linear", mode, subsys2.name)
            dinputs_cahce = system._dinputs.asarray(copy=True)
            b_vec = subsys1._doutputs
            # b_vec_cache1 = subsys1._doutputs.asarray(copy=True)
            off = b_vec._root_offset - parent_offset

            b_vec2 = subsys2._doutputs
            # b_vec_cache2 = subsys2._doutputs.asarray(copy=True)
            off2 = b_vec2._root_offset - parent_offset

            subsys1_rhs = self._rhs_vec[off : off + len(b_vec)].copy()
            subsys2_rhs = self._rhs_vec[off2 : off2 + len(b_vec2)].copy()

            ########################
            #### schur_jacobian ####
            ########################

            ## Schur_Jac = D - C A^-1 B ##

            rvec.set_val(np.zeros(len(rvec)))

            # inpu_cache=system._doutputs.asarray(copy=True)
            outp_cache = system._doutputs.asarray(copy=True)
            resd_cache = system._dresiduals.asarray(copy=True)

            for ii, var in enumerate(resd_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2

                rvec[f"{subsys2.name}.{var}"] = 1.0

                # we get the C[{ii},:] vector by apply_linear on the system
                scope_out, scope_in = system._get_matvec_scope()
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                system._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                scope_out, scope_in = system._get_matvec_scope(subsys1)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)

                # do a solve_linear to find C[{ii},:] A^-1
                subsys1._solve_linear(mode, self._rel_systems, scope_out, scope_in)

                # the same solve requires in the rhs too, so we save them
                schur_rhs[ii] = subsys1._vectors["residual"]["linear"].asarray().dot(subsys1_rhs)

                # negate the resdiual first
                subsys1._vectors["residual"]["linear"] *= -1.0

                # do a apply_linear on the subsys1 to find the D[{ii},:] - C[{ii},:] A^-1 B
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                subsys1._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

                system._transfer("linear", mode, subsys2.name)

                # put this value into the jacobian.
                schur_jac[ii, :] = subsys2._vectors["output"]["linear"].asarray()

                # set back the seed to zero for the next vector
                rvec[f"{subsys2.name}.{var}"] = 0.0

            ########################
            #### schur_jacobian ####
            ########################

            # put back the vectors
            rvec.set_val(r_data)
            ovec.set_val(o_data)
            ivec.set_val(i_data)

            ################################
            #### Beg solve for subsys 2 ####
            ################################
            system._dinputs.set_val(dinputs_cahce)
            system._doutputs.set_val(outp_cache)
            system._dresiduals.set_val(resd_cache)
            b_vec2.set_val(0.0)

            system._transfer("linear", mode, subsys2.name)

            b_vec2 *= -1.0
            # b_vec += subsys1_rhs

            b_vec2 += subsys2_rhs

            # b_vec.set_val(b_vec_cache1)
            # if self._rel_systems is None and subsys2.pathname in self._rel_systems:

            #     if subsys2._is_local:
            # b_vec2.set_val(b_vec_cache2)

            # b_vec2.set_val(0.0)
            # system._transfer("linear", mode, subsys2.name)
            # b_vec2 *= -1.0
            # b_vec2 += subsys2_rhs
            schur_rhs = subsys2_rhs - schur_rhs

            d_subsys2 = scipy.linalg.solve(schur_jac, schur_rhs)

            if system.comm.rank == 0:
                print("\nupdate vector: ", d_subsys2, flush=True)

            scope_out, scope_in = system._get_matvec_scope(subsys2)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)

            # loop over the variables just to be safe with the ordering
            for ii, var in enumerate(resd_to_solve):
                system._dresiduals[f"{subsys2.name}.{var}"] = d_subsys2[ii]

            # scope_out, scope_in = system._get_matvec_scope()
            # scope_out = self._vars_union(self._scope_out, scope_out)
            # scope_in = self._vars_union(self._scope_in, scope_in)
            # subsys1._dresiduals.set_val(0.0)
            # system._apply_linear(None, None, mode, scope_out, scope_in)

            if subsys2._iter_call_apply_linear():
                subsys2._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)
            else:
                b_vec2.set_val(0.0)
                # else:
                #     system._transfer('linear', mode, subsys2.name)
            ################################
            #### End solve for subsys 2 ####
            ################################

            ################################
            #### Beg solve for subsys 1 ####
            ################################
            # if self._rel_systems is None and subsys1.pathname in self._rel_systems:
            #     if subsys1._is_local:
            b_vec.set_val(0.0)

            system._transfer("linear", mode, subsys1.name)

            b_vec *= -1.0
            # b_vec += subsys1_rhs

            b_vec += subsys1_rhs

            scope_out, scope_in = system._get_matvec_scope(subsys1)
            scope_out = self._vars_union(self._scope_out, scope_out)
            scope_in = self._vars_union(self._scope_in, scope_in)
            subsys1._solve_linear(mode, self._rel_systems, scope_out, scope_in)

            if subsys1._iter_call_apply_linear():
                subsys1._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)
            else:
                b_vec.set_val(0.0)
                # else:
                #     system._transfer('linear', mode, subsys1.name)

            ################################
            #### End solve for subsys 1 ####
            ################################

        if use_aitken:
            if self._mode_linear == "fwd":
                d_resid_vec = system._dresiduals
                d_out_vec = system._doutputs
            else:
                d_resid_vec = system._doutputs
                d_out_vec = system._dresiduals

            theta_n = self.options["aitken_initial_factor"]

            # compute the change in the outputs after the NLBGS iteration
            delta_d_n -= d_out_vec.asarray()
            delta_d_n *= -1

            if self._iter_count >= 2:
                # Compute relaxation factor. This method is used by Kenway et al. in
                # "Scalable Parallel Approach for High-Fidelity Steady-State Aero-
                # elastic Analysis and Adjoint Derivative Computations" (ln 22 of Algo 1)

                temp = delta_d_n.copy()
                temp -= delta_d_n_1

                # If MPI, piggyback on the residual vector to perform a distributed norm.
                if system.comm.size > 1:
                    backup_r = d_resid_vec.asarray(copy=True)
                    d_resid_vec.set_val(temp)
                    temp_norm = d_resid_vec.get_norm()
                else:
                    temp_norm = np.linalg.norm(temp)

                if temp_norm == 0.0:
                    temp_norm = 1e-12  # prevent division by 0 below

                # If MPI, piggyback on the output and residual vectors to perform a distributed
                # dot product.
                if system.comm.size > 1:
                    backup_o = d_out_vec.asarray(copy=True)
                    d_out_vec.set_val(delta_d_n)
                    tddo = d_resid_vec.dot(d_out_vec)
                    d_resid_vec.set_val(backup_r)
                    d_out_vec.set_val(backup_o)
                else:
                    tddo = temp.dot(delta_d_n)

                theta_n = theta_n_1 * (1 - tddo / temp_norm**2)

            else:
                # keep the initial the relaxation factor
                pass

            # limit relaxation factor to the specified range
            theta_n = max(aitken_min_factor, min(aitken_max_factor, theta_n))

            # save relaxation factor for the next iteration
            self._theta_n_1 = theta_n

            d_out_vec.set_val(d_n)

            # compute relaxed outputs
            d_out_vec += theta_n * delta_d_n

            # save update to use in next iteration
            delta_d_n_1[:] = delta_d_n