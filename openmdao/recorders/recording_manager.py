"""
RecordingManager class definition.
"""
import time

from openmdao.utils.om_warnings import issue_warning


class RecordingManager(object):
    """
    Object that routes function calls to all recorders that are attached to the same object.

    Attributes
    ----------
    _recorders : list of CaseRecorder
        All of the recorders attached to the current object.
    """

    def __init__(self):
        """
        init.
        """
        self._recorders = []

    def __getitem__(self, index):
        """
        Get a particular recorder in the manager.

        Parameters
        ----------
        index : int
            an index into _recorders.

        Returns
        -------
        recorder : CaseRecorder
            a recorder from _recorders
        """
        return self._recorders[index]

    def __iter__(self):
        """
        Iterate.

        Returns
        -------
        iter : CaseRecorder
            a recorder from _recorders.
        """
        return iter(self._recorders)

    def append(self, recorder):
        """
        Add a recorder for recording.

        Parameters
        ----------
        recorder : CaseRecorder
           Recorder instance to be added to the manager.
        """
        self._recorders.append(recorder)

    def startup(self, recording_requester, comm=None):
        """
        Run startup on each recorder in the manager.

        Parameters
        ----------
        recording_requester : object
            The object that needs an iteration of itself recorded.
        comm : MPI.Comm or <FakeComm> or None
            The communicator for recorders (should be the comm for the Problem).
        """
        for recorder in self._recorders:
            recorder.startup(recording_requester, comm)

    def shutdown(self):
        """
        Shut down and remove all recorders.
        """
        for recorder in self._recorders:
            recorder.shutdown()
        self._recorders = []

    def record_iteration(self, recording_requester, data, metadata):
        """
        Call record_iteration on all recorders.

        Parameters
        ----------
        recording_requester : object
            The object that needs an iteration of itself recorded.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Metadata for iteration coordinate.
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.perf_counter()

        for recorder in self._recorders:
            recorder.record_iteration(recording_requester, data, metadata)

    def record_derivatives(self, recording_requester, data, metadata):
        """
        Call record_derivatives on all recorders.

        Parameters
        ----------
        recording_requester : object
            The object that needs an iteration of itself recorded.
        data : dict
            Dictionary containing derivatives keyed by 'of,wrt' to be recorded.
        metadata : dict
            Metadata for iteration coordinate.
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.perf_counter()

        for recorder in self._recorders:
            recorder.record_derivatives(recording_requester, data, metadata)

    def has_recorders(self):
        """
        Are there any recorders managed by this RecordingManager.

        Returns
        -------
        True/False: bool
            True if RecordingManager is managing at least one recorder.
        """
        return True if self._recorders else False

    def _check_parallel(self):
        pset = {bool(r.parallel) for r in self._recorders}

        # check to make sure we don't have mixed parallel/non-parallel, because that
        # currently won't work properly.
        if len(pset) > 1:
            raise RuntimeError("OpenMDAO currently does not support a mixture of parallel "
                               "and non-parallel recorders.")
        return pset.pop()

    def _check_gather(self):
        for rec in self._recorders:
            if rec._do_gather:
                return True


def _get_all_requesters(problem):
    yield problem
    yield problem.driver
    for system in problem.model.system_iter(include_self=True, recurse=True):
        yield system
        nl = system._nonlinear_solver
        if nl:
            yield nl
            if nl.linesearch:
                yield nl.linesearch


def _get_all_viewer_data_recorders(problem):
    for req in _get_all_requesters(problem):
        for r in req._rec_mgr._recorders:
            if r._record_viewer_data:
                yield r


def _get_all_recorders(problem):
    for req in _get_all_requesters(problem):
        for r in req._rec_mgr._recorders:
            yield r


def record_viewer_data(problem):
    """
    Record model viewer data for all recorders that have that option enabled.

    We don't want to collect the viewer data if it's not needed though,
    so first we'll find all recorders that need the data (if any) and
    then record it for those recorders.

    Parameters
    ----------
    problem : Problem
        The problem for which model viewer data is to be recorded.
    """
    # get all recorders that need to record the viewer data
    recorders = set(_get_all_viewer_data_recorders(problem))

    # if any recorders were found, get the viewer data and record it
    if recorders:
        from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
        try:
            viewer_data = _get_viewer_data(problem, values=True)
        except TypeError as err:
            viewer_data = {}
            issue_warning(str(err))

        viewer_data['md5_hash'] = problem.model._generate_md5_hash()
        viewer_data.pop('abs2prom', None)  # abs2prom already recorded in metadata table
        for recorder in recorders:
            recorder.record_viewer_data(viewer_data)


def record_model_options(problem, run_number):
    """
    Record the options for all systems and solvers in the model.

    Parameters
    ----------
    problem : Problem
        The problem for which all its system and solver options are to be recorded.
    run_number : int or None
        Number indicating which run the metadata is associated with.
        Zero or None for the first run, 1 for the second, etc.
    """
    # for backward compatibility, the first run does not have a run number
    if run_number is not None and run_number < 1:
        run_number = None

    recorders = set(_get_all_recorders(problem))

    for system in problem.model.system_iter(recurse=True, include_self=True):
        for recorder in recorders:
            # record system metadata (options)
            recorder.record_metadata_system(system, run_number)

            # record solver metadata (options) for this system's solvers
            nl = system._nonlinear_solver
            if nl:
                recorder.record_metadata_solver(nl, run_number)
                if nl.linesearch:
                    recorder.record_metadata_solver(nl.linesearch, run_number)

            ln = system._linear_solver
            if ln:
                recorder.record_metadata_solver(ln, run_number)
