# Core functions for DANSE.
#
# ~created on 19.10.2022 by Paul Didier

# General TODO:'s
# -- Allow computation of local and centralised estimates

import time, datetime
from pyinstrument import Profiler
import danse_toolbox.d_base as base
from danse_toolbox.d_sros import *
from danse_toolbox.d_classes import *
from danse_toolbox.d_post import DANSEoutputs


def danse(
    wasn: list[Node],
    p: base.DANSEparameters
    ) -> tuple[DANSEoutputs, list[str]]:
    """
    Main DANSE function.

    Parameters
    ----------
    wasn : list of `Node` objects
        WASN under consideration.
    p : DANSEparameters object
        Parameters.

    Returns
    -------
    out : DANSEoutputs object
        DANSE outputs.
    wasn : list of `Node` objects
        WASN under consideration, after DANSE.
    """

    # Initialize variables
    dv = DANSEvariables()
    dv.import_params(p)
    dv.init_from_wasn(wasn)

    # Compute events
    eventInstants, fs = base.initialize_events(dv.timeInstants, p)

    # Profiling
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')
    if not is_interactive():
        profiler = Profiler()
        profiler.start()
    t0 = time.perf_counter()    # timing

    # Loop over event instants
    for idx_t in range(len(eventInstants)):

        # Parse event matrix and inform user (is asked)
        base.events_parser(
            eventInstants[idx_t],
            dv.startUpdates,
            dv.printout_eventsParser,
            dv.printout_eventsParserNoBC
        )

        # Process events at current instant
        events = eventInstants[idx_t] 
        for idx_e in range(events.nEvents):
            k = events.nodes[idx_e]  # node index
            # Broadcast event
            if events.type[idx_e] == 'bc':
                dv.broadcast(events.t, fs[k], k)
            # Filter updates and desired signal estimates event
            elif events.type[idx_e] == 'up':
                dv.update_and_estimate(events.t, fs[k], k)
            else:
                raise ValueError(f'Unknown event: "{events.type[idx_e]}".')

    # Profiling
    if not is_interactive():
        profiler.stop()
        if dv.printout_profiler:
            profiler.print()

    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.amax(dv.timeInstants)}s of signal processed in \
        {str(datetime.timedelta(seconds=dur))}.')
    print(f'(Real-time processing factor: \
        {np.round(np.amax(dv.timeInstants) / dur, 4)})')

    # Build output
    out = DANSEoutputs()
    out.import_params(p)
    out.from_variables(dv)
    # Update WASN object
    for k in range(len(wasn)):
        wasn[k].enhancedData = dv.d[:, k]
        if dv.computeCentralised:
            wasn[k].enhancedData_c = dv.dCentr[:, k]
        if dv.computeLocal:
            wasn[k].enhancedData_l = dv.dLocal[:, k]

    return out, wasn