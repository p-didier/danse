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

import danse_toolbox.d_post as pp

def danse(
    wasnObj: WASN,
    p: base.DANSEparameters
    ) -> tuple[DANSEoutputs, list[str]]:
    """
    Fully connected DANSE main function.

    Parameters
    ----------
    wasnObj : `WASN` object
        WASN under consideration.
    p : DANSEparameters object
        Parameters.

    Returns
    -------
    out : DANSEoutputs object
        DANSE outputs.
    wasnObj : `WASN` object
        WASN under consideration, after DANSE.

    References
    ----------
    [1] A. Bertrand and M. Moonen, "Distributed Adaptive Node-Specific Signal
    Estimation in Fully Connected Sensor Networks—Part I: Sequential Node
    Updating," in IEEE Transactions on Signal Processing, vol. 58, no. 10,
    pp. 5277-5291, Oct. 2010, doi: 10.1109/TSP.2010.2052612.
    """

    # Initialize variables
    dv = DANSEvariables()
    dv.import_params(p)
    dv.init_from_wasn(wasnObj.wasn)

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

        # Parse event matrix and inform user (if asked)
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
    for k in range(len(wasnObj.wasn)):
        wasnObj.wasn[k].enhancedData = dv.d[:, k]
        if dv.computeCentralised:
            wasnObj.wasn[k].enhancedData_c = dv.dCentr[:, k]
        if dv.computeLocal:
            wasnObj.wasn[k].enhancedData_l = dv.dLocal[:, k]

    return out, wasnObj


def tidanse(
    wasnObj: WASN,
    p: base.DANSEparameters
    ) -> tuple[DANSEoutputs, list[str]]:
    """
    Topology-independent DANSE main function.

    Parameters
    ----------
    wasnObj : `WASN` object
        WASN under consideration.
    p : DANSEparameters object
        Parameters.

    Returns
    -------
    out : DANSEoutputs object
        DANSE outputs.
    wasnObj : `WASN` object
        WASN under consideration, after DANSE.

    References
    ----------
    [1] J. Szurley, A. Bertrand and M. Moonen, "Topology-Independent
    Distributed Adaptive Node-Specific Signal Estimation in Wireless
    Sensor Networks," in IEEE Transactions on Signal and Information
    Processing over Networks, vol. 3, no. 1, pp. 130-144, March 2017,
    doi: 10.1109/TSIPN.2016.2623095.
    """

    # Prune WASN to tree
    wasnTreeObj = base.prune_wasn_to_tree(
        wasnObj,
        algorithm=p.treeFormationAlgorithm,
        plotit=False
    )

    # Initialize TI-DANSE variables
    tidv = TIDANSEvariables(p, wasnTreeObj.wasn)

    # Compute events
    eventInstants, fs = base.initialize_events(
        timeInstants=tidv.timeInstants,
        p=p,
        nodeTypes=[node.nodeType for node in wasnTreeObj.wasn],
        leafToRootOrdering=wasnTreeObj.leafToRootOrdering
    )

    # Profiling
    def is_interactive():  # if file is run from notebook
        import __main__ as main
        return not hasattr(main, '__file__')
    if not is_interactive():
        profiler = Profiler()
        profiler.start()
    t0 = time.perf_counter()    # timing

    # Loop over event instants
    for idx_t in range(len(eventInstants)):

        # Parse event matrix and inform user (if asked)
        if tidv.printout_eventsParser:
            base.events_parser_ti_danse(
                eventInstants[idx_t],
                tidv.startUpdates,
                tidv.printout_eventsParserNoBC
            )

        # Process events at current instant
        currEvents = eventInstants[idx_t] 
        for idx_e in range(currEvents.nEvents):
            evType = currEvents.type[idx_e]     # event type
            k = currEvents.nodes[idx_e]         # node index
            if not currEvents.bypass[idx_e]:
                if evType == 'fu':
                    # Fuse local signals
                    tidv.ti_fusion(k, currEvents.t, fs[k])
                elif evType == 'bc':
                    # Build partial in-network sum and broadcast downstream
                    tidv.ti_compute_partial_sum(k)
                    tidv.ti_broadcast_partial_sum_downstream(k)
                elif evType == 're':
                    # Relay in-network sum upstream
                    tidv.ti_relay_innetwork_sum_upstream(k)
                elif evType == 'up':
                    # Update DANSE filter coefficients and estimate target
                    tidv.ti_update_and_estimate(k, currEvents.t, fs[k])
                else:
                    raise ValueError(
                        f'Unknown event type: "{evType}".'
                    )
            else:
                print(f'Event at node {currEvents.nodes[idx_e]} at t={np.round(currEvents.t[idx_e], 3)}s (type: "{evType}") is bypassed.')
    
    # Profiling
    if not is_interactive():
        profiler.stop()
        if tidv.printout_profiler:
            profiler.print()
    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.amax(tidv.timeInstants)}s of signal processed in \
        {str(datetime.timedelta(seconds=dur))}.')
    print(f'(Real-time processing factor: \
        {np.round(np.amax(tidv.timeInstants) / dur, 4)})')

    # Build output
    out = DANSEoutputs()
    out.import_params(p)
    out.from_variables(tidv)
    # Update WASN object
    for k in range(len(wasnTreeObj.wasn)):
        wasnTreeObj.wasn[k].enhancedData = tidv.d[:, k]
        if tidv.computeCentralised:
            wasnTreeObj.wasn[k].enhancedData_c = tidv.dCentr[:, k]
        if tidv.computeLocal:
            wasnTreeObj.wasn[k].enhancedData_l = tidv.dLocal[:, k]

    return out, wasnTreeObj