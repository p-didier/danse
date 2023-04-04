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
    eventInstants, fs, _ = base.initialize_events(dv.timeInstants, p)

    # Profiling
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')
    if not is_interactive():
        profiler = Profiler()
        profiler.start()
    t0 = time.perf_counter()    # timing

    # Loop over event instants
    for idxInstant in range(len(eventInstants)):

        # Process events at current instant
        events = eventInstants[idxInstant] 

        # Parse event matrix and inform user (if asked)
        base.events_parser(
            events,
            dv.startUpdates,
            dv.printout_eventsParser,
            dv.printout_eventsParserNoBC
        )

        for idxEventCurrInstant in range(events.nEvents):
            k = events.nodes[idxEventCurrInstant]  # node index
            # Broadcast event
            if events.type[idxEventCurrInstant] == 'bc':
                dv.broadcast(events.t, fs[k], k)
            # Filter updates and desired signal estimates event
            elif events.type[idxEventCurrInstant] == 'up':
                dv.update_and_estimate(
                    events.t,
                    fs[k],
                    k,
                    events.bypassUpdate[idxEventCurrInstant]
                )
            else:
                raise ValueError(f'Unknown event: "{events.type[idxEventCurrInstant]}".')

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

    # Initialize TI-DANSE variables
    tidv = TIDANSEvariables(p, wasnObj.wasn)

    # Compute events
    eventInstants, fs, wasnObjList = base.initialize_events(
        timeInstants=tidv.timeInstants,
        p=p,
        wasnObj=wasnObj
    )

    # Profiling
    def is_interactive():  # if file is run from notebook
        import __main__ as main
        return not hasattr(main, '__file__')
    if not is_interactive():
        profiler = Profiler()
        profiler.start()
    t0 = time.perf_counter()    # timing

    # Plotting
    if p.printoutsAndPlotting.showWASNs:
        fig = plt.figure()
        fig.set_size_inches(4.5, 4.5)
        ax = fig.add_subplot(projection='3d')
        plt.show(block=False)
        scatterSize = np.amax(fig.get_size_inches()) * 50
    else:
        scatterSize, ax = None, None

    # Loop over event instants
    for instantIdx, currEvents in enumerate(eventInstants):

        # Parse event matrix and inform user (if asked)
        if tidv.printout_eventsParser:
            base.events_parser_ti_danse(
                currEvents,
                tidv.startUpdates,
                tidv.printout_eventsParserNoBC
            )

        for idxEventCurrInstant in range(currEvents.nEvents):
            # Event type
            evType = currEvents.type[idxEventCurrInstant]
            # Node index
            k = currEvents.nodes[idxEventCurrInstant]
            # Bypass update boolean
            bypassUpdate = currEvents.bypassUpdate[idxEventCurrInstant]

            if evType == 'fu':
                # Fuse local signals
                tidv.ti_fusion(k, currEvents.t, fs[k])
            elif evType == 'bc':
                # Build partial in-network sum and broadcast downstream
                tidv.ti_compute_partial_sum(k)
                # tidv.ti_broadcast_partial_sum_downstream(k)
            elif evType == 're':
                # Relay in-network sum upstream
                tidv.ti_relay_innetwork_sum_upstream(k)
            elif evType == 'up':
                # Update DANSE filter coefficients and estimate target
                tidv.ti_update_and_estimate(
                    k,
                    currEvents.t,
                    fs[k],
                    bypassUpdate
                )
                if k == tidv.currentWasnTreeObj.rootIdx and\
                    bypassUpdate:
                    raise ValueError()
            elif evType == 'tr':
                # New tree formation: update up-/downstream neighbors lists
                tidv.update_up_downstream_neighbors(
                    wasnObjList[tidv.treeFormationCounter],
                    plotit=p.printoutsAndPlotting.showWASNs,
                    ax=ax,
                    scatterSize=scatterSize
                )
                if p.printoutsAndPlotting.showWASNs:
                    ax.set_title(f't={np.round(currEvents.t, 3)}s; Tree formation #{tidv.treeFormationCounter}')
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.05)
            else:
                raise ValueError(f'Unknown event type: "{evType}".')
    
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
    for k in range(len(wasnObj.wasn)):
        wasnObj.wasn[k].enhancedData = tidv.d[:, k]
        if tidv.computeCentralised:
            wasnObj.wasn[k].enhancedData_c = tidv.dCentr[:, k]
        if tidv.computeLocal:
            wasnObj.wasn[k].enhancedData_l = tidv.dLocal[:, k]

    return out, wasnObj