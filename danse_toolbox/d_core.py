# Core functions for DANSE.
#
# ~created on 19.10.2022 by Paul Didier

# General TODO:'s
# -- Allow computation of local and centralised estimates

import time, datetime
from pyinstrument import Profiler
import danse.danse_toolbox.d_base as base
from danse.danse_toolbox.d_sros import *
from danse.danse_toolbox.d_classes import *


def danse(wasn: list[Node], p: DANSEparameters) -> DANSEoutputs:
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
    """

    # Initialize variables
    dv = DANSEvariables().fromWASN(wasn)

    # Events
    eventInstants, fs = base.initialize_events(dv.timeInstants, p)

    # Profiling
    profiler = Profiler()
    profiler.start()
    t0 = time.perf_counter()    # timing

    # Loop over event instants
    for idx_t in range(len(eventInstants)):

        # Parse event matrix and inform user (is asked)
        base.events_parser(
            eventInstants[idx_t],
            dv.startUpdates,
            p.printout_eventsParser,
            p.printout_eventsParserNoBC
        )

        # Process events at current instant
        events = eventInstants[idx_t] 
        for idx_e in range(events.nEvents):
            k = events.nodes[idx_e]  # node index
            # Broadcast event
            if events.type[idx_e] == 'bc':
                # TODO: maybe not needed to recompute `fs`
                # (already there in `wasn[k].fs`)
                dv.broadcast(wasn[k].data, events.t, fs[k], k, p)
            
            # Filter updates and desired signal estimates event
            elif events.type[idx_e] == 'up':
                dv.update_and_estimate(wasn[k].data, events.t, fs[k], k, p)
            else:
                raise ValueError(f'Unknown event: "{events.type[idx_e]}".')

    # Profiling
    profiler.stop()
    if p.printout_profiler:
        profiler.print()

    print('\nSimultaneous DANSE processing all done.')
    dur = time.perf_counter() - t0
    print(f'{np.amax(dv.timeInstants)}s of signal processed in \
        {str(datetime.timedelta(seconds=dur))}.')
    print(f'(Real-time processing factor: \
        {np.round(np.amax(dv.timeInstants) / dur, 4)})')

    out = DANSEoutputs().fromVariables(dv)

    return out