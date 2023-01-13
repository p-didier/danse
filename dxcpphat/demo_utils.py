'''
Functions specific to "Distributed SRO Estimation and Synchronization" Demo
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from mpl_toolkits import mplot3d
from scipy.signal.windows import hann
from stl import mesh
from tqdm.notebook import tqdm
from multiprocessing import Process, Queue

from sro_estimation import CL_DXCPPhaT
from delay_buffer import DelayBuffer


def get_example_topology(form='SOT'):
    '''
    Returns one of three stock examples for topologies (see documentation)
    Input: 
        form (string): Example type. Either 'SOT', 'POT', or 'ROT'.
    Output:
        nodes_levels (list): Topology description.
    '''
    if form == 'SOT':
        nodes_levels = [
            [['node_0', 'node_2', 'node_3', 'node_4', 'node_5']]
        ]
    elif form == 'POT':
        nodes_levels = [
            [['node_0', 'node_2']],
            [['node_2', 'node_3']],
            [['node_3', 'node_4']],
            [['node_4', 'node_5']]
        ]
    elif form == 'ROT':
        nodes_levels = [
            [['node_0', 'node_2', 'node_3']],
            [['node_3', 'node_4', 'node_5']],
        ]
    else:
        nodes_levels = False
        print('Topology must be either SOT, POT, or ROT')
    
    return nodes_levels



# Compute smoothed node SNR trajectory
def compute_node_snr(sig, node_id, q):
    '''
    For parallel call: Compute regular and smoothed SNR trajectory for node signal.
    Input: 
        sig (N,) Mic signal
        node_id (string): node id, for output identification
        q: Output queue
    Output:
        node_snr (N,): SNR trajectory
        node_snr_smoothed (N,): Smoothed SNR trajectory
    '''
    n_var = (7.07*10**(-4))**2 #as configured during signal generation
    alpha=0.999
    smooth_window_smp = 16000*10

    sig_var = np.zeros(sig.shape)
    for smp in range(np.size(sig)):
        sig_var[smp] = alpha*sig_var[smp-1] + (1-alpha)*sig[smp]**2 if smp > 0 else 0
        if sig_var[smp] < n_var:
            sig_var[smp] = n_var

    sig_var_smoothed = signal.convolve(sig_var, np.ones(smooth_window_smp), mode='same') / smooth_window_smp
    s_var = sig_var-n_var
    s_var_smoothed = sig_var_smoothed-n_var
    node_snr = 10*np.log10(s_var/n_var, where=(s_var!=0))
    node_snr[np.argwhere(s_var==0)] = np.NINF
    node_snr_smoothed = 10*np.log10(s_var_smoothed/n_var, where=(s_var_smoothed>0))
    node_snr_smoothed[np.argwhere(s_var_smoothed<=0)] = np.NINF   

    q.put({
        'node_id': node_id,
        'node_snr': node_snr,
        'node_snr_smoothed': node_snr_smoothed
    })


def evaluate_simulation_results(nodes_select, fs, frame_len, SRO_est, SRO_true, signals, signals_synced, signals_sync, ssnr_t_max):
    '''
    Calculates various evaluation measures for synchronized signals
    Input:
        nodes_select (list):        Topology description
        fs (int):                   Sampling rate
        frame_len(int):             Signal frame length
        SRO_Est (np.array):         SRO estimates
        SRO_true (list):            SRO ground truth 
        signals (np.array):         Asynchronous signals
        signals_synced (np.array):  Synchronized signals
        signals_sync (np.array):    Synchronous signals
        ssnr_t_max (float):         Time in seconds up until which ssnr should be calculated
    Output: rmse_t, nodes_snr, nodes_snr_smoothed, ssnr
        rmse_t (np.array):          Time varying RMSE for SRO estimates
        nodes_snr (np.array):       SNR trajectory for all nodes
        nodes_snr_smoothed (np.array): Smoothed SNR trajectories
        ssnr (list):                Signal-to-synchronization-noise ratios after sync  
        ssnr_async(list):           Signal-to-synchronization-noise ratios before sync

    '''
    n_frames, n_async_nodes = np.shape(SRO_est)

    # RMSE(t)
    mse_t = np.zeros((n_frames, n_async_nodes))
    rmse_t = np.zeros((n_frames, n_async_nodes))
    for fr in range(n_frames):
        for n in range(n_async_nodes):
            if fr == 0:
                mse_t[fr, n] = (SRO_est[fr, n] - SRO_true[n])**2
            else:
                mse_t[fr, n] = 0.9*mse_t[fr-1, n] + 0.1*(SRO_est[fr, n] - SRO_true[n])**2
            rmse_t[fr, n] = np.sqrt(mse_t[fr, n])

    # Estimate residual time offset (RTO)
    rto = []
    for n in range(n_async_nodes):
        k_max = n_frames-1 if n_frames < int(np.floor(ssnr_t_max*fs/frame_len)) else int(np.floor(ssnr_t_max*fs/frame_len)) -1# at t=150s latest
        rto.append(np.sum(SRO_est[k_max, n]-SRO_est[:k_max, n])*1e-6*frame_len) # every estimate representative of frame_len samples

    # Compute settling times (RMSE < 1ppm)
    Tc = []
    for n in range(n_async_nodes):
        if np.size(np.argwhere(rmse_t[:, n] > 1)) == 0:
            Tc.append(-1)
        else:
            Tc.append(np.argwhere(rmse_t[:, n] > 1)[-1][0]*2**11/fs)

    # Compute global SINR for all signals
    ssnr = []
    ssnr_async = [] 
    for n in range(n_async_nodes):
        if Tc[n] == -1 or np.size(signals_synced[:,:,n+1].flatten()[int(Tc[n]*fs):int(ssnr_t_max*fs)]) < 2**13:
            ssnr.append(None)
            ssnr_async.append(None)
            print('Warning: No SSNR could be computed for ', nodes_select[n+1], '(signal segment too short). Please increase the signal duration and/or ssnr_t_max')
            continue
        # compensate rto
        s = gen_offset(signals_synced[:,:,n+1].flatten()[int(Tc[n]*fs):], rto[n], 2**13) 
        s_async = gen_offset(signals[:,:,n+1].flatten()[int(Tc[n]*fs):], rto[n], 2**13) 
        # ssnr after sync
        if np.size(s) > ssnr_t_max*fs:
            s = s[:int(ssnr_t_max*fs)]
        var_sync = np.var(signals_sync[:,:,n+1].flatten()[int(Tc[n]*fs):(len(s)+int(Tc[n]*fs))])
        var_diff = np.var(signals_sync[:,:,n+1].flatten()[int(Tc[n]*fs):(len(s)+int(Tc[n]*fs))] - s)
        ssnr.append(10*np.log10(var_sync/var_diff))
        # ssnr before sync
        if np.size(s_async) > ssnr_t_max*fs:
            s_async = s_async[:int(ssnr_t_max*fs)]
        var_async = np.var(signals[:,:,n+1].flatten()[int(Tc[n]*fs):(len(s_async)+int(Tc[n]*fs))])
        var_diff_async = np.var(signals_sync[:,:,n+1].flatten()[int(Tc[n]*fs):(len(s_async)+int(Tc[n]*fs))] - s_async)
        ssnr_async.append(10*np.log10(var_async/var_diff_async))

    procs = []
    q = Queue()
    got_results = {}
    for idx, node in enumerate(nodes_select):
        got_results[node] = False
        node_id = nodes_select.index(node)
        procs.append(Process(target=compute_node_snr, args=(signals[:,:,idx].flatten(), node, q)))
        procs[-1].start()
    nodes_snr = np.zeros((np.size(signals[:,:,0]), n_async_nodes+1))
    nodes_snr_smoothed = np.zeros((np.size(signals[:,:,0]), n_async_nodes+1))
    while True:
        res = q.get()
        node_idx = nodes_select.index(res['node_id'])
        got_results[res['node_id']] = True
        # save results
        nodes_snr[:, node_idx] = res['node_snr']
        nodes_snr_smoothed[:, node_idx] = res['node_snr_smoothed']
        # Check if finished
        all_fin = True
        for n in got_results.keys():
            if got_results[n] == False: 
                all_fin = False
                break
        if all_fin == True: break

    ''' #depr: non-parallel snr computation
    n_var = (7.07*10**(-4))**2
    alpha=0.999
    smooth_window_smp = 16000*10
    nodes_snr = np.zeros((np.size(signals[:,:,0]), n_async_nodes+1))
    nodes_snr_smoothed = np.zeros((np.size(signals[:,:,0]), n_async_nodes+1))
    for idx, node in enumerate(nodes_select):
        sig = signals[:,:,idx].flatten()
        sig_var = np.zeros(sig.shape)
        for smp in range(np.size(sig)):
            sig_var[smp] = alpha*sig_var[smp-1] + (1-alpha)*sig[smp]**2 if smp > 0 else 0
            if sig_var[smp] < n_var:
                sig_var[smp] = n_var
        sig_var_smoothed = signal.convolve(sig_var, np.ones(smooth_window_smp), mode='same') / smooth_window_smp
        nodes_snr[:, idx] = 10*np.log10((sig_var-n_var)/n_var)
        nodes_snr_smoothed[:, idx] = 10*np.log10((sig_var_smoothed-n_var)/n_var)
    '''

    return rmse_t, nodes_snr, nodes_snr_smoothed, ssnr, ssnr_async


def run_simulation_online(nodes_levels, signals, acs, control_init_sro_est=False):

    nodes_select = get_unique_node_list(nodes_levels)
    n_async_nodes = len(nodes_select)-1
    n_frames = np.shape(signals)[0]
    frame_len = np.shape(signals)[1]
    resamplerDelay = 2 # frames
    start_delay_per_level = 100 if control_init_sro_est else 0
    async_nodes_delays = get_async_node_delays(nodes_levels, delay_per_level=resamplerDelay)

    DXCP_PhaT_Estimators = [CL_DXCPPhaT(start_delay=async_nodes_delays[i]*start_delay_per_level) for i in range(n_async_nodes)]
    InputBuffers = [DelayBuffer((frame_len, async_nodes_delays[i]+1)) for i in range(n_async_nodes)]
    signals_synced = np.zeros_like(signals)
    signals_synced[:,:,0] = signals[:,:,0] # ref signal is already sync
    # -- main loop
    SRO_est = np.zeros((n_frames, n_async_nodes))
    dSRO_est = np.zeros((n_frames, n_async_nodes))
    for frame_idx in tqdm(range(n_frames)):
        root_node = nodes_select[0] # 1st level, 1st branch, 1st node
        synced_blocks_prev_level = {root_node: signals[frame_idx, :, 0].flatten()}
        for sid, level in enumerate(nodes_levels):
            synced_blocks_level = {}
            for branch in level:
                ref_node = branch[0]
                synced_block_ref = synced_blocks_prev_level[ref_node]
                for nid, node in enumerate(branch):
                    if nid == 0: continue
                    n = nodes_select.index(node)-1 #idx w.r.t. async (not all) nodes
                    InputBuffers[n].write(signals[frame_idx,:,n+1].flatten())        
                    dSRO_est_, SRO_est_, _, synced_block = DXCP_PhaT_Estimators[n].process(np.stack((synced_block_ref, InputBuffers[n].read()), axis=1), acs=acs[frame_idx])
                    synced_blocks_level[node] = synced_block
                    SRO_est[frame_idx, n] = -SRO_est_
                    dSRO_est[frame_idx, n] = -dSRO_est_
                    if frame_idx > async_nodes_delays[n]+resamplerDelay:#resamplerDelay*(n+1):
                        # Save synced frame once progress allows compensation of delay
                        signals_synced[frame_idx-(async_nodes_delays[n]+resamplerDelay), :, n+1] = synced_block
            synced_blocks_prev_level = synced_blocks_level

    return signals_synced, SRO_est, dSRO_est


def run_simulation_parallel(nodes_levels, signals, acs, control_init_sro_est=False):

    def process_node_pair(q, ref_signal, signal, node_id, start_delay=0):
        '''
        SRO Estimation and block-wise synchronization via CL-DXCP-PhaT for signals of one node pair.
        This function is meant to be executed multiple times in parallel for each tree-level.
        Input: 
            q: output Queue
            ref_signal (n_frames, frame_len) numpy array of reference signal
            signal (n_frames, frame_len) numpy array of signal to be synchronised.
            node_id (string): id of non-reference node, passed along to ouput queue.
        '''
        Estimator = CL_DXCPPhaT(start_delay)
        n_frames = np.shape(ref_signal)[0]
        
        signal_synced = np.zeros_like(signal)
        SRO_est = np.zeros((n_frames,))
        dSRO_est = np.zeros((n_frames,))
        for frame_idx in range(n_frames):
            dSRO_est_, SRO_est_, _, synced_block = Estimator.process(np.stack((ref_signal[frame_idx,:], signal[frame_idx,:]), axis=1), acs=acs[frame_idx])
            SRO_est[frame_idx] = -SRO_est_
            dSRO_est[frame_idx] = -dSRO_est_
            signal_synced[frame_idx,:] = synced_block

        q.put({
            'node_id': node_id,
            'SRO_est': SRO_est,
            'dSRO_est': dSRO_est,
            'signal_synced': signal_synced
        })

    nodes_select = get_unique_node_list(nodes_levels)
    n_async_nodes = len(nodes_select)-1
    n_frames = np.shape(signals)[0]
    resamplerDelay = 2 # frames
    start_delay_per_level = 100 if control_init_sro_est else 0
    async_nodes_delays = get_async_node_delays(nodes_levels, delay_per_level=resamplerDelay)

    processed_nodes = [0 for _ in nodes_select[1:]]
    pbar = tqdm(total=len(processed_nodes))

    # Prepare results
    signals_synced = np.zeros_like(signals)
    signals_synced[:,:,0] = signals[:,:,0] # ref signal is already sync
    SRO_est = np.zeros((n_frames, n_async_nodes))
    dSRO_est = np.zeros((n_frames, n_async_nodes))

    for sid, level in enumerate(nodes_levels):
        procs = []
        q = Queue()
        got_results = {}
        # Start  processes...
        for branch in level:
            ref_node, ref_node_id = branch[0], nodes_select.index(branch[0])
            for nid, node in enumerate(branch[1:]):
                got_results[node] = False
                node_id = nodes_select.index(node)
                # Shift signal to accommodate for delayed ref signal
                delay = resamplerDelay*sid
                sig = np.zeros_like(signals[:,:,node_id])
                sig[delay:, :] = signals[:-delay, :, node_id] if delay > 0 else signals[:,:, node_id]
                # Start
                procs.append(Process(target=process_node_pair, args=(q, signals_synced[:,:,ref_node_id], sig, node, start_delay_per_level*async_nodes_delays[node_id-1])))
                procs[-1].start()
        # Collect results
        while True:
            res = q.get()
            node_idx = nodes_select.index(res['node_id'])
            got_results[res['node_id']] = True
            processed_nodes[node_idx-1] = 1 #global tracker
            pbar.update(1)
            signals_synced[:,:, node_idx] = res['signal_synced']
            SRO_est[:, node_idx-1] = res['SRO_est']
            dSRO_est[:, node_idx-1] = res['dSRO_est']
            # Check if finished
            all_fin = True
            for n in got_results.keys():
                if got_results[n] == False: 
                    all_fin = False
                    break
            if all_fin == True: break

    pbar.close()

    # Compensate delays in synchronized signals to allow SINR computation
    async_nodes_delays = get_async_node_delays(nodes_levels, resamplerDelay)
    for n in range(n_async_nodes):
        d = async_nodes_delays[n]+2 #+2 to transform input- to output-delay
        signals_synced[:-d,:, n+1] = signals_synced[d:,:, n+1]
        signals_synced[-d:-1,:,n+1] = 0

    return signals_synced, SRO_est, dSRO_est


def is_unique_list(l):   
    for elem in l:
        if l.count(elem) > 1:
            return False
    return True

def verify_topology(nodes_levels, nodes_whitelist):
    '''
    Verifies topology description
    Input:
        nodes_levels (list): Topology description
        nodes_whitelist (list): List of allowed node_id strings. 
    Output:
        is_valid (bool): True if nodes_levels is a valid topology description, False otherwise
    '''
    
    if not isinstance(nodes_levels, list) or len(nodes_levels) == 0:
        print('Error: nodes_levels must be non-empty list!')
        return False

    prev_level_leaf_nodes = []
    leaf_nodes_encountered = [] # filled up on the go

    for lid, level in enumerate(nodes_levels):

        if not isinstance(level, list) or len(level) == 0:
            print('Error: Levels must be non-empty lists!')
            return False
        if lid == 0 and len(level) > 1:
            print('Error: First level must only contain one branch (tree-root)')
            return False

        level_all_nodes = []
        level_leaf_nodes = []
        for _, branch in enumerate(level):
            if not isinstance(branch, list) or len(branch) < 2:
                print('Error: Branches must be lists with at least two entries.')
                return False
            for nid, node_id in enumerate(branch):
                if not isinstance(node_id, str):
                    print('Error: Nodes must only be referenced with strings.')
                    return False
                if not node_id in nodes_whitelist:
                    print('Error: Referenced non-existing node. Available nodes:', nodes_whitelist)
                    return False
                level_all_nodes.append(node_id)
                if nid > 0 and node_id in leaf_nodes_encountered:
                    print('Error: Nodes can only be referenced as leafs once.')
                    return False              
                if nid == 0 and lid > 0:
                    if not node_id in prev_level_leaf_nodes:
                        print('Error: Branch root-nodes must appear as leafs in the preceding level.')
                        print(node_id, prev_level_leaf_nodes)
                        return False
                else:
                    level_leaf_nodes.append(node_id)

        leaf_nodes_encountered.extend(level_leaf_nodes)
        prev_level_leaf_nodes = level_leaf_nodes

        # nodes must only be referenced once within level
        if not is_unique_list(level_all_nodes):
            print('Error: Nodes can only be referenced once within a level.')
            return False

    return True



def get_unique_node_list(nodes_levels):
    '''
    Generates list of unique nodes selected within nested 'nodes_levels' list.
    This list is required for loading signals via AudioReader
    Input: 
        nodes_levels (list): Topology description
    Output:
        nodes (list): List of unique nodes
    '''
    nodes = []
    for level in nodes_levels:
        for branch in level:
            for node in branch:
                if node not in nodes:
                    nodes.append(node)
    return nodes

def get_async_node_delays(nodes_levels, delay_per_level=2):
    '''
    Generates list of required input delays for each async node based on topology.
    Each async node delays it's own microphone signal according to the level it is in to accomodate
    the dalay with which the reference signal is received.
    Input:
        nodes_levels (list): topology description
        dalay_per_level (int): delay (in frames) that is added in each level. governed by OnlineResampler.
    Output:
        async_node_delays (list): list of required frame delays. The indices correspond to those in the unique
                                    node list that the provided topology is associated with, shifted by -1, since
                                    no delay for the synchronous root-node is included. 
    '''
    nodes_select = get_unique_node_list(nodes_levels)
    async_node_delays = [0 for _ in nodes_select] #init

    for lid, level in enumerate(nodes_levels):
        delay = lid*delay_per_level
        for branch in level:
            for nid, node in enumerate(branch):
                if nid == 0: continue
                async_node_delays[nodes_select.index(node)] = delay

    return async_node_delays[1:]


def get_oracle_acs(example, sig_len, len_dxcpphat_est, delay=12.8):
    '''
    Compute simple oracle acoustic coherence state (ACS) based on scene diary. Consider source activity and scene stationarity.
    ACS is 1 when at least one source is active and scene (source composition) is unchanged for a certain time, 0 else.
    Input:
        example: The database example
        sig_len: Signal length in seconds
        len_dxcpphat_est: Length of DXCP-PhaT estimate time-series (resulting from block-wise processing) 
        delay: Time in seconds that has to pass after changes in scene for it to be considered stationary.
    Output:
        oracle_acs: Binary oracle ACS time-series
    '''
    dxcpphat_time_axis = np.linspace(0, sig_len, len_dxcpphat_est)
    delay = 12.8 # ~DXCP-PhaT Time Constant
    scene_changes = []
    scene_activity_mask = np.zeros_like(dxcpphat_time_axis)
    scene_stationary_mask = np.zeros_like(dxcpphat_time_axis)

    # Build scene acticity mask and collect scene change time points
    for scene in example['scene_diary']:
        scene_changes.append(scene['onset'])
        scene_changes.append(scene['offset'])
        scene_activity_mask[(dxcpphat_time_axis > scene['onset']) & (dxcpphat_time_axis < scene['offset'])] = 1

    # Build scene stationary mask by looking at changing time points (and consider delay)
    scene_changes.sort()
    for idx, t in enumerate(scene_changes):
        t_next = scene_changes[idx+1] if idx < len(scene_changes)-1 else np.max(dxcpphat_time_axis)
        scene_stationary_mask[(dxcpphat_time_axis >= t+delay) & (dxcpphat_time_axis < t_next)] = 1

    # Build oracle acs mask by AND-linking activity- with stationary mask
    oracle_acs = np.zeros_like(dxcpphat_time_axis)
    oracle_acs[(scene_activity_mask == 1) & (scene_stationary_mask == 1)] = 1

    return oracle_acs



def gen_offset(sig, offset, fft_size):
    '''
    Generate/compensate STO
    '''
    k = np.fft.fftshift(np.arange(-fft_size / 2, fft_size / 2))
    block_len = int(fft_size // 2)
    sig_offset = np.zeros_like(sig)
    block_idx = 0
    len_rest = 0

    integer_offset = np.round(offset)
    rest_offset = integer_offset - offset

    # The STFT uses a Hann window with 50% overlap as analysis window
    win = hann(block_len, sym=False)

    while True:

        block_start = int(block_idx * block_len / 2 + integer_offset)
        if block_start < 0:
            if block_start + block_len < 0:
                block = np.zeros(block_len)
            else:
                block = np.pad(sig[0:block_start + block_len],
                               (block_len - (block_start + block_len), 0),
                               'constant')
        else:
            if (block_start + block_len) > sig.size:
                block = np.zeros(block_len)
                block[:sig[block_start:].size] = sig[block_start:]
                len_rest = sig[block_start:].size
            else:
                block = sig[block_start:block_start + block_len]

        sig_fft = np.fft.fft(win * block, fft_size)
        sig_fft *= np.exp(-1j * 2 * np.pi * k / fft_size * rest_offset)

        block_start = int(block_idx * block_len / 2)

        if block_start+block_len > sig_offset.size:
            n_pad = block_start + block_len - sig_offset.size
            sig_offset = np.pad(sig_offset, (0, n_pad), 'constant')
            
        sig_offset[block_start:block_start + block_len] += \
            np.real(np.fft.ifft(sig_fft))[:block_len]
        block_end = int(block_idx * block_len / 2 - integer_offset) + block_len

        if block_end > sig.size :
            return sig_offset[:block_start + len_rest]

        block_idx += 1


############ PLOTS

def plot_scene_diary(scene_diary, max_len):

    figure = plt.figure(figsize=(12, 2))
    scenes = []
    for scene in sorted(scene_diary, key=lambda x: x['onset']):
        scene_id = scene['scene']
        onset = scene['onset']
        if onset > max_len:
            break
        if scene_id not in scenes:
            scenes.append(scene_id)

    rows = {scene: i for i, scene in enumerate(scenes)}
    num_rows = len(rows)
    for scene in sorted(scene_diary, key=lambda x: x['onset']):
        scene_id = scene['scene']
        onset = scene['onset']
        offset = np.minimum(scene['offset'], max_len)
        if onset > max_len:
            break
        ymin = (rows[scene_id]+.1)/num_rows
        ymax = (rows[scene_id]+.9)/num_rows
        plt.axvspan(onset, offset, ymin, ymax)

    plt.title('Fig.2 (b) Scene diary')
    plt.xlabel('Time [s]')
    plt.xlim(0, max_len)
    plt.ylabel('Scene')
    plt.yticks((np.arange(num_rows) + .5) / num_rows , scenes)
    plt.grid()

    #plt.savefig('scene_diary.svg')

def plot_positions_and_topology(example, room_model, max_len, nodes_levels=None):

    room_mesh = mesh.Mesh.from_file(room_model)
    figure = plt.figure(figsize=(8, 8))
    figure.suptitle('Fig.1 Geometry')
    ax = mplot3d.Axes3D(figure)
    ax.view_init(azim=-90, elev=90)
    poly = mplot3d.art3d.Poly3DCollection(room_mesh.vectors)
    poly.set_alpha(0.4)
    poly.set_facecolor('lightgray')
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    scale = room_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.add_collection3d(poly)
    
    for node_id, params in example['nodes'].items():
        pos = params['position']['coordinates']
        ax.scatter(pos[0], pos[1], 5, s=50, color='r')
        ax.text(pos[0]+.05, pos[1]+.05, 5, node_id.split("_")[-1], fontsize=12, zorder=99, fontweight=800)


    src_positions = []
    for src in sorted(example['src_diary'], key=lambda x: x['onset']):
        pos_id = src['pos_id']
        onset = src['onset']
        if onset > max_len:
            break
        if src['pos_id'] not in src_positions:
            src_positions.append(src['pos_id'])
            pos = src['position']['coordinates']
            ax.scatter(pos[0], pos[1], 15, s=50, color='b')
            ax.text(pos[0] + .05, pos[1] + .05, 15, src['pos_id'], fontsize=12)


    # Plot topolgy: Graph edges
    if nodes_levels is not None:
        colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
        zorder=100
        opacity = 0.6
        for lid, level in enumerate(nodes_levels):
            zorder -= 1
            for bid, branch in enumerate(level):
                col = colors[lid%len(colors)]
                root_pos = example['nodes'][branch[0]]['position']['coordinates']
                for idx, node in enumerate(branch):
                    if idx == 0: continue
                    pos = example['nodes'][node]['position']['coordinates']
                    ax.plot([root_pos[0], pos[0]], [root_pos[1], pos[1]], color=col, linewidth=3, zorder=zorder, alpha=opacity)
                    # arrow-head: (ax.arrow() is buggy)
                    alpha = np.pi/4 # arrow-dash angle
                    z = 0.2 # arrow-dash len
                    pos2d = np.array(pos[:2])
                    root_pos2d = np.array(root_pos[:2])
                    R_top = np.array([[np.cos(alpha), -np.sin(alpha)], 
                            [np.sin(alpha), np.cos(alpha)]])
                    R_bottom = np.array([[np.cos(-alpha), -np.sin(-alpha)], 
                            [np.sin(-alpha), np.cos(-alpha)]])
                    diff = ((root_pos2d - pos2d)/np.linalg.norm(root_pos2d - pos2d))*z
                    dash1 = R_top.dot(diff)
                    dash2 = R_bottom.dot(diff)
                    ax.plot([pos[0], pos[0]+dash1[0]], [pos[1], pos[1]+dash1[1]], color=col, linewidth=3, zorder=zorder, alpha=opacity) 
                    ax.plot([pos[0], pos[0]+dash2[0]], [pos[1], pos[1]+dash2[1]], color=col, linewidth=3, zorder=zorder, alpha=opacity)

    #plt.savefig('1_Topo.svg')

def plot_pos_hist(src_diary, max_len):
    
    figure = plt.figure(figsize=(12.5, 2))
    src_positions = []
    sources = []
    for src in sorted(src_diary, key=lambda x: x['onset']):
        pos_id = src['pos_id']
        onset = src['onset']
        if onset > max_len:
            break
        if 'speaker_id' in src.keys():
            src_id = f'speaker_{src["speaker_id"]}'
        else:
            src_id = 'loudspeaker'
        if src['pos_id'] not in src_positions:
            src_positions.append(src['pos_id'])
        if src_id not in sources:
            sources.append(src_id)

    colors = list(mcolors.TABLEAU_COLORS.values())
    colors = {src: colors[i] for i, src in enumerate(sources)}
    rows = {src: i for i, src in enumerate(src_positions)}
    num_rows = len(src_positions)
    handles = []
    labels = []
    for src in sorted(src_diary, key=lambda x: x['onset']):
        pos_id = src['pos_id']
        onset = src['onset']
        offset = np.minimum(src['offset'], max_len)
        if onset > max_len:
            break
        if 'speaker_id' in src.keys():
            src_id = f'speaker_{src["speaker_id"]}'
        else:
            src_id = 'loudspeaker'
        ymin = (rows[pos_id]+.1)/num_rows
        ymax = (rows[pos_id]+.9)/num_rows
        handle = plt.axvspan(onset, offset, ymin, ymax, label='source', facecolor=colors[src_id])
        if src_id not in labels:
            handles.append(handle)
            labels.append(src_id)
    plt.title('Fig.2 (a) Source activity')
    plt.xlabel('Time [s]')
    plt.xlim(0, max_len)
    plt.ylabel('Position ID')
    plt.yticks((np.arange(num_rows) + .5) / num_rows , src_positions)
    plt.grid()
    #plt.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(sources))
    plt.legend(handles, labels,loc='upper center', ncol=len(sources))

    #plt.savefig('source_hist.svg')