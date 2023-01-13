import math
import numpy as np
from scipy import signal
from online_resampler import OnlineResampler
from delay_buffer import DelayBuffer

class SRO_Est_Controller:
    def __init__(self):
        pass

    
class CL_DXCPPhaT():
    '''
    Proposed controller from "CONTROL ARCHITECTURE OF THE DOUBLE-CROSS-CORRELATION PROCESSOR
    FOR SAMPLING-RATE-OFFSET ESTIMATION IN ACOUSTIC SENSOR NETWORKS", 2021. A. Chinaev, S. Wienand, G. Enzner
    '''
    def __init__(self, start_delay=0):
        '''
        start_delay: number of frames to freeze IMC controllers output (to 0) after initialisation.
        this delay should be proportional to the position of the node within the tree/topology to avoid
        resampling based on unreliable results.
        '''
        self.DXCPPhaT = DXCPPhaT()
        self.Resampler = OnlineResampler()
        self.zjBuffer = DelayBuffer((self.DXCPPhaT.FrameSize_input, 2+1)) # Resampler adds 2 frames delay
        self.dSRO_est = np.zeros(3) # buffered, internal
        self.SRO_est = np.zeros(3) # buffered, internal
        self.dSRO_est_curr = 0 # current, out
        self.dSRO_est_curr_raw = 0 # raw DXCPP output, not affected by ACS
        self.SRO_est_curr = 0 #current, out
        self.SRO_est_op = 0 # in MATLAB "SROppm_init"
        self.STO_est = 0
        # IMC Controller (Variant 'PIT1'/Tf=8)
        self.K_Nom = [0, 0.0251941968627353, -0.0249422548941180]
        self.K_Denom = [1, -1.96825464010938, 0.968254640109407]
        self.maxDelta = 10**6 / (self.DXCPPhaT.AccumTime_B_NumFr*self.DXCPPhaT.FFTshift_dxcp)
        self.L_hold = 100
        self.forwardControl = False
        self.forwardControlStepCount = 0
        self.start_delay = start_delay 
        self.ell = 0


    def process(self, x_12_ell, acs=1):
        '''
        Resample z_i, delay z_j, get and return DXCPP Results together with synced frame.
        Note: The returned synced z_i frame will be the frame from 2 iterations earlier, 
        resampled based on the previously estimated SRO (not the updated SRO estimate resulting from this call) 
        
        acs: ACS value, 0 or 1, for controlling DXCPP output (dSRO). Temporary solution.
        '''
        # Resample z_i based on latest SRO estimate
        z_i = self.Resampler.process(x_12_ell[:, 1].flatten(), -self.SRO_est_curr)
        x_12_ell[:, 1] = z_i    
        # Delay z_j to accomodate delay of z_i caused by buffer
        self.zjBuffer.write(x_12_ell[:, 0].flatten())
        x_12_ell[:, 0] = self.zjBuffer.read()  
        # Estimate residual SRO (Buffer filled from left)
        res = self.DXCPPhaT.process_data(x_12_ell)
        self.dSRO_est_curr_raw = res['SROppm_est_out']
        self.dSRO_est_curr = res['SROppm_est_out'] if acs == 1 else 0 #force dSRO=0 in bad conditions
        if self.ell <= self.start_delay:
            self.dSRO_est_curr = 0
        self.dSRO_est[1:] = self.dSRO_est[:-1]
        self.dSRO_est[0] = self.dSRO_est_curr
        self.SRO_est[1:] = self.SRO_est[:-1]
        self.SRO_est[0] = np.dot(self.K_Nom, self.dSRO_est) - np.dot(self.K_Denom[1:], self.SRO_est[1:]) 
        self.SRO_est_curr = self.SRO_est[0] + self.SRO_est_op

        self.ell += 1
        
        return self.dSRO_est_curr_raw, self.SRO_est_curr, self.Resampler.shift, z_i




class CL_DXCPPhaT_Simple():

    '''
    Simplified controller design just for testing.
    '''

    def __init__(self):
        self.DXCPPhaT = DXCPPhaT()
        self.Resampler = OnlineResampler()
        self.zjBuffer = DelayBuffer((self.DXCPPhaT.FrameSize_input, 2+1)) # Resampler adds 2 frames delay
        self.dSRO_est = 0
        self.SRO_est = 0
        self.STO_est = 0

    def process(self, x_12_ell, acs=1):
        '''
        Resample z_i, delay z_j, get and return DXCPP Results together with synced frame.
        Note: The returned synced z_i frame will be the frame from 2 iterations earlier, 
        resampled based on the previously estimated SRO (not the updated SRO estimate resulting from this call) 
        '''
        # Resample z_i based on latest SRO estimate
        z_i = self.Resampler.process(x_12_ell[:, 1].flatten(), -self.SRO_est)
        x_12_ell[:, 1] = z_i    
        # Delay z_j to accomodate delay of z_i caused by buffer
        self.zjBuffer.write(x_12_ell[:, 0].flatten())
        x_12_ell[:, 0] = self.zjBuffer.read()  
        # Estimate...
        res = self.DXCPPhaT.process_data(x_12_ell)
        self.dSRO_est = (1-acs)*self.dSRO_est + acs*res['SROppm_est_out']
        # Limit state
        self.SRO_est = self.SRO_est + 0.01*self.dSRO_est
        if self.SRO_est < -100:
            self.SRO_est = -100
        elif self.SRO_est > 100:
            self.SRO_est = 100
        self.STO_est = res['STOsmp_est_out']

        return self.dSRO_est, self.SRO_est, self.Resampler.shift, z_i


class DXCPPhaT():

    defaultParams = {
        'RefSampRate_fs_Hz': 16000,      # reference sampling rate
        'FrameSize_input': 2048,         # frame size (power of 2) of input data
        'FFTshift_dxcp': 2**11,          # frame shift of DXCP-PhaT (power of 2 & >= FrameSize_input)
        'FFTsize_dxcp': 2**13,           # FFT size of DXCP-PhaT (power of 2 & >= FFTshift_dxcp)
        'AccumTime_B_sec': 5,            # accumulation time in sec (usually 5s as in DXCP)
        'ResetPeriod_sec': 30,           # resetting period of DXCP-PhaT in sec. Default: 30 (>=2*AccumTime_B_sec)
        'SmoConst_CSDPhaT_alpha': .53,   # smoothing constant for GCSD1 averaging (DXCP-PhaT)
        'SmoConst_CSDPhaT_alpha2': .99,  # smoothing constant for GCSD2 averaging (DXCP-PhaT)
        'SmoConst_SSOest_alpha': .99,    # smoothing constant of SRO-comp. CCF-1 used to estimate d12 (DXCP-PhaT) [.995 for big mic-dist]
        'AddContWait_NumFr': 0,          # additional waiting for container filling (>InvShiftFactor-1)
        'SettlingCSD2avg_NumFr': 4,      # settling time of CSD-2 averaging (SettlingCSD2avg_NumFr < Cont_NumFr-AddContWait_NumFr)
        'X_12_abs_min': 1e-12,           # minimum value of |X1*conj(X2)| to avoid devision by 0 in GCC-PhaT
        'SROmax_abs_ppm': 1000,          # maximum absolute SRO value possible to estimate (-> Lambda)
        'p_upsmpFac': 4,
        'Flag_DisplayResults': 1 
    }

    def __init__(
        self, 
        RefSampRate_fs_Hz = defaultParams['RefSampRate_fs_Hz'],      
        FrameSize_input = defaultParams['FrameSize_input'],        
        FFTshift_dxcp = defaultParams['FFTshift_dxcp'],         
        FFTsize_dxcp = defaultParams['FFTsize_dxcp'],           
        AccumTime_B_sec = defaultParams['AccumTime_B_sec'],            
        ResetPeriod_sec = defaultParams['ResetPeriod_sec'],           
        SmoConst_CSDPhaT_alpha = defaultParams['SmoConst_CSDPhaT_alpha'],   
        SmoConst_CSDPhaT_alpha2 = defaultParams['SmoConst_CSDPhaT_alpha2'],  
        SmoConst_SSOest_alpha = defaultParams['SmoConst_SSOest_alpha'],    
        AddContWait_NumFr = defaultParams['AddContWait_NumFr'],          
        SettlingCSD2avg_NumFr = defaultParams['SettlingCSD2avg_NumFr'],      
        X_12_abs_min = defaultParams['X_12_abs_min'],           
        SROmax_abs_ppm = defaultParams['SROmax_abs_ppm'],     
        p_upsmpFac = defaultParams['p_upsmpFac'],     
        Flag_DisplayResults = defaultParams['Flag_DisplayResults']         
    ):

        # General parameters (config)
        self.RefSampRate_fs_Hz = RefSampRate_fs_Hz      
        self.FrameSize_input = FrameSize_input
        self.FFTshift_dxcp = FFTshift_dxcp        
        self.FFTsize_dxcp = FFTsize_dxcp         
        self.AccumTime_B_sec = AccumTime_B_sec            
        self.ResetPeriod_sec = ResetPeriod_sec           
        self.SmoConst_CSDPhaT_alpha = SmoConst_CSDPhaT_alpha   
        self.SmoConst_CSDPhaT_alpha2 = SmoConst_CSDPhaT_alpha2  
        self.SmoConst_SSOest_alpha = SmoConst_SSOest_alpha    
        self.AddContWait_NumFr = AddContWait_NumFr
        self.SettlingCSD2avg_NumFr = SettlingCSD2avg_NumFr
        self.X_12_abs_min = X_12_abs_min
        self.SROmax_abs_ppm = SROmax_abs_ppm
        self.p_upsmpFac = p_upsmpFac
        self.Flag_DisplayResults = Flag_DisplayResults

        # Implicit parameters
        self.LowFreq_InpSig_fl_Hz = .01 * self.RefSampRate_fs_Hz / 2
        self.UppFreq_InpSig_fu_Hz = .95 * self.RefSampRate_fs_Hz / 2
        self.RateDXCPPhaT_Hz = self.RefSampRate_fs_Hz / self.FFTshift_dxcp
        self.AccumTime_B_NumFr = int(self.AccumTime_B_sec // (1 / self.RateDXCPPhaT_Hz))
        self.B_smpls = self.AccumTime_B_NumFr * self.FFTshift_dxcp
        self.Upsilon = int(self.FFTsize_dxcp / 2 - 1)
        self.Lambda = int(((self.B_smpls * self.SROmax_abs_ppm) // 1e6) + 1)
        self.Cont_NumFr = self.AccumTime_B_NumFr + 1
        self.InvShiftFactor_NumFr = int(self.FFTsize_dxcp / self.FFTshift_dxcp)
        self.ResetPeriod_NumFr = int(self.ResetPeriod_sec // (1 / self.RateDXCPPhaT_Hz))
        self.FFT_Nyq = int(self.FFTsize_dxcp / 2 + 1)
        self.FreqResol = self.RefSampRate_fs_Hz / self.FFTsize_dxcp
        self.LowFreq_InpSig_fl_bin = int(self.LowFreq_InpSig_fl_Hz // self.FreqResol)
        self.UppFreq_InpSig_fu_bin = int(self.UppFreq_InpSig_fu_Hz // self.FreqResol)
        self.NyqDist_fu_bin = self.FFT_Nyq - self.UppFreq_InpSig_fu_bin    

        # STATE: SCALARS
        self.SROppm_est_ell = 0 # current SRO estimate
        self.SSOsmp_est_ell = 0 # current SSO estimate

        # STATE: MULTIDIM
        self.GCSD_PhaT_avg = np.zeros((self.FFTsize_dxcp, 1), dtype=complex)                 # Averaged CSD with Phase Transform        
        self.GCSD_PhaT_avg_Cont = np.zeros((self.FFTsize_dxcp, self.Cont_NumFr), dtype=complex)   # Container with past GCSD_PhaT_avg values        
        self.GCSD2_avg = np.zeros((self.FFTsize_dxcp, 1), dtype=complex)                     # averaged CSD-2        
        self.GCCF1_smShftAvg = np.zeros((2 * self.Upsilon + 1, 1), dtype=float)              # smoothed shifted first CCF      
        self.GCCF2_avg_ell_big = np.zeros(17, dtype=float)                              # time-domain cff2 for debugging  
        self.InputBuffer = np.zeros((self.FFTsize_dxcp, 2), dtype=float)                     # input buffer of DXCP-PhaT-CL

        # STATE: COUNTERS AND FLAGS      
        self.flag_initiated = False         # flag for initialized recursive averaging.  
        self.ell_execDXCPPhaT = 1           # counter for filling of the input buffer before DXCP-PhaT can be executed        
        self.ell = 1                        # counter within signal     


    def process_data(self, x_12_ell, tdoa=0): # process data of the current input frame

        # fill the internal buffer of DXCP-PhaT-CL (from right to left)
        self.InputBuffer[np.arange(0, self.FFTsize_dxcp - self.FrameSize_input), :] = self.InputBuffer[np.arange(self.FrameSize_input, self.FFTsize_dxcp), :]
        self.InputBuffer[np.arange(self.FFTsize_dxcp - self.FrameSize_input, self.FFTsize_dxcp), :] = x_12_ell

        # execute DXCP-PhaT-CL when enough input frames are collected
        if self.ell_execDXCPPhaT == int(self.FFTshift_dxcp / self.FrameSize_input):
            self.ell_execDXCPPhaT = 0
            self._stateupdate(tdoa)

        # update counter for filling of the input buffer
        self.ell_execDXCPPhaT += 1

        # compose output
        OutputDXCPPhaTcl = {}
        OutputDXCPPhaTcl['SROppm_est_out'] = self.SROppm_est_ell
        OutputDXCPPhaTcl['STOsmp_est_out'] = self.SSOsmp_est_ell
        #OutputDXCPPhaTcl['GCCF2_avg'] = self.GCCF2_avg_ell_big
        return OutputDXCPPhaTcl


    def _stateupdate(self, tdoa=0): 

        # (1) Windowing to the current frames acc. to eq. (5) in [1]
        analWin = signal.blackman(self.FFTsize_dxcp, sym=False)
        x_12_win = self.InputBuffer * np.vstack((analWin, analWin)).transpose()
        
        # (2) Estimate generalized (normalized) GCSD with Phase Transform (GCSD-PhaT) via recursive averaging
        X_12 = np.fft.fft(x_12_win, self.FFTsize_dxcp, 0)
        X_12_act = X_12[:, 0] * np.conj(X_12[:, 1])
        X_12_act_abs = abs(X_12_act)
        X_12_act_abs[X_12_act_abs < self.X_12_abs_min] = self.X_12_abs_min  # avoid division by 0
        GCSD_PhaT_act = X_12_act / X_12_act_abs
        if self.flag_initiated == False:
            self.GCSD_PhaT_avg = GCSD_PhaT_act
        else:
            self.GCSD_PhaT_avg = self.SmoConst_CSDPhaT_alpha * self.GCSD_PhaT_avg + (1 - self.SmoConst_CSDPhaT_alpha) * GCSD_PhaT_act
        # (3) Fill the DXCP-container with self.Cont_NumFr number of past GCSD_PhaT_avg
        self.GCSD_PhaT_avg_Cont[:, np.arange(self.Cont_NumFr - 1)] = self.GCSD_PhaT_avg_Cont[:, 1:]
        self.GCSD_PhaT_avg_Cont[:, (self.Cont_NumFr - 1)] = self.GCSD_PhaT_avg
    
        # (4) As soon as DXCP-container is filled with resampled data, calculate the second GCSD based
        # on last and first vectors of DXCP-container and perform time averaging
        if self.ell >= self.Cont_NumFr + (self.InvShiftFactor_NumFr - 1) + self.AddContWait_NumFr:
            # Estimate second GCSD via recursive averaging
            GCSD2_act = self.GCSD_PhaT_avg_Cont[:, -1] * np.conj(self.GCSD_PhaT_avg_Cont[:, 0])
            if self.flag_initiated == False:
                self.GCSD2_avg[:, 0] = GCSD2_act
            else:
                self.GCSD2_avg[:, 0] = self.SmoConst_CSDPhaT_alpha2 * self.GCSD2_avg[:, 0] + (1 - self.SmoConst_CSDPhaT_alpha2) * GCSD2_act
            # remove non-coherent components
            GCSD2_avg_ifft = self.GCSD2_avg
            # set lower frequency bins (w.o. coherent components) to 0
            GCSD2_avg_ifft[np.arange(self.LowFreq_InpSig_fl_bin), 0] = 0
            GCSD2_avg_ifft[np.arange(self.FFTsize_dxcp - self.LowFreq_InpSig_fl_bin + 1, self.FFTsize_dxcp), 0] = 0
            # set upper frequency bins (w.o. coherent components) to 0
            GCSD2_avg_ifft[np.arange(self.FFT_Nyq - self.NyqDist_fu_bin - 1, self.FFT_Nyq + self.NyqDist_fu_bin), 0] = 0
            # Calculate averaged CCF-2 in time domain
            GCCF2_avg_ell_big = np.fft.fftshift(np.real(np.fft.ifft(GCSD2_avg_ifft, n=self.FFTsize_dxcp, axis=0)))
            idx = np.arange(self.FFT_Nyq - self.Lambda - 1, self.FFT_Nyq + self.Lambda)
            GCCF2avg_ell = GCCF2_avg_ell_big[idx, 0]
            # Log in state for debugging
            self.GCCF2_avg_ell_big = GCCF2avg_ell[72:89] # only middle-part
    
  
        # (5) Parabolic interpolation (13) with (14) with maximum search as in [1]
        # and calculation of the remaining current SRO estimate sim. to (15) in [1]
        # As soon as GCSD2_avg is smoothed enough in every reseting section
        if self.ell >= self.Cont_NumFr + (self.InvShiftFactor_NumFr - 1) + self.AddContWait_NumFr + self.SettlingCSD2avg_NumFr:
            
            # p-fold upsampling
            upsmpWindow = signal.get_window(('kaiser', 5.0), Nx=2*self.Lambda+1, fftbins=False)
            GCCF2avg_ell_upsmp = signal.resample(GCCF2avg_ell, num=(2*self.Lambda+1)*self.p_upsmpFac, window=upsmpWindow)
            lambda_vec_upsmp = np.arange(-self.Lambda, self.Lambda+1, 1/self.p_upsmpFac) # upsampled lambda-scale
            # maximum search
            idx_max = GCCF2avg_ell_upsmp.argmax(0)
            #if (idx_max == 0) or (idx_max == 2 * self.Lambda):
            if (idx_max == 0) or (idx_max == len(lambda_vec_upsmp)-1):
                DelATSest_ell_frac = 0
            else:
                # set supporting points for search of real-valued maximum
                # NOTICE: This needs fixing! Out-of-range error occurs when maximum itself is at the border
                #         limit idx_max +2 and -1 for these edge-cases
                sup_pnts = GCCF2avg_ell_upsmp[np.arange(idx_max - 1, idx_max + 2)]  # supporting points y(x) for x={-1,0,1}
                # calculate fractional of the maximum via x_max=-b/2/a for y(x) = a*x^2 + b*x + c
                DelATSest_ell_frac = (sup_pnts[2, ] - sup_pnts[0, ]) / 2 / ( 2 * sup_pnts[1, ] - sup_pnts[2, ] - sup_pnts[0, ])

            # [old] w/o resmp: DelATSest_ell = lambda_vec_upsmp[idx_max] - self.Lambda + DelATSest_ell_frac  # resulting real-valued x_max
            DelATSest_ell = lambda_vec_upsmp[idx_max] + DelATSest_ell_frac/self.p_upsmpFac  # resulting real-valued x_max
            self.SROppm_est_ell = DelATSest_ell / self.B_smpls * 10 ** 6


        # (6) STO-estimation after removing of SRO-induced time offset in CCF-1
        if self.ell >= self.Cont_NumFr + (self.InvShiftFactor_NumFr - 1) + self.AddContWait_NumFr + self.SettlingCSD2avg_NumFr:
            
            # a) phase shifting of GCSD-1 to remove SRO-induced time offset
            timeOffset_forShift = self.SROppm_est_ell * 10 ** (-6) * self.FFTshift_dxcp * (self.ell - 1)
            idx = np.arange(self.FFTsize_dxcp).transpose()
            expTerm = np.power(math.e, 1j * 2 * math.pi / self.FFTsize_dxcp * timeOffset_forShift * idx)
            GCSD1_smShft = self.GCSD_PhaT_avg * expTerm
            # b) remove components w.o. coherent components
            GCSD1_smShft_ifft = GCSD1_smShft
            # set lower frequency bins (w.o. coherent components) to 0
            GCSD1_smShft_ifft[np.arange(self.LowFreq_InpSig_fl_bin),] = 0
            GCSD1_smShft_ifft[np.arange(self.FFTsize_dxcp - self.LowFreq_InpSig_fl_bin + 1, self.FFTsize_dxcp),] = 0
            # set upper frequency bins (w.o. coherent components) to 0
            GCSD1_smShft_ifft[np.arange(self.FFT_Nyq - self.NyqDist_fu_bin - 1, self.FFT_Nyq + self.NyqDist_fu_bin),] = 0
            # c) go into the time domain via calculation of shifted GCC-1
            GCCF1_sroComp_big = np.fft.fftshift(np.real(np.fft.ifft(GCSD1_smShft_ifft, n=self.FFTsize_dxcp)))
            GCCF1_sroComp = GCCF1_sroComp_big[np.arange(self.FFT_Nyq - self.Upsilon - 1, self.FFT_Nyq + self.Upsilon),]
            # d) averaging over time and zero-phase filtering within the frame (if necessary)
            if self.flag_initiated == False:
                self.GCCF1_smShftAvg[:, 0] = GCCF1_sroComp
            else:
                self.GCCF1_smShftAvg[:, 0] = self.SmoConst_SSOest_alpha * self.GCCF1_smShftAvg[:, 0] + (1 - self.SmoConst_SSOest_alpha) * GCCF1_sroComp

            GCCF1_smShftAvgAbs = np.abs(self.GCCF1_smShftAvg)
            # e) Maximum search over averaged filtered shifted GCC-1 (with real-valued SSO estimates)
            idx_max = GCCF1_smShftAvgAbs.argmax(0)
            if (idx_max == 0) or (idx_max == 2 * self.Upsilon):
                SSOsmp_est_ell_frac = 0
                self.SSOsmp_est_ell = idx_max[0] - self.Upsilon + SSOsmp_est_ell_frac  # resulting real-valued x_max
            else:
                # set supporting points for search of real-valued maximum
                sup_pnts = GCCF1_smShftAvgAbs[
                    np.arange(idx_max - 1, idx_max + 2)]  # supporting points y(x) for x={-1,0,1}
                # calculate fractional of the maximum via x_max=-b/2/a for y(x) = a*x^2 + b*x + c
                SSOsmp_est_ell_frac = (sup_pnts[2, ] - sup_pnts[0, ]) / 2 / (2 * sup_pnts[1, ] - sup_pnts[2, ] - sup_pnts[0, ])
                self.SSOsmp_est_ell = idx_max[0] - self.Upsilon + SSOsmp_est_ell_frac[0]  # resulting real-valued x_max
                # correct for TDOA
                self.SSOsmp_est_ell = self.SSOsmp_est_ell + tdoa*self.RefSampRate_fs_Hz # !verify sign


        # (8) Update counter of DXCP frames within signal and flag initiation
        self.ell += 1
        if (self.flag_initiated == False):
            self.flag_initiated = True

