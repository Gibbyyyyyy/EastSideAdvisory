import numpy as np
from scipy.signal import find_peaks, lfilter, correlate


#Window class that stores all the information on the selected FT window
class Window:
    
    #Initalization
    def __init__(self, size):
        self.size = size
        self.edge1 = 1
        self.edge2 = size
        self.mid = self.edge1 + (size // 2)

    #Shifting the window to go through the whole signal
    def shift(self,amt):
        self.edge1 = self.edge1 + amt
        self.edge2 = self.edge1 + self.size - 1
        self.mid = self.edge1 + (self.size // 2)
    
    #Placing the window
    def place(self,loc):
        self.edge1 = loc
        self.edge2 = self.edge1 + self.size - 1
    
    def reset(self):
        self.edge1 = 1
        self.edge = self.size
        self.mid = self.edge1 + (self.size // 2)


#Detecting where in the landed window, the wavefront is
def wavefrontSelection(XWIN, FSIGHZ, window):

    #Error check
    if XWIN is None or len(XWIN) == 0:
        return 0
    
    #Max value in the window
    windowMax = np.max(np.abs(XWIN))

    #Setting the amplitude threshold to 10% of the max value found above
    maxminThreshold = .1 *windowMax

    #Getting the maximum and minimum values of the landed window
    maxima = np.flipud(getMaxima(XWIN))
    minima = np.flipud(getMinima(XWIN))

    #Getting rid of the max/min values that are not above the amplitude threshold
    maxima = maxima[np.abs(maxima[:,0]) >= maxminThreshold]
    minima = minima[np.abs(minima[:,0]) >= maxminThreshold]

    #Putting max and min into one array
    allExtrema = np.vstack((maxima,minima))

    #Error Check
    if allExtrema.size == 0:
        return 0
    
    #Sorting the array to make sure that the max or min with the lowest indices (the furthest left in the window) is first
    allExtrema = allExtrema[allExtrema[:,1].argsort()]

    #Making sure there is more than one extrema in the array. Can't take a ratio or find a zero crossing between two extrema with 
    # only one data point
    if allExtrema.shape[0] > 1:

        #Hardcoded sampling rate and then getting the samples per half cycle. Ideally, this is the number of samples that would be 
        # between a max and a min.
        samplingRate = 3.5e6
        samplesPerCyc = samplingRate / FSIGHZ
        samplesPerHalfCyc = samplesPerCyc / 2

        #Adding a buffer to the samples between the two extrema
        sampleBuffer = 0.5 * samplesPerHalfCyc
        lowerSampleBound = samplesPerHalfCyc - sampleBuffer
        upperSampleBound = samplesPerHalfCyc + sampleBuffer

        #Checking to see if two extrema are within the limits and getting rid of them if they are not. 
        # This eliminates the low chance that the window landed in a very high or low frequency noise part of the signal
        deltas = np.diff(allExtrema[:, 1])
        validDelta = (deltas >= lowerSampleBound) & (deltas <= upperSampleBound)

        keepIdx = np.concatenate(([True], validDelta))
        allExtrema = allExtrema[keepIdx]

        #Once again, checking to see if there is still more than 1 min/max
        if allExtrema.shape[0] > 1:
            
            #The number of ratios between the mins and maxes that there will be
            numRatios = allExtrema.shape[0] - 1
            ratios = np.zeros(numRatios)

            #Getting the ratios between of mins and maxes to determine which ratio is the largest. This tells us where the biggest "jump"
            #in the window is.
            for k in range(numRatios):
                a = abs(allExtrema[k, 0])
                b = abs(allExtrema[k + 1, 0])
                #check check to make sure there is no divide by zero
                if min(a, b) != 0:
                    ratios[k] = max(a, b) / min(a, b)
                else:
                    ratios[k] = 0

            #Getting the largest ratio
            choose = np.argmax(ratios)

            #Getting where the indices of the two extrema are located
            location1 = int(allExtrema[choose, 1])
            location2 = int(allExtrema[choose + 1, 1])

            #Getting the zero crossings that are in the window
            crossings = findZeroCrossings(XWIN)

            #Choosing the zero crossing that is between the two chosen extrema
            zeroCrossing = crossings[(crossings > location1) & (crossings < location2)]

            #quick error check
            if len(zeroCrossing) > 0:
                return int(zeroCrossing[0] + window.edge1)
            else:
                return 0
        else:
            return 0

    #If there was only 1 extrema in the array    
    elif allExtrema.shape[0] == 1:

        #same process as above but taking the zero crossing that is two the right of the one extrema
        location = int(allExtrema[0,1])
        crossings = findZeroCrossings(XWIN)
        zeroCrossing = crossings[crossings > location]

        if len(zeroCrossing) > 0:
            return int(zeroCrossing[0] + window.edge1)
        else:
            return 0
    return 0

#Getting the maximums of the given signal
def getMaxima(vector):

    #error check
    if vector is None or len(vector) == 0:
        return np.array([0,2])
    
    #simply gets the peaks of the passed in signal and store in the array
    peaks,_ = find_peaks(vector)
    maxs = vector[peaks]

    return np.column_stack((maxs,peaks))

#Getting the minimums of the given signal
def getMinima(vector):

    #error check
    if vector is None or len(vector) == 0:
        return np.array([0,2])
    
    #simply, gets the peaks of the passed in signal and store in array
    peaks, _ = find_peaks(-vector)
    mins = -vector[peaks]

    return np.column_stack((mins,peaks))

#Gets the zero crossings of the passed in signal
def findZeroCrossings(x):
    x = np.array(x)
    
    #returns the location where the multiplication of two elements are negative (indicating the data crossed zero)
    return np.where(x[:-1] * x[1:] < 0)[0]

def dft(signal, fundamental=False, suppress_output=True):
    """
    Calculates 1 sided DFT
    frequencies: radians/sample
    phasors:  DFT scaled down using *2/N
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64).reshape(-1)
    N = len(signal)

    if fundamental is True:
        # 1 cycle DFT at the single fundamental frequency — direct dot product
        n = np.arange(N)
        omega = 2*np.pi/N
        correlates = np.exp(-1j * n * omega)
        IP = np.dot(signal, correlates)
        phasors = 2/N*IP
        return IP, phasors, omega, N

    # Full spectrum DFT via FFT — O(N log N) instead of O(N^2)
    IP = np.fft.rfft(signal)
    frequencies = np.arange(len(IP)) * 2 * np.pi / N
    phasors = (2/N) * IP

    return IP, phasors, frequencies, N

def SNR(signal, noise):

    signalInstPower = (signal**2) / 2
    signalEnergy = signalInstPower.sum()

    noiseInstPower = (noise**2) / 2
    noiseEngery = noiseInstPower.sum()

    ratio = signalEnergy/noiseEngery

    return ratio, signalInstPower, noiseInstPower

def process_signal(x, n, snrThresh, verbose=False):

    # All the chatty debug prints in this function go through _p.
    # Default verbose=False so parallel workers don't shred stdout.
    _p = print if verbose else (lambda *a, **k: None)

    #Setup
    endOfQuiet = 350
    bandLimitsHz = [18e3, 40e3]


    prate = 3.5e6
    Ts = 1/prate
    

    #Get full spectrum
    ft,_,omega,sampleLen = dft(x)
 
    #Figure out peak freq = FSIG, units are rad/sample
    bandLimitsOmega = 2*np.pi*Ts*np.array(bandLimitsHz)

    idx1 = np.searchsorted(omega,bandLimitsOmega[0],side="left")
    idx2 = np.searchsorted(omega,bandLimitsOmega[1],side="left")-1

    ft_band = ft[idx1:idx2+1]

    #Get max magnitude within the band
    val = np.max(np.abs(ft_band))
    idx = np.argmax(np.abs(ft_band))

    BANDOMEGA = omega[idx1:idx2+1]

    FSIG = BANDOMEGA[idx]
    FSIGHZ = FSIG / (2*np.pi) / Ts

    _p()
    _p(f"Peak frequency: {FSIGHZ} Hz  {FSIG} rad/sample")

    #Get energy of expected frequency = binEnergy
    idx = np.max(np.abs(ft))

    winSize = round(2*np.pi / FSIG)
    window = Window(winSize)
    numShifts = len(x) - winSize
    
    _p(f"WindowSize: {winSize} samples  rounded from {2*np.pi / FSIG}")
    _p(f"Fundamental frequency of window using rounded length N: {1 / (winSize * Ts)} Hz")
    _p(f"Number of shifts: {numShifts} phasor values")

    # Slide the window via FFT-based cross-correlation — O(N log N) instead of O(N * winSize)
    # IP[s] = sum_{m=0}^{winSize-1} x[s+m] * exp(-j*2*pi*m/winSize)
    # scipy.signal.correlate(x, y, 'valid') computes sum_m x[s+m] * conj(y[m]),
    # so we pass the conjugate of the correlator as y
    x = np.ascontiguousarray(x, dtype=np.float64)
    w_conj = np.exp(1j * 2 * np.pi * np.arange(winSize) / winSize)
    IP_full = correlate(x, w_conj, mode='valid', method='fft')
    IP = IP_full[:numShifts - 1]

    # Preserve the post-loop window state that downstream code expects
    # (original loop did numShifts-1 forward shifts, then one shift(-1))
    window.edge1 = numShifts - 1
    window.edge2 = window.edge1 + winSize - 1
    window.mid = window.edge1 + (winSize // 2)

    # omega was being overwritten inside the old loop; restore to match full-spectrum frequencies
    omega = np.arange(0, len(x) // 2 + 1) * 2 * np.pi / len(x)

    phasor = 2/winSize*IP

    '''SNR calculations
    {
    maxiumum IP in buffer = center of mass of signal (COM)
    Let the signal energy be considered on either side of the COM. 
    The noise floor will be the samples 0 to end of quiet.
    for SNR, # signal samples = # noise samples
    }'''

    signalIndices = np.arange(endOfQuiet, len(IP) - endOfQuiet)
    noiseIndices = np.arange(0,endOfQuiet)

    _p("len(IP):", len(IP))
    _p("endOfQuiet:", endOfQuiet)
    _p("signalIndices length:", len(signalIndices))
    _p("noiseIndices length:", len(noiseIndices))

    # Find center of mass (peak)
    idx = np.argmax(np.abs(IP[signalIndices]))
    

    # Adjust signal window
    startIdx = idx - endOfQuiet // 2 + endOfQuiet
    signalIndices = np.arange(startIdx+1, startIdx + endOfQuiet+1)

   
    # Extract periods
    quietPeriod = np.abs(phasor[noiseIndices])
    signalPeriod = np.abs(phasor[signalIndices])

    snr, SignalInstPower, noiseInstPower = SNR(signalPeriod, quietPeriod)

    if snr >= snrThresh:
        decision = 'good sample'
        flag=1
    else:
        decision = 'bad sample'
        flag = 0
    
    _p()
    _p(f"SNR: {snr} -> {decision}")

    if flag == 1:
        
        #compute powers for each window
        windowPowers = np.abs(phasor)**2/2

        #Get power threshold and start of signal based on noise floor and snr
        NoisePowerMax = np.max(noiseInstPower)
        SignalPowerMax = np.max(SignalInstPower)
        powerThreshold = (SignalPowerMax-NoisePowerMax)*.1 + NoisePowerMax
        

        startOfSignal = np.argmax(windowPowers > powerThreshold)


        #Smooth phasors with SMA before peak finding
        num = np.ones(window.size // 2) / (window.size // 2)
        den = 1
        y = lfilter(num,1,np.abs(phasor))
        smoothedPhasor = y[round(window.size/4):]


        #Get first peak
        derivative = np.diff(smoothedPhasor)
        derivative = derivative[startOfSignal:]

        neg_indices = np.where(derivative < 0)[0]
        idx_real = neg_indices[0]
        firstPeakIdx = idx_real + startOfSignal

        wavefrontRegion = np.abs(phasor[:firstPeakIdx + 1])
        midPoint = np.abs(phasor[firstPeakIdx]) / 2

        # Find index closest to midPoint
        midPointIdx = np.argmin(np.abs(wavefrontRegion - midPoint))
        elevation = 1/2

        '''#Adjust midPoint and elevation based on distance
        distance = firstPeakIdx - midPointIdx
        if distance > 2 * window.size:
            midPoint /= 2
            elevation = 1 / 8
        elif distance > window.size:
            midPoint /= 4
            elevation = 1 / 4
        else:
            elevation = 1 / 2'''

        # Refine placement using zero crossings
        closeness = wavefrontRegion - midPoint
        closeness_shifted = closeness[1:]
        closeness = closeness[:-1]  # same size as closeness_shifted
        products = closeness * closeness_shifted

        zero_crossings = np.where(products < 0)[0]
        if len(zero_crossings) == 0:
            midPointIdx = midPointIdx
        else:
            midPointIdx = zero_crossings[-1]

        # Land the window
        window.place(midPointIdx - 1) # or assign as needed

        # Print info
        _p(f"window.edge1 placed at {elevation} * first peak")

        #Wavefront selection func

        XWIN = x[window.edge1-1 : window.edge2]
        WF = wavefrontSelection(XWIN,FSIGHZ,window)
    else:
        WF = 0

    return WF, window.edge1, window.edge2, phasor, FSIGHZ, snr, flag, IP
