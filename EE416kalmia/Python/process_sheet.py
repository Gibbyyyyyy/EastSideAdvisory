import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from sample import Sample
import pandas as bear
import numpy as np
from dsp_functions import dft, process_signal
from auxilary_functions import openSheet


# Module-level executor — built once, reused across every process_sheet call.
# This keeps workers alive between GUI "Start" clicks so we only pay the
# Python/NumPy/SciPy import cost once per GUI session instead of once per run.
_EXECUTOR = None


def _get_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(initializer=_worker_init)
    return _EXECUTOR


def _worker_init():
    """Runs once in each worker when it spawns. Pre-imports the heavy
    scientific stack so the first real task doesn't eat the import cost."""
    # Force matplotlib (pulled in transitively) to use a non-GUI backend.
    # Without this, pandas/matplotlib can instantiate a Tk backend in the
    # worker, which then throws "main thread is not in main loop" errors
    # when the worker shuts down because there is no Tk mainloop here.
    import matplotlib
    matplotlib.use("Agg", force=True)

    import numpy  # noqa: F401
    import scipy.signal  # noqa: F401
    import pandas  # noqa: F401
    import dsp_functions  # noqa: F401
    import sample  # noqa: F401


def _process_one_sample(args):
    """Worker function — runs in a separate process.
    Does CSV read + both L/R signal processes for one sample, so both I/O
    and compute are parallelized together."""
    samplePathObject, SNRThresh = args

    sample_obj = Sample(path=samplePathObject)

    table = bear.read_csv(sample_obj.path)
    sample_obj.amplitudeL = table["RawL"].to_numpy()
    sample_obj.amplitudeR = table["RawR"].to_numpy()
    sample_obj.n = np.arange(len(table))
    sample_obj.metSelectionL = np.int64(table["IdxL"][1])
    sample_obj.metSelectionR = np.int64(table["IdxR"][1])
    sample_obj.metStatusL = int(table["IdxL"][1] > 0)
    sample_obj.metStatusR = int(table["IdxR"][1] > 0)

    WF, edge1, edge2, phasor, FSIGHZ, snr, flag, IP = process_signal(
        sample_obj.amplitudeL, sample_obj.n, SNRThresh
    )
    sample_obj.phasorL = phasor
    sample_obj.IPL = IP
    sample_obj.snrL = snr
    sample_obj.statusL = flag
    sample_obj.edge1L = edge1
    sample_obj.edge2L = edge2
    sample_obj.wavefrontL = WF

    WF, edge1, edge2, phasor, FSIGHZ, snr, flag, IP = process_signal(
        sample_obj.amplitudeR, sample_obj.n, SNRThresh
    )
    sample_obj.phasorR = phasor
    sample_obj.IPR = IP
    sample_obj.snrR = snr
    sample_obj.statusR = flag
    sample_obj.edge1R = edge1
    sample_obj.edge2R = edge2
    sample_obj.wavefrontR = WF

    return int(sample_obj.sample_number), sample_obj


def process_sheet(sheetNum, path, SNRThresh, progress_callback=None):

    samplePathObjects = openSheet(r"C:\EE416kalmia\Python\Lab Data", sheetNum)

    args_list = [(spo, SNRThresh) for spo in samplePathObjects]
    total = len(args_list)
    samples = {}

    # Submit work. If the pool is dead from a prior crash, rebuild it and retry once.
    try:
        executor = _get_executor()
        futures = [executor.submit(_process_one_sample, args) for args in args_list]
    except BrokenProcessPool:
        global _EXECUTOR
        _EXECUTOR = None
        executor = _get_executor()
        futures = [executor.submit(_process_one_sample, args) for args in args_list]

    for i, future in enumerate(as_completed(futures), 1):
        try:
            sample_num, sample_obj = future.result()
            samples[sample_num] = sample_obj
        except Exception as e:
            # One sample blew up — log it, skip it, keep going so the rest of
            # the sheet still finishes instead of taking the whole run down.
            print(f"[process_sheet] sample failed: {type(e).__name__}: {e}")
        if progress_callback:
            progress_callback(i, total)

    return samples