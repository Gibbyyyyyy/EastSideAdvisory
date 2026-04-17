import tkinter as tk
from tkinter import ttk
import os
import threading

#Function that runs the entire GUI and starts the signal processing
def start_gui():

    #GUI Window
    root = tk.Tk() 
    root.title("Ultrasonic Wavefront Detection")
    root.geometry("500x420")

    # Match to dark mode theme
    BG = "#1a202c"
    FG = "#e2e8f0"
    CTRL_BG = "#4a5568"
    CTRL_HOVER = "#5a677d"
    root.tk_setPalette(background=BG, foreground=FG, activeBackground=CTRL_HOVER, activeForeground=FG)

    # ttk.Combobox needs special styling because it ignores tk_setPalette
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox", fieldbackground=CTRL_BG, background=CTRL_BG, foreground=FG, arrowcolor=FG, font=("DejaVu Sans", 11, "bold italic"))
    style.map("TCombobox", fieldbackground=[("readonly", CTRL_BG)])
    style.configure("TProgressbar", background="#20c997", troughcolor=CTRL_BG)
    root.option_add("*TCombobox*Listbox.background", CTRL_BG)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.font", ("DejaVu Sans", 11, "bold italic"))

#This will close the Python instance so that it will properly open again after closing the GUI
    def on_close():
        # Shut the worker pool down cleanly before killing the interpreter,
        # so workers exit in an orderly way instead of being orphaned.
        try:
            from process_sheet import _EXECUTOR
            if _EXECUTOR is not None:
                _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        root.destroy()
        os._exit(0)
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    #Main Page / Main Menu
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    #Title
    top_label = tk.Label(main_frame, text="Ultrasonic Wavefront Detection GUI", font=("DejaVu Sans", 16, "bold italic"))
    top_label.pack(pady=20)



    #### Sheet Number drop down ####
    tk.Label(main_frame, text="Select Sheet Number:", font=("DejaVu Sans", 12, "bold italic")).pack(pady=5)

    #Hard coded path. This path will not change.
    path = r"C:\EastSideAdvisory\EE416kalmia\Python\Lab Data\export"

    #get_sheet_nums returns all the different sheet numbers that are found in the folder. We could hard code the sheet numbers that are 
    #in the folder but this is more robust
    sheetVals = get_sheet_nums(path)

    #Simple error check
    if not sheetVals:
        sheetVals = ["No sheets found"]

    #Creating the dropdown box with sheets numbers that were found in the data folder
    sheetNumber = tk.StringVar()
    sheetDropdown = ttk.Combobox(
        main_frame, textvariable=sheetNumber, 
        values = sheetVals, state="readonly"
    )
    sheetDropdown.pack(pady=5)
    sheetDropdown.current(0)

# Pre-warm the worker pool in the background while the user picks their
    # options. By the time they click Start, workers have already booted and
    # imported NumPy/SciPy, so processing begins immediately.
    def _prewarm():
        import process_sheet
        # If a pool already exists from a previous GUI session, it's warm — skip.
        if process_sheet._EXECUTOR is not None:
            return
        executor = process_sheet._get_executor()
        # Submit a trivial no-op to every worker to force them to actually spawn
        # and run their initializer (the pool is lazy until the first submit).
        n_workers = executor._max_workers
        list(executor.map(int, [0] * n_workers))
    threading.Thread(target=_prewarm, daemon=True).start()


    #### Noise Variance drop down ####
    tk.Label(main_frame, text="Select Noise Variance:", font=("DejaVu Sans", 12, "bold italic")).pack(pady=5)

    #Hard coding a few different options for AWGN. Just showing the functionality of the program. Will never be used outside of testing. 
    noiseVariance = tk.StringVar()
    noiseDropdown = ttk.Combobox(
        main_frame, textvariable=noiseVariance, 
        values = ["0.0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "1.0", "2.0"], state="readonly"
    )
    noiseDropdown.pack(pady=5)
    noiseDropdown.current(0)



    #### SNR Threshold drop down ####
    tk.Label(main_frame, text="Select SNR Threshold:", font=("DejaVu Sans", 12, "bold italic")).pack(pady=5)

    #Hard coding different SNR thresholds that the user can choose to use. The default will be set to 100
    SNRThresh = tk.StringVar()
    SNRDropdown = ttk.Combobox(
        main_frame, textvariable=SNRThresh, 
        values = ["10", "50", "75", "100", "150", "200"], state="readonly"
    )
    SNRDropdown.pack(pady=5)
    SNRDropdown.current(3)

#### Start Button and starting the signal processing ####
    def start():
        # Clean up any matplotlib figures from a previous run BEFORE we kick
        # off a new background thread. matplotlib's TkAgg backend wraps Tk
        # PhotoImage and Tk Variable objects — those have to be destroyed on
        # the thread that owns the Tk main loop (this one). If we let them
        # linger, Python's GC will eventually collect them from the worker
        # thread, which throws "main thread is not in main loop".
        import gc
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        gc.collect()

        sheet = sheetNumber.get()
        noise = noiseVariance.get()
        snr = SNRThresh.get()

        #Printing out the variables to help with debugging/testing
        print("Starting with:")
        print("Sheet:", sheet)
        print("Noise Variance:", noise)
        print("SNR Threshold:", snr)

        # Disable start button and show progress widgets
        start_button.config(state="disabled")
        progress_var.set(0)
        progress_label.config(text="0% Complete")
        progress_bar.pack(pady=5)
        progress_label.pack(pady=5)

        # Callback used by process_sheet; schedules UI update on main thread
        def update_progress(current, total):
            pct = round(100 * current / total)
            root.after(0, lambda: (progress_var.set(pct), progress_label.config(text=f"{pct}% Complete")))

# Timestamp the moment Start was pressed so we can report total
        # processing time once the last sample finishes.
        import time
        t_start = time.perf_counter()

        # Worker runs in background so the window stays responsive
        def worker():
            from process_sheet import process_sheet
            samples = process_sheet(int(sheet), r"C:\EastSideAdvisory\EE416kalmia\Python\Lab Data", int(snr), progress_callback=update_progress)

            elapsed = time.perf_counter() - t_start
            print(f"\nTotal processing time: {elapsed:.2f} seconds")

            def done():
                progress_bar.pack_forget()
                progress_label.pack_forget()
                start_button.config(state="normal")
                from InitialGraphingFunction import show_samples
                show_samples(samples)
            root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    #When the start button is pressed on the main menu it initiates start() as shown above
    start_button = tk.Button(main_frame, text="Start", font=("DejaVu Sans", 14, "bold italic"), command=start)
    start_button.pack(pady=30)

    # Progress bar widgets — hidden until Start is pressed
    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100, length=300)
    progress_label = tk.Label(main_frame, text="", font=("DejaVu Sans", 12, "bold italic"))

    root.mainloop()


#Dynamically getting the sheet numbers that are in the data folder. Not essential but nice functionality.
def get_sheet_nums(path):
    folders = []

    #Getting all the items in the data folder path
    for item in os.listdir(path):
        full_path = os.path.join(path,item)
        if os.path.isdir(full_path):
            folders.append(item)
    
    #Sorting the sheet numbers and doing a bit of checking to verify that it is a digit
    folders.sort(key=lambda x: int(x) if x.isdigit() else x)

    return folders


if __name__== "__main__":

    #Function that runs the entire GUI and starts the signal processing
    start_gui()