from pathlib import Path

# Define a class
class Sample:
    def __init__(self, path, amplitudeL=None, amplitudeR=None, n=None, wavefrontL=None, wavefrontR=None,
                  IPL=None, IPR=None, phasorL=None, phasorR=None, snrL=None, snrR=None, statusL=None, statusR=None, 
                  metSelectionL=None, metSelectionR=None, metStatusL=None, metStatusR=None):
        self.path = Path(path)
        self.sample_number = self.path.stem.split('-')[-1] 

        self.amplitudeL = amplitudeL 
        self.amplitudeR = amplitudeR   
        self.n = n
        self.wavefrontL = wavefrontL
        self.wavefrontR = wavefrontR
        self.IPL = IPL
        self.IPR = IPR
        self.phasorL = phasorL
        self.phasorR = phasorR
        self.snrL= snrL
        self.snrR= snrR
        self.statusL = statusL
        self.statusR = statusR

        self.metSelectionL = metSelectionL
        self.metSelectionR = metSelectionR
        self.metStatusL = metStatusL
        self.metStatusR = metStatusR
        


