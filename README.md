# sdr_helper

[![pypi](https://img.shields.io/pypi/v/sdr-helper.svg)](https://pypi.python.org/pypi/sdr-helper)
[![Docs](https://readthedocs.org/projects/sdr-helper/badge/?version=latest)](http://sdr-helper.readthedocs.io/en/latest/?badge=latest)

## Software Defined Radio Streaming to `rtlsdr_helper` with Interface to `pyaudio_helper`
A recent push to the master branch now allows real-time SDR streaming from the RTL-SDR to `pyaudio_helper`.


This capability is made possible via the new `asynch` and `await` capabilities of Python 3.7. For the details as to how this works you have to dig into the details found in the module `rtlsdr_helper.py` and the examples found in the notebook `rtlsdr_helper_streaming_sample.ipynb`.

This is just the beginning of making a complete SDR receiver possible in a Jupyter notebook. Not only is the receiver a reality, the algorithms that implement the receiver, in Python, can easily be coded by the user.

To help develop demodulator algorithms a streaming code block interface standard, of sorts, is being developed this summer. The idea is to provide examples of how to write a simple Python class that will manage states in the DSP code that is inside the *Callback Process* block of the block diagram. More details by the end of the summer is expected, along with another sample notebook.

### Extras

This package contains the helper module `rtlsdr_helper`, and depends on `pyaudio_helper` which require the packages [pyrtlsdr](https://pypi.python.org/pypi/pyrtlsdr) and [PyAudio](https://pypi.python.org/pypi/PyAudio). To use the full functionality of these helpers, install the package from the scikit-dsp-comm folder as follows:

```
pip install -e .[helpers]
```

1. `rtlsdr_helper.py` interfaces with `pyrtldsr` to provide a simple captures means for complex baseband software defined radio (SDR) samples from the low-cost (~$20) RTL-SDR USB hardware dongle. The remaining functions in this module support the implementation of demodulators for FM modulation and examples of complete receivers for FM mono, FM stereo, and tools for FSK demodulation, including bit synchronization. Real-time streaming is a new capability included.

## Feature: Added Software Defined Radio Streaming to `rtlsdr_helper` with Interface to `pyaudio_helper`

A recent push to the master branch now allows real-time SDR streaming from the RTL-SDR to `pyaudio_helper`. In this first release of the API, the system block diagram takes the from shown in the figure below:

![Block diagram for RTL-SDR streaming](rtlsdr_helper_streaming_block.png)

This capability is made possible via the new `aynch` and `await` capabilities of Python 3.7. For the details as to how this works you have to dig into the details found in the module `rtlsdr_helper.py` and the examples found in the notebook `rtlsdr_helper_streaming_sample.ipynb`. A screenshot from the sample Jupyter notebook, that implements a broadcast FM receiver, is shown below:

 ![Code snippet for an FM radio receiver.](rtlsdr_helper_streaming_FM_receiver.png)

This is just the beginning of making a complete SDR receiver possible in a Jupyter notebook. Not only is the receiver a reality, the algorithms that implement the receiver, in Python, can easily be coded by the user.

To help develop demodulator algorithms a streaming code block interface standard, of sorts, is being developed this summer. The idea is to provide examples of how to write a simple Python class that will manage states in the DSP code that is inside the *Callback Process* block of the block diagram. More details by the end of the summer is expected, along with another sample notebook.

## Authors

[@mwickert](https://github.com/mwickert)

[@andrewsmitty](https://github.com/andrewsmitty)
