# sdr_helper

## Software Defined Radio Streaming to `rtlsdr_helper` with Interface to `pyaudio_helper`
A recent push to the master branch now allows real-time SDR streaming from the RTL-SDR to `pyaudio_helper`.


This capability is made possible via the new `asynch` and `await` capabilities of Python 3.7. For the details as to how this works you have to dig into the details found in the module `rtlsdr_helper.py` and the examples found in the notebook `rtlsdr_helper_streaming_sample.ipynb`.

This is just the beginning of making a complete SDR receiver possible in a Jupyter notebook. Not only is the receiver a reality, the algorithms that implement the receiver, in Python, can easily be coded by the user.

To help develop demodulator algorithms a streaming code block interface standard, of sorts, is being developed this summer. The idea is to provide examples of how to write a simple Python class that will manage states in the DSP code that is inside the *Callback Process* block of the block diagram. More details by the end of the summer is expected, along with another sample notebook.

## Authors

[@mwickert](https://github.com/mwickert)

[@andrewsmitty](https://github.com/andrewsmitty)
