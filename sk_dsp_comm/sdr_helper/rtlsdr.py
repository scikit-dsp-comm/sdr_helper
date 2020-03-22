"""
Support functions for the RTL-SDR using pyrtlsdr

Copyright (c) July 2017, Mark Wickert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""
import warnings

try:
    from rtlsdr import RtlSdr
except ImportError:
    warnings.warn("Please install the helpers extras for full functionality", ImportWarning)
from sk_dsp_comm import sigsys as ss
import numpy as np
import scipy.signal as signal
import asyncio
import colorama
from pyaudio_helper import pyaudio_helper as pah
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Math
except ImportError:
    warnings.warn("Please install IPython for full functionality", ImportWarning)
try:
    from ipywidgets import interactive
    from ipywidgets import ToggleButtons
    from ipywidgets import FloatSlider
    from ipywidgets import Layout
    from ipywidgets import widgets
except ImportError:
    warnings.warn("Please install ipywidgets for full functionality", ImportWarning)
from matplotlib.mlab import psd


# Bokeh plotting
# from bokeh.io import push_notebook, show, output_notebook
# from bokeh.models import HoverTool
# import bokeh.plotting.figure as bfigure
# from bokeh.models.annotations import Title as bTitle

class RTLSDRStream(object):
    """
    Class used to set up an RTLSDR stream object
    """

    def __init__(self, rtl_in=0, rtl_fs=2.4e6, fc=103.9e6, gain=40, rtl_buffer_size=2 ** 15, audio_out=1,
                 audio_buffsize=4096, audio_fs=48000):
        '''
        RTLSDR async streaming class

        Parameters
        ----------
        rtl_in: The index of the RTLSDR (either 0 or 1)
        rtl_fs: Sets the sample rate of the RTLSDR (Generally want to keep 2.4e6)
        fc: Sets the tuning frequency of the RTLSDR
        gain: Sets the gain of the RTLSDR
        rtl_buffer_size: Sets the size of the circular buffer (see below)
        audio_out: Select the audio output device index (check devices with )

        The following shows the basic flow of an RTLSDR_stream object 

                                      __________________________________________
                                      |              Audio Sink                |
        ____________    __________    |____________    ___________    _________|
        |Stage1 Dec| -> |callback| -> ||Stage2 Dec| -> |Circ Buff| -> |PyAudio||
        |__________|    |________| |  ||__________|    |_________|    |_______||
                                   or |________________________________________|
                                   |
                                   |   ______________
                                    -->| Data Sink  |
                                       |____________|

        It consists of a Stage 1 Decimator that can be used to decimate the RF 
        signal coming in from the RTLSDR to a lower rate to make processing 
        easier. The stage 1 decimator can be defined by setting the filter
        taps, initial conditions, and decimation factor M. The numerator 
        coefficients, initial conditions, and denominator coefficients of the
        filter are available as parameters in the run_user_stream() method.

        Custom Stage 1 FIR decimation filter example:
        >>> sdr_stream = RTLSDRStream()
        >>> M1 = 10
        >>> M2 = 5
        >>> b = signal.firwin(32,2*200e3/2.4e6)
        >>> stage1_ic = signal.lfilter_zi(b,1)
        >>> sdr_stream.run_user_stream(callback,M1,M2,b,stage1_ic)


        The Callback block can be used to process the decimated samples. This is
        where the "meat" of the processing is done. This processing callback receives
        the following parameters:
            samples: incoming decimated frame of samples
            fs: RTLSDR sample rate
            user_var: user-defined variable that gets passed through the class
        The callback can return data in two forms. It can either return an array of 
        process samples of the same length as the input frame which can then be sent
        along to the stage 2 decimation filter, circular buffer, and pyAudio output, or
        the data can be stored in an output buffer that can be accessed through an 
        async method (see below)

        The following is an example of a callback function that implements an FM
        discriminator:
        >>> def callback(samples,fs,user_var):
        >>>     # discriminator
        >>>     x = samples
        >>>     X=np.real(x)
        >>>     Y=np.imag(x)
        >>>     b=np.array([1,-1])
        >>>     a=np.array([1,0])
        >>>     derY=signal.lfilter(b,a,Y)
        >>>     derX=signal.lfilter(b,a,Y)
        >>>     z_bb=(X*derY-Y*derX)/(X**2+Y**2)
        >>>     return z_bb,user_var

        The Stage 2 Decimator callback can be used to decimate the processed frame
        of samples down to an audio rate. The interface is basically the same as the
        stage 1 decimator callback. The numerator and denominator coefficients can
        be given for the stage 2 filter

        Custom Stage 2 IIR Filter:
        >>> import sk_dsp_comm.iir_design_helper as iir_d
        >>> sdr_stream = RTLSDRStream()
        >>> M1 = 10
        >>> M2 = 5
        >>> fc2 = 15e3
        >>> bb,aa,sos2 = iir_d.IIR_lpf(fc2,fc2+5000,1,10,2.4e6/M1)
        >>> stage2_ic = signal.lfilter_zi(bb,aa)
        >>> sdr_stream.run_user_stream(callback,M1,M2,bb=bb,stage2_ic=stage2_ic,aa=aa)

        When the audio sink parameter is set to True, the processed audio rate 
        samples are stored in a circular buffer using the _write_circular_buffer() 
        private method. PyAudio runs in a separate thread that reads from the 
        circular buffer using the _read_circular_buffer() private method. It then 
        sends the audio samples to the selected audio output.

        When the audio sink parameter is set to False, then the data is stored in a 
        buffer that can be accessed by an async method get_data_out_async(). This
        method waits on a queue to return the filled output data buffer. The following
        example shows a very simple application of this idea. The example shows a 
        frame counter which simply 

        Async Data Out Example:
        >>> import asyncio
        >>> sdr_stream.set_rtl_buffer_size(16)

        >>> def no_audio_callback(samples,fs,user_var):
        >>>     frame_count = user_var
        >>>     user_var = user_var+1
        >>>     return np.array([frame_count]),user_var
        >>> global keep_collecting

        >>> async def handle_data_out():
        >>>     global keep_collecting
        >>>     keep_collecting = True
        >>>     while keep_collecting:
        >>>         data_out = await sdr_stream.get_data_out_async()
        >>>         print(data_out)
        >>>     sdr_stream.reset_data_out_queue()
        >>>     print('Done')

        >>> sdr_stream.run_user_stream(no_audio_callback,1,1,audio_sink=False,user_var=1)
        >>> task = asyncio.create_task(handle_data_out())

        >>> keep_collecting = False
        >>> sdr_stream.stop()

        Andrew Smit April 2019
        .. ._.. .._
        '''
        self.rtl_in = rtl_in
        self.fs = rtl_fs
        self.fc = fc
        self.gain = gain
        self.rtl_buffer_size = rtl_buffer_size
        self.audio_buffsize = audio_buffsize
        self.audio_fs = audio_fs
        self.keep_streaming = False
        self.audio_in = 0
        self.audio_out = audio_out
        self.output = widgets.Output()
        self.z_out = np.zeros(rtl_buffer_size)
        self.rx_idx = 0
        self.audio_idx = int(rtl_buffer_size / 2)
        self.audio_gain = 1.0
        self.audio_sink = True
        self.user_var = False
        self.buffer_exceeded = False

        # Stage 1 Decimation Filter
        self.M1 = 10.0
        self.b = signal.firwin(32, 2.0 * 200e3 / float(self.fs))
        self.a = np.array([1])
        self.stage1_ic = signal.lfilter_zi(self.b, self.a)

        # Stage 2 Decimation Filter
        self.M2 = 5.0
        self.bb = signal.firwin(32, 2.0 * 16.0e3 / float(self.fs) * 10.0)
        self.aa = np.array([1])
        self.stage2_ic = signal.lfilter_zi(self.bb, self.aa)

        # Discriminator Filter Initial conditions
        self.Y_ic = signal.lfilter_zi(np.array([1, -1]), np.array([1, 0]))
        self.X_ic = self.Y_ic

        # Connect to the SDR
        self.sdr = RtlSdr(rtl_in)
        self.output.append_stdout('LOGS:\n')
        self.sdr.set_sample_rate(rtl_fs)
        self.sdr.set_center_freq(fc)
        self.sdr.set_gain(gain)

        # Audio
        self.DSP_IO = pah.DSPIOStream(self._audio_callback, self.audio_in, self.audio_out, self.audio_buffsize,
                                        self.audio_fs, 0, 0)

        # Async Queues/Plotting
        self.rx_data = asyncio.Queue()
        self.rf_queue = asyncio.Queue()
        self.plot_NFFT = 1024
        self.update_rf = False
        self.refresh_rate = 1
        self.stage1_queue = asyncio.Queue()
        self.update_stage1 = False
        self.processed_stage1_queue = asyncio.Queue()
        self.update_processed_stage1 = False
        self.stage2_queue = asyncio.Queue()
        self.update_stage2 = False
        self.data_out_queue = asyncio.Queue()

        self.store_rf = False
        self.rf_frame = np.array([])
        self.store_stage1 = False
        self.stage1_frame = np.array([])
        self.store_processed_stage1 = False
        self.processed_stage1_frame = np.array([])
        self.store_stage2 = False
        self.stage2_frame = np.array([])
        self.invert = False
        # output_notebook()

    def _interaction(self, stream):
        '''
        Enables Jupyter Widgets for mono FM example
        '''
        if (stream == 'Start Streaming'):
            self.clear_buffer()
            task = asyncio.create_task(self._start_streaming())
            print('Status: Streaming')
        else:
            self.stop()
            print('Status: Stopped')

    def interactive_fm_rx(self, fc=103.9e6, gain=40, audio_out=1, audio_buffsize=4096, audio_fs=48000):
        '''
        Sets up interactive mono FM example
        '''
        self.set_fs(2.4e6)
        self.set_fc(fc)
        self.set_gain(gain)
        self.set_audio_in(0)
        self.set_audio_out(audio_out)
        self.set_audio_buffsize(audio_buffsize)
        self.set_audio_fs(audio_fs)
        self.togglebutts = ToggleButtons(
            options=['Start Streaming', 'Stop Streaming'],
            description=' ',
            value='Stop Streaming',
        )
        self.togglebutts.style.button_width = "400px"
        self.togglebutts.style.description_width = "1px"

        self.play = interactive(self._interaction, Stream=self.togglebutts)

        title = widgets.Output()
        title.append_stdout("Interactive FM Receiver")
        display(title)
        display(self.play)
        self._interact_audio_gain()
        self._interact_frequency(self.fc / 1e6)

    def _interact_frequency(self, freq_val, min_freq=87.5, max_freq=108, freq_step=0.2):
        '''
        Sets up tuning frequency slider widget for Mono FM Example
        '''
        self.slider = FloatSlider(
            value=freq_val,
            min=min_freq,
            max=max_freq,
            step=freq_step,
            description=r'$f_c\;$',
            continuous_update=False,
            orientation='horizontal',
            readout_format='0.1f',
            layout=Layout(
                width='90%',
            )
        )
        self.slider.style.handle_color = 'lightblue'

        self.center_freq_widget = interactive(self.set_fc_mhz, fc=self.slider)
        display(self.center_freq_widget)

    def _interact_audio_gain(self, gain_val=0, min_gain=-60, max_gain=6, gain_step=0.1):
        '''
        Sets up audio gain slider widget for Mono FM Example
        '''
        self.gain_slider = FloatSlider(
            value=gain_val,
            min=min_gain,
            max=max_gain,
            step=gain_step,
            description='Gain (dB)',
            continuous_update=True,
            orientation='horizontal',
            readout_format='0.1f',
            layout=Layout(
                width='90%',
            )
        )
        self.gain_slider.style.handle_color = 'lightgreen'

        self.audio_gain_widget = interactive(self.set_audio_gain_db, gain=self.gain_slider)
        display(self.audio_gain_widget)

    async def _get_rx_data(self):
        '''
        Gets samples from RTLSDR and decimates by 10 for mono FM example
        '''
        extra = np.array([])
        async for samples in self.sdr.stream():
            # Do something with the incoming samples
            samples = np.concatenate((extra, samples))
            mod = len(samples) % 10
            if (mod):
                extra = samples[len(samples) - mod:]
                samples = samples[:len(samples) - mod]
            else:
                extra = np.array([])

            if (self.store_rf):
                self.store_rf = False
                await self.rf_queue.put(samples)

            z = self._decimate(samples, 10, self.fs)

            if (self.store_stage1):
                self.store_stage1 = False
                await self.stage1_queue.put(z)

            await self.rx_data.put(z)

    async def _process_rx_data(self):
        '''
        Processes decimated samples, and decimates further to audio rate.
        Implements an FM discriminator for the mono FM example.
        '''
        while self.keep_streaming:
            samples = await self.rx_data.get()
            samples = np.array(samples)

            ##############################################
            # Process Downconverted Data

            z_bb = self._discrim(samples)
            if (self.store_processed_stage1):
                self.store_processed_stage1 = False
                await self.processed_stage1_queue.put(z_bb)

            z = self._decimate(z_bb, 5, self.fs, 2)
            if (self.store_stage2):
                self.store_stage2 = False
                await self.stage2_queue.put(z)

            # Wrap circular buffer
            self._write_circ_buff(z)
            ###############################################

        with self.output:
            self.output.append_stdout(colorama.Fore.BLUE + 'Stopping SDR\n')

        self.sdr.stop()
        if (self.DSP_IO):
            with self.output:
                self.output.append_stdout(colorama.Fore.BLUE + "Stopping Audio\n")
            self.DSP_IO.stop()
        self.play.children[0].value = "Stop Streaming"

    async def _get_rx_data_user(self):
        '''
        Used by run_user_stream() method. Asynchronously reads in samples from
        RTLSDR and implements the stage 1 decimator. The stage 1 decimator
        can be defined by the user in run_user_stream() or a default decimator
        can be used. This is a private method that is only used internally. 
        Decimated samples are stored in the rx_data queue which is consumed by
        the _process_rx_data_user private method.
        '''
        extra = np.array([])
        async for samples in self.sdr.stream():
            # Do something with the incoming samples
            samples = np.concatenate((extra, samples))
            mod = len(samples) % self.M1
            if (mod):
                extra = samples[len(samples) - mod:]
                samples = samples[:len(samples) - mod]
            else:
                extra = np.array([])

            if (self.store_rf):
                self.store_rf = False
                await self.rf_queue.put(samples)

            y, self.stage1_ic = signal.lfilter(self.b, self.a, samples, zi=self.stage1_ic)
            z = ss.downsample(y, self.M1)

            if (self.store_stage1):
                self.store_stage1 = False
                await self.stage1_queue.put(z)

            await self.rx_data.put(z)

    async def _process_rx_data_user(self, callback):
        '''
        Used by run_user_stream() method. Consumed decimated samples from
        stage 1 decimator stored in the rx_data queue. Passes the data along
        to a user defined callback. The processed samples are passed along
        to either the audio sink or the data sink. The audio sink contains
        a stage 2 decimator and outputs to an audio device via PyAudio. The
        data sink stores processed samples in a buffer that can be read out
        asynchronously. This is a private method that is used internally.

        parameters:
        -----------
        callback: user-defined callback passed in from run_user_stream()
        '''
        while self.keep_streaming:
            samples = await self.rx_data.get()
            samples = np.array(samples)

            ##############################################
            # Process Downconverted Data in user callback
            z_bb, self.user_var = callback(samples, self.fs / self.M1, self.user_var)
            ##############################################

            if (self.audio_sink):
                if (self.store_processed_stage1):
                    self.store_processed_stage1 = False
                    await self.processed_stage1_queue.put(z_bb)

                y, self.stage2_ic = signal.lfilter(self.bb, self.aa, z_bb, zi=self.stage2_ic)
                z = ss.downsample(y, self.M2)

                if (self.store_stage2):
                    self.store_stage2 = False
                    await self.stage2_queue.put(z)

                # Wrap circular buffer
                self._write_circ_buff(z)

            else:
                await self._write_circ_buff_async(z_bb)

        print(colorama.Fore.BLUE + 'Stopping SDR')

        self.sdr.stop()
        if (self.DSP_IO and self.audio_sink):
            print(colorama.Fore.BLUE + "Stopping Audio")
            self.DSP_IO.stop()

        print(colorama.Fore.BLUE + 'Completed')

    def _write_circ_buff(self, samples):
        '''
        Private method used to write samples to a circular buffer. This circular
        buffer takes in audio-rate samples from the _process_rx_user_data method
        and _read_circ_buff reads the samples back out asynchronously in a PyAudio
        thread. This method increments a write pointer by the length of the samples
        being written to the buffer and wraps the pointer when the buffer is filled.

        Parameters:
        -----------
        samples: audio-rate samples to be written to the circular buffer
        '''
        # Wrap circular buffer
        if (self.rx_idx + len(samples) >= self.rtl_buffer_size):
            self.z_out[self.rx_idx:] = samples[:(self.rtl_buffer_size - self.rx_idx)]
            if (not self.audio_sink):
                print(colorama.Fore.RED + 'Exceeded allocated output buffer space. Returning, then overwriting buffer')
                self.buffer_exceeded = True
                # await self.data_out_queue.put(self.z_out)
            self.z_out[:abs(self.rtl_buffer_size - self.rx_idx - len(samples))] = samples[(abs(
                self.rtl_buffer_size - self.rx_idx)):]
            self.rx_idx = abs(self.rtl_buffer_size - self.rx_idx - len(samples))
        else:
            self.z_out[self.rx_idx:self.rx_idx + len(samples)] = samples
            self.rx_idx = self.rx_idx + len(samples)

    async def _write_circ_buff_async(self, samples):
        '''
        Private method used to asynchronously store processed data from the user
        callback to a buffer when the data sink is being used. This method wraps the
        buffer and increments the buffer pointer.

        parameters:
        -----------
        samples: decimated processed samples to be stored in data sink buffer
        '''
        # Wrap circular buffer
        if (self.rx_idx + len(samples) >= self.rtl_buffer_size):
            if (not self.audio_sink):
                # print(colorama.Fore.RED + 'Exceeded allocated output buffer space. Returning, then overwriting buffer')
                self.buffer_exceeded = True
                await self.data_out_queue.put(self.z_out)
            self.z_out[self.rx_idx:] = samples[:(self.rtl_buffer_size - self.rx_idx)]
            self.z_out[:abs(self.rtl_buffer_size - self.rx_idx - len(samples))] = samples[(abs(
                self.rtl_buffer_size - self.rx_idx)):]
            self.rx_idx = abs(self.rtl_buffer_size - self.rx_idx - len(samples))
        else:
            self.z_out[self.rx_idx:self.rx_idx + len(samples)] = samples
            self.rx_idx = self.rx_idx + len(samples)

    def _read_circ_buff(self, frame_count):
        '''
        Private method used to read samples from the circular buffer. This is used
        by the audio sink to consume audio-rate samples. This method handles incrementing
        a read pointer and wrapping the circular buffer.
        '''
        y = np.zeros(frame_count)
        if (self.audio_idx + frame_count >= self.rtl_buffer_size):
            y[:(self.rtl_buffer_size - self.audio_idx)] = self.z_out[self.audio_idx:]
            y[(self.rtl_buffer_size - self.audio_idx):] = self.z_out[
                                                          :(self.rtl_buffer_size - frame_count - self.audio_idx)]
            self.audio_idx = abs(self.rtl_buffer_size - self.audio_idx - frame_count)
        else:
            y = self.z_out[self.audio_idx:self.audio_idx + frame_count]
            self.audio_idx = self.audio_idx + frame_count

        return y

    async def _audio(self):
        '''
        private method that starts a PyAudio Thread
        '''
        self.DSP_IO.thread_stream(0, 1)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        '''
        private audio callback method that is used by the PyAudio thread in
        the audio sink. Reads samples out of the circular buffer and sends the
        samples out an audio device.
        '''
        # convert byte data to ndarray
        # in_data_nda = np.frombuffer(in_data, dtype=np.int16)
        # ***********************************************
        # DSP operations here

        # Read samples in from circular buffer
        y = self._read_circ_buff(frame_count)
        y = y * self.audio_gain * 2 ** 14
        # ***********************************************
        # Convert from float back to int16
        y = y.astype(np.int16)
        # Convert ndarray back to bytes
        return y.tobytes(), pah.pyaudio.paContinue

    async def _start_streaming(self):
        '''
        Async method used to start coroutine for the Mono FM example
        '''
        self.rx_data = asyncio.Queue()
        self.clear_buffer()
        self.DSP_IO = pah.DSPIOStream(self._audio_callback, self.audio_in, self.audio_out, self.audio_buffsize,
                                        self.audio_fs, 0, 0)
        self.keep_streaming = True

        loop = asyncio.get_event_loop()

        with self.output:
            self.output.append_stdout(colorama.Fore.LIGHTBLUE_EX + 'Starting SDR and Audio Event Loop\n')

        await asyncio.gather(
            self._get_rx_data(),
            self._process_rx_data(),
            self._audio()
        )

    async def _start_user_stream(self, callback, m1, m2, b, a, stage1_ic, bb, aa, stage2_ic, audio_sink, user_var):
        '''
        Async method used by run_user_stream method to start a coroutine running all of the
        different async stages in the chain. 

        parameters:
        -----------
        callback: user-defined callback passed in from run_user_stream()
        M1: Stage 1 decimation factor passed in from run_user_stream()
        M2: Stage 2 decimation factor passed in from run_user_stream()
        b: Stage 1 filter numerator coefficients passed in from run_user_stream()
        stage1_ic: Stage 1 filter initial conditions passed in from run_user_stream()
        a: Stage 1 filter denominator coefficients passed in from run_user_stream()
        bb: Stage 2 filter numerator coefficients passed in from run_user_stream()
        stage2_ic: Stage 2 filter initial conditions passed in from run_user_stream()
        aa: Stage 2 filter denominator coefficients passed in from run_user_stream()

        '''
        if (type(b) == np.ndarray):
            self.b = b
        else:
            print(colorama.Fore.LIGHTBLUE_EX + 'Using default stage 1 decimation filter')
        if (type(bb) == np.ndarray):
            self.bb = bb
        else:
            if (audio_sink):
                print(colorama.Fore.LIGHTBLUE_EX + 'Using default stage 2 decimation filter')
        if (type(stage1_ic) == np.ndarray):
            if (len(stage1_ic) == len(self.b) - 1):
                self.stage1_ic = stage1_ic
            else:
                raise ValueError('Stage 1 Filter initial conditions length does not match filter taps')
        else:
            if (len(self.stage1_ic) == len(self.b) - 1):
                print(colorama.Fore.LIGHTBLUE_EX + 'Using default stage 1 initial conditions')
            else:
                self.stage1_ic = np.zeros(len(self.b) - 1)
                # raise ValueError('Stage 1 Filter initial conditions length does not match filter taps')

        if (audio_sink):
            if (type(stage2_ic) == np.ndarray):
                if (len(stage2_ic) == len(self.bb) - 1):
                    self.stage2_ic = stage2_ic
                else:
                    raise ValueError('Stage 2 Filter initial conditions length does not match filter taps')
            else:
                if (len(self.stage2_ic) == len(self.bb) - 1):
                    print(colorama.Fore.LIGHTBLUE_EX + 'Using default stage 2 initial conditions')
                else:
                    self.stage2_ic = np.zeros(len(self.bb) - 1)
                    # raise ValueError('Stage 2 Filter initial conditions length does not match filter taps')

        if (type(a) == np.ndarray):
            self.a = a

        if (type(aa) == np.ndarray):
            self.aa = aa

        self.audio_sink = audio_sink
        self.rx_data = asyncio.Queue()
        self.clear_buffer()
        self.DSP_IO = pah.DSPIOStream(self._audio_callback, self.audio_in, self.audio_out, self.audio_buffsize,
                                        self.audio_fs, 0, 0)
        self.keep_streaming = True
        self.M1 = m1
        self.M2 = m2
        if (user_var is not None):
            self.user_var = user_var
        if (int(self.fs / self.M1 / self.M2) != int(self.audio_fs) and audio_sink):
            print(colorama.Fore.RED + 'Stage 2 Decimated rate does not match audio sample rate')
            print('\t Decimated Rate: %.2f' % (self.fs / self.M1 / self.M2))
            print('\t Audio Rate: %.2f' % (self.audio_fs))

        self.buffer_exceeded = False

        loop = asyncio.get_event_loop()

        print(colorama.Fore.LIGHTBLUE_EX + 'Starting SDR and Audio Event Loop')
        print(colorama.Fore.BLACK + '')

        if (audio_sink):
            await asyncio.gather(
                self._get_rx_data_user(),
                self._process_rx_data_user(callback),
                self._audio()
            )
        else:
            self.reset_data_out_queue()
            await asyncio.gather(
                self._get_rx_data_user(),
                self._process_rx_data_user(callback),
            )

    def run_user_stream(self, callback, m1, m2, b=False, stage1_ic=False, a=False, bb=False, stage2_ic=False, aa=False,
                        audio_sink=True, user_var=None):
        '''
        Starts a user stream. A user stream follows the flow diagram in the 
        class description. When audio_sink is True, the audio_sink blocks will 
        be used and when audio_sink is False, the data sink block will be used. 
        For any optional parameters set to false, default values will be used 
        for stage 1 or stage 2 filters. The stop() method may be used to stop 
        the stream.

        Parameters:
        -----------
        callback: User-defined processing callback (see example in class 
            description)
        M1: Stage 1 decimation factor - must be >= 1
        M2: Stage 2 decimation factor - must be >= 1
        b: ndarray of stage 1 decimation filter numerator coefficients
        stage1_ic: ndarray of stage 1 decimation filter initial conditions. Must 
            be of length len(b)-1
        a: ndarray of stage 1 decimation filter denominator coefficients
        bb: ndarray of stage 2 decimation filter numerator coefficients
        stage2_ic: ndarray of stage 2 decimation filter initial conditions. Must 
            be of length len(b)-1
        a: ndarray of stage 2 decimation filter numerator coefficients
        audio_sink: When True, the audio sink path is used. When false, the 
            data_sink path is used. (see class definition)
        user_var: Initialization of a user-defined variable that can be used 
            within the user-defined callback. The state of the user-defined 
            variable is maintained within the class

        callback example:
        >>> def callback(samples,fs,user_var):
        >>>     # discriminator
        >>>     x = samples
        >>>     X=np.real(x)
        >>>     Y=np.imag(x)
        >>>     b=np.array([1,-1])
        >>>     a=np.array([1,0])
        >>>     derY=signal.lfilter(b,a,Y)
        >>>     derX=signal.lfilter(b,a,Y)
        >>>     z_bb=(X*derY-Y*derX)/(X**2+Y**2)
        >>>     return z_bb,user_var

        method call:
        >>> sdr_stream = RTLSDRStream()
        >>> run_user_stream(callback,10,5)

        stop streaming:
        >>> sdr_stream.stop()

        '''
        task = asyncio.create_task(
            self._start_user_stream(callback, m1, m2, b, a, stage1_ic, bb, aa, stage2_ic, audio_sink, user_var))

    async def get_data_out_async(self):
        '''
        This method asynchronously returns data from the data_sink buffer when
        it is full. This is used in the data_sink mode (audio_sink=False).

        The following example shows how to continuously stream data and handle 
        the buffer when it is full. The buffer will automatically get rewritten 
        whenever it runs out of space, so the returned buffer must be handled 
        whenever it is filled.

        Async Data Out Example:
        -----------------------
        import asyncio in order to create coroutines and set the data_sink buffer 
        size
        >>> import asyncio
        >>> sdr_stream.set_rtl_buffer_size(16)

        define an data_sink callback that will count the number of frames coming 
        into the radio and store the count in the data_sink buffer
        >>> def no_audio_callback(samples,fs,user_var):
        >>>     frame_count = user_var
        >>>     user_var = user_var+1
        >>>     return np.array([frame_count]),user_var

        create a global variable in order to stop the data_sink buffer processing 
        loop
        >>> global keep_collecting

        create an async function to handle the returned data_sink buffer. Simply 
        print out the buffer for this scenario
        >>> async def handle_data_out():
        >>>     global keep_collecting
        >>>     keep_collecting = True
        >>>     while keep_collecting:
        >>>         data_out = await sdr_stream.get_data_out_async()
        >>>         print(data_out)
        >>>     sdr_stream.reset_data_out_queue()
        >>>     print('Done')

        start a user stream as well as our async data handler coroutine. Should 
        see the data_sink buffer values being printed whenever the buffer is full.
        >>> sdr_stream.run_user_stream(no_audio_callback,1,1,audio_sink=False,user_var=1)
        >>> task = asyncio.create_task(handle_data_out())

        Stop handling data and stop streaming
        >>> keep_collecting = False
        >>> sdr_stream.stop()

        '''
        data_out = await self.data_out_queue.get()
        return data_out

    async def plot_rf(self, nfft=2 ** 10, w=6, h=5):
        '''
        Async method that can be used to plot the PSD of a frame of incoming 
        samples from the SDR. This essentially acts as a power spectrum "probe" 
        right before the Stage 1 decimator. Make sure a stream is running 
        before calling this method. This method must be awaited when called.

        parameters:
        NFFT: Number of points used in the spectrum plot. Should be 2^N value
        w: width of figure
        h: height of figure

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> await sdr_stream.plot_rf(1024,6,5)

        This will return a spectrum plot
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited.')
        self.store_rf = True
        samples = await self.rf_queue.get()
        plt.figure(figsize=(w, h))
        plt.psd(samples, nfft, self.sdr.get_sample_rate(), self.sdr.get_center_freq())
        plt.title('PSD of RF Input')
        plt.show()

    async def plot_stage1(self, nfft=2 ** 10, w=6, h=5):
        '''
        Async method that can be used to plot the PSD of a frame of decimated 
        samples from the SDR. This essentially acts as a power spectrum "probe" 
        after the stage 1 decimator and before the user-defined callback. 
        Make sure a stream is running before calling this method. This method 
        must be awaited when called.

        parameters:
        NFFT: Number of points used in the spectrum plot. Should be 2^N value
        w: width of figure
        h: height of figure

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> await sdr_stream.plot_stage1(1024,6,5)

        This will return a spectrum plot
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited.')
        self.store_stage1 = True
        samples = await self.stage1_queue.get()
        plt.figure(figsize=(w, h))
        plt.psd(samples, nfft, self.sdr.get_sample_rate() / self.M1, self.sdr.get_center_freq())
        plt.title('PSD after Stage 1 Decimation')
        plt.show()

    async def plot_processed_stage1(self, nfft=2 ** 10, fc=0, w=6, h=5):
        '''
        Async method that can be used to plot the PSD of a frame of 
        decimated and processed samples from the SDR. This essentially 
        acts as a power spectrum "probe" after the user-defined callback 
        and before the audio_sink or data_sink blocks.1 decimator and 
        before the user-defined callback. Make sure a stream is running 
        before calling this method. This method must be awaited when 
        called.

        parameters:
        NFFT: Number of points used in the spectrum plot. Should be 2^N value
        FC: Frequency offset for plotting
        w: width of figure
        h: height of figure

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> await sdr_stream.plot_processed_stage1(1024,0,6,5)

        This will return a spectrum plot
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited.')
        self.store_processed_stage1 = True
        samples = await self.processed_stage1_queue.get()
        plt.figure(figsize=(w, h))
        plt.psd(samples, nfft, self.fs / self.M1, fc)
        plt.title('PSD after Processing')
        plt.show()

    async def plot_stage2(self, nfft=2 ** 10, fc=0, w=6, h=5):
        '''
        Async method that can be used to plot the PSD of a frame of data 
        after the stage 2 decimator. This essentially acts as a power 
        spectrum "probe" after the stage2 decimator Make sure a stream is 
        running before calling this method. This method must be awaited 
        when called.

        parameters:
        NFFT: Number of points used in the spectrum plot. Should be 2^N value
        FC: Frequency offset for plotting
        w: width of figure
        h: height of figure

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> await sdr_stream.plot_processed_stage1(1024,0,6,5)

        This will return a spectrum plot
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited.')
        self.store_stage2 = True
        samples = await self.stage2_queue.get()
        plt.figure(figsize=(w, h))
        plt.psd(samples, nfft, self.fs / self.M1 / self.M2, fc)
        plt.title('PSD after Stage 2 Decimation')
        plt.show()

    async def _plot_rf_stream(self, w, h):
        '''
        Private method used to create and update a spectrum analyzer plot of the
        RF input using matplotlib.
        '''
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)

        plt.ion()

        fig.show()
        fig.canvas.draw()

        while (self.update_rf):
            samples = await self.rf_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs)
            f = f + self.fc
            ax.clear()
            ax.grid()
            plt.title('PSD at RF Input')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.xlabel('Frequency (MHz)')
            power = 10 * np.log10(Px * self.fs / self.plot_NFFT)
            f = f / 1e6
            if (self.invert):
                ax.set_facecolor('xkcd:black')
                ax.plot(f, power, 'g')
                ax.plot([self.fc / 1e6, self.fc / 1e6], [-100, 20], '--', color='orange')
            else:
                ax.set_facecolor('xkcd:white')
                ax.plot(f, power)
                ax.plot([self.fc / 1e6, self.fc / 1e6], [-100, 20], '--')
            plt.ylim([np.min(power) - 3, np.max(power) + 3])
            fig.canvas.draw()

        self.update_rf = False

    async def _plot_rf_stream_bokeh(self, w, h):
        '''
        Private method used the create a spectrum analyzer of the RF input using
        a bokeh plot.
        '''
        fig = bfigure(width=w, height=h, title='PSD at RF Input')
        fig.xaxis.axis_label = "Frequency (MHz)"
        fig.yaxis.axis_label = "Power Spectral Density (dB/Hz)"
        samples = await self.rf_queue.get()
        Px, f = psd(samples, self.plot_NFFT, self.fs)
        Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
        f = (f + self.fc) / 1e6
        r = fig.line(f, Px)
        if (self.invert):
            fig.background_fill_color = "Black"
            fig.background_fill_alpha = 0.8
            r.glyph.line_color = "Green"
        else:
            fig.background_fill_color = "White"
            fig.background_fill_alpha = 1.0
            r.glyph.line_color = "Blue"
        fc_line = fig.line(np.array([self.fc / 1e6, self.fc / 1e6]), np.array([np.min(Px) - 2, np.max(Px) + 2]))
        fc_line.glyph.line_color = "Orange"
        fc_line.glyph.line_alpha = 0.5
        fc_line.glyph.line_width = 3
        fc_line.glyph.line_dash = [10, 5]
        target = show(fig, notebook_handle=True)
        while (self.update_rf):
            samples = await self.rf_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs)
            Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
            f = (f + self.fc) / 1e6
            r.data_source.data['x'] = f
            r.data_source.data['y'] = Px
            fc_line.data_source.data['y'] = np.array([np.min(Px) - 2, np.max(Px) + 2])
            fc_line.data_source.data['x'] = np.array([self.fc / 1e6, self.fc / 1e6])
            if (self.invert):
                fig.background_fill_color = "Black"
                fig.background_fill_alpha = 0.8
                r.glyph.line_color = "Green"
            else:
                fig.background_fill_color = "White"
                fig.background_fill_alpha = 1.0
                r.glyph.line_color = "Blue"
            push_notebook(handle=target)

    async def _update_rf_plot(self):
        '''
        Private method used to control the refresh rate of the rf spectrum
        analyzer plot.
        '''
        while (self.update_rf):
            # for i in range(0,10):
            await asyncio.sleep(1.0 / self.refresh_rate)
            self.store_rf = True
        print(colorama.Fore.LIGHTBLUE_EX + 'Stopped RF PSD Stream')

    async def _start_plot_rf_stream(self, nfft, refresh_rate, invert, w, h):
        '''
        Private method used to initialize and start the RF spectrum analyzer.
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited')
        # Stop any other running plots
        self.stop_all_plots()
        self.update_rf = True
        self.refresh_rate = refresh_rate
        self.plot_NFFT = nfft
        self.invert = invert
        loop = asyncio.get_event_loop()

        await asyncio.gather(
            # self._plot_rf_stream_bokeh(w,h),
            self._plot_rf_stream(w, h),
            self._update_rf_plot()
        )

    def run_plot_rf_stream(self, nfft=2 ** 10, refresh_rate=2, invert=True, w=8, h=5):
        '''
        This method can be used to instantiate a spectrum analyzer of the RF input
        during a stream. Call the stop_plot_rf_plot method in order to stop the 
        plot from updating. Only one spectrum analyzer instance my be running at
        once. This only works when using %pylab widget or %pylab notebook

        parameters:
        ----------
        NFFT: fftsize used in plotting
        refresh_rate: defines how often the spectrum analyzer updates (in Hz)
        invert: Inverts the background to black when true or leaves it white when false
        w: width of figure
        h: height of figure

        Example:
        >>> %pylab widget
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> sdr_stream.run_plot_rf_stream(1024,2,True,8,5)

        >>> sdr_stream.stop_rf_plot()
        >>> sdr_stream.stop()

        '''
        task = asyncio.create_task(self._start_plot_rf_stream(nfft, refresh_rate, invert, w, h))

    async def _plot_stage1_stream(self, w, h):
        '''
        Private method used to create and update a spectrum analyzer plot after the
        stage 1 decimator using matplotlib.
        '''
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)
        plt.ion()

        fig.show()
        fig.canvas.draw()

        while (self.update_stage1):
            samples = await self.stage1_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs / self.M1)
            f = f + self.fc
            f = f / 1e3
            ax.clear()
            ax.grid()
            plt.title('PSD after Stage 1')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.xlabel('Frequency (kHz)')
            if (self.invert):
                ax.set_facecolor('xkcd:black')
                ax.plot(f, 10 * np.log10(Px * self.fs / self.plot_NFFT), 'g')
            else:
                ax.set_facecolor('xkcd:white')
                ax.plot(f, 10 * np.log10(Px * self.fs / self.plot_NFFT))
            fig.canvas.draw()

        self.update_stage1 = False

    async def _plot_stage1_stream_bokeh(self, w, h):
        '''
        Private method used the create a spectrum analyzer after the
        stage 1 decimator using a bokeh plot.
        '''
        fig = bfigure(width=w, height=h, title='PSD at RF Input')
        fig.xaxis.axis_label = "Frequency (MHz)"
        fig.yaxis.axis_label = "Power Spectral Density (dB/Hz)"
        samples = await self.rf_queue.get()
        Px, f = psd(samples, self.plot_NFFT, self.fs)
        Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
        f = (f + self.fc) / 1e6
        r = fig.line(f, Px)
        if (self.invert):
            fig.background_fill_color = "Black"
            fig.background_fill_alpha = 0.8
            r.glyph.line_color = "Green"
        else:
            fig.background_fill_color = "White"
            fig.background_fill_alpha = 1.0
            r.glyph.line_color = "Blue"
        self.target1 = show(fig, notebook_handle=True)
        while (self.update_rf):
            samples = await self.rf_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs)
            Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
            f = (f + self.fc) / 1e6
            r.data_source.data['x'] = f
            r.data_source.data['y'] = Px
            if (self.invert):
                fig.background_fill_color = "Black"
                fig.background_fill_alpha = 0.8
                r.glyph.line_color = "Green"
            else:
                fig.background_fill_color = "White"
                fig.background_fill_alpha = 1.0
                r.glyph.line_color = "Blue"
            push_notebook(handle=self.target1)

    async def _update_stage1_plot(self):
        '''
        Private method used to control the refresh rate of the stage 1 spectrum
        analyzer plot.
        '''
        while (self.update_stage1):
            # for i in range(0,10):
            await asyncio.sleep(1.0 / self.refresh_rate)
            self.store_stage1 = True
        print(colorama.Fore.LIGHTBLUE_EX + 'Stopped Stage 1 PSD Stream')

    async def _start_plot_stage1_stream(self, nfft, refresh_rate, invert, w, h):
        '''
        Private method used to initialize and start the stage 1 spectrum analyzer.
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited')
        # Stop any other running plots
        self.stop_all_plots()
        self.update_stage1 = True
        self.refresh_rate = refresh_rate
        self.plot_NFFT = nfft
        self.invert = invert
        loop = asyncio.get_event_loop()

        await asyncio.gather(
            # self.plot_stage1_stream_bokeh(w,h),
            self._plot_stage1_stream(w, h),
            self._update_stage1_plot()
        )

    def run_plot_stage1_stream(self, nfft=2 ** 10, refresh_rate=2, invert=True, w=8, h=5):
        '''
        This method can be used to instantiate a spectrum analyzer after stage 1
        during a stream. Call the stop_plot_rf_plot method in order to stop the 
        plot from updating. Only one spectrum analyzer instance my be running at
        once. This only works when using %pylab widget or %pylab notebook

        parameters:
        ----------
        NFFT: fftsize used in plotting
        refresh_rate: defines how often the spectrum analyzer updates (in Hz)
        invert: Inverts the background to black when true or leaves it white when false
        w: width of figure
        h: height of figure

        Example:
        >>> %pylab widget
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> sdr_stream.run_plot_stage1_stream(1024,2,True,8,5)

        >>> sdr_stream.stop_stage1_plot()
        >>> sdr_stream.stop()

        '''
        task = asyncio.create_task(self._start_plot_stage1_stream(nfft, refresh_rate, invert, w, h))

    async def _plot_processed_stage1_stream(self, w, h):
        '''
        Private method used to create and update a spectrum analyzer plot after the
        callback using matplotlib.
        '''
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)
        plt.ion()

        fig.show()
        fig.canvas.draw()

        while (self.update_processed_stage1):
            samples = await self.processed_stage1_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs / self.M1)
            ax.clear()
            ax.grid()
            plt.title('PSD after Processing')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.xlabel('Frequency')

            if (self.invert):
                ax.set_facecolor('xkcd:black')
                ax.plot(f, 10 * np.log10(Px * self.fs / self.plot_NFFT), 'g')
            else:
                ax.set_facecolor('xkcd:white')
                ax.plot(f, 10 * np.log10(Px * self.fs / self.plot_NFFT))

            fig.canvas.draw()

        self.update_processed_stage1 = False

    async def _plot_processed_stage1_stream_bokeh(self, w, h):
        '''
        Private method used the create a spectrum analyzer after the
        callback using a bokeh plot.
        '''
        fig = bfigure(width=w, height=h, title='PSD after User Callback')
        fig.xaxis.axis_label = "Frequency (Hz)"
        fig.yaxis.axis_label = "Power Spectral Density (dB/Hz)"
        samples = await self.rf_queue.get()
        Px, f = psd(samples, self.plot_NFFT, self.fs)
        Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
        f = (f + self.fc)
        r = fig.line(f, Px)
        if (self.invert):
            fig.background_fill_color = "Black"
            fig.background_fill_alpha = 0.8
            r.glyph.line_color = "Green"
        else:
            fig.background_fill_color = "White"
            fig.background_fill_alpha = 1.0
            r.glyph.line_color = "Blue"
        self.target2 = show(fig, notebook_handle=True)
        while (self.update_rf):
            samples = await
            self.rf_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs)
            Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
            f = (f + self.fc)
            r.data_source.data['x'] = f
            r.data_source.data['y'] = Px
            if (self.invert):
                fig.background_fill_color = "Black"
                fig.background_fill_alpha = 0.8
                r.glyph.line_color = "Green"
            else:
                fig.background_fill_color = "White"
                fig.background_fill_alpha = 1.0
                r.glyph.line_color = "Blue"
            push_notebook(handle=self.target2)

    async def _update_processed_stage1_plot(self):
        '''
        Private method used to control the refresh rate of the callback spectrum
        analyzer plot.
        '''
        while (self.update_processed_stage1):
            # for i in range(0,10):
            await asyncio.sleep(1.0 / self.refresh_rate)
            self.store_processed_stage1 = True
        print(colorama.Fore.LIGHTBLUE_EX + 'Stopped Processed Stage 1 PSD Stream')

    async def _start_plot_processed_stage1_stream(self, nfft, refresh_rate, invert, w, h):
        '''
        Private method used to initialize and start the callback spectrum analyzer.
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited')
        # Stop any other running plots
        self.stop_all_plots()
        self.update_processed_stage1 = True
        self.refresh_rate = refresh_rate
        self.plot_NFFT = nfft
        self.invert = invert
        loop = asyncio.get_event_loop()

        await asyncio.gather(
            # self._plot_processed_stage1_stream_bokeh(w,h),
            self._plot_processed_stage1_stream(w, h),
            self._update_processed_stage1_plot()
        )

    def run_plot_processed_stage1_stream(self, nfft=2 ** 10, refresh_rate=2, invert=True, w=8, h=5):
        '''
        This method can be used to instantiate a spectrum analyzer after the callback
        during a stream. Call the stop_plot_rf_plot method in order to stop the 
        plot from updating. Only one spectrum analyzer instance my be running at
        once. This only works when using %pylab widget or %pylab notebook

        parameters:
        ----------
        NFFT: fftsize used in plotting
        refresh_rate: defines how often the spectrum analyzer updates (in Hz)
        invert: Inverts the background to black when true or leaves it white when false
        w: width of figure
        h: height of figure

        Example:
        >>> %pylab widget
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> sdr_stream.run_plot_processed_stage1_stream(1024,2,True,8,5)

        >>> sdr_stream.stop_processed_stage1_plot()
        >>> sdr_stream.stop()

        '''
        task = asyncio.create_task(self._start_plot_processed_stage1_stream(nfft, refresh_rate, invert, w, h))

    async def _plot_stage2_stream(self, w, h):
        '''
        Private method used to create and update a spectrum analyzer plot after the
        stage 2 decimator using matplotlib.
        '''
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)
        plt.ion()

        fig.show()
        fig.canvas.draw()
        ax.grid()

        while (self.update_stage2):
            samples = await self.stage2_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs / self.M1 / self.M2)
            ax.clear()
            ax.grid()
            plt.title('PSD after Stage 2')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.xlabel('Frequency')
            if (self.invert):
                ax.set_facecolor('xkcd:black')
                ax.plot(f, 10 * np.log10(Px * self.fs / self.plot_NFFT), 'g')
            else:
                ax.set_facecolor('xkcd:white')
                ax.plot(f, 10 * np.log10(Px * self.fs / self.plot_NFFT))
            fig.canvas.draw()

        self.update_stage2 = False

    async def _plot_stage2_stream_bokeh(self, w, h):
        '''
        Private method used the create a spectrum analyzer after the
        stage 2 decimator using a bokeh plot.
        '''
        fig = bfigure(width=w, height=h, title='PSD after Stage 2 Decimation')
        fig.xaxis.axis_label = "Frequency (Hz)"
        fig.yaxis.axis_label = "Power Spectral Density (dB/Hz)"
        samples = await self.rf_queue.get()
        Px, f = psd(samples, self.plot_NFFT, self.fs)
        Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
        f = (f + self.fc)
        r = fig.line(f, Px)
        if (self.invert):
            fig.background_fill_color = "Black"
            fig.background_fill_alpha = 0.8
            r.glyph.line_color = "Green"
        else:
            fig.background_fill_color = "White"
            fig.background_fill_alpha = 1.0
            r.glyph.line_color = "Blue"
        target = show(fig, notebook_handle=True)
        while (self.update_rf):
            samples = await self.rf_queue.get()
            Px, f = psd(samples, self.plot_NFFT, self.fs)
            Px = 10 * np.log10(Px * self.fs / self.plot_NFFT)
            f = (f + self.fc)
            r.data_source.data['x'] = f
            r.data_source.data['y'] = Px
            if (self.invert):
                fig.background_fill_color = "Black"
                fig.background_fill_alpha = 0.8
                r.glyph.line_color = "Green"
            else:
                fig.background_fill_color = "White"
                fig.background_fill_alpha = 1.0
                r.glyph.line_color = "Blue"
            push_notebook(handle=target)

    async def _update_stage2_plot(self):
        '''
        Private method used to control the refresh rate of the stage 2 spectrum
        analyzer plot.
        '''
        while (self.update_stage2):
            # for i in range(0,10):
            await asyncio.sleep(1.0 / self.refresh_rate)
            self.store_stage2 = True
        print(colorama.Fore.LIGHTBLUE_EX + 'Stopped Stage 2 PSD Stream')

    async def _start_plot_stage2_stream(self, NFFT, refresh_rate, invert, w, h):
        '''
        Private method used to initialize and start the stage 2 spectrum analyzer.
        '''
        if (not self.keep_streaming):
            raise RuntimeError('No running stream. Plot cannot be awaited')
        # Stop any other running plots
        self.stop_all_plots()
        self.update_stage2 = True
        self.refresh_rate = refresh_rate
        self.plot_NFFT = NFFT
        self.invert = invert
        loop = asyncio.get_event_loop()

        await asyncio.gather(
            # self._plot_stage2_stream_bokeh(w,h),
            self._plot_stage2_stream(w, h),
            self._update_stage2_plot()
        )

    def run_plot_stage2_stream(self, NFFT=2 ** 10, refresh_rate=2, invert=True, w=8, h=5):
        '''
        This method can be used to instantiate a spectrum analyzer after the callback
        during a stream. Call the stop_plot_rf_plot method in order to stop the 
        plot from updating. Only one spectrum analyzer instance my be running at
        once. This only works when using %pylab widget or %pylab notebook

        parameters:
        ----------
        NFFT: fftsize used in plotting
        refresh_rate: defines how often the spectrum analyzer updates (in Hz)
        invert: Inverts the background to black when true or leaves it white when false
        w: width of figure
        h: height of figure

        Example:
        >>> %pylab widget
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> sdr_stream.run_plot_stage2_stream(1024,2,True,8,5)

        >>> sdr_stream.stop_stage2_plot()
        >>> sdr_stream.stop()

        '''
        task = asyncio.create_task(self._start_plot_stage2_stream(NFFT, refresh_rate, invert, w, h))

    def stop_rf_plot(self):
        '''
        Stops updating an RF spectrum analyzer instance
        '''
        self.update_rf = False

    def stop_stage1_plot(self):
        '''
        Stops updating a stage 1 spectrum analyzer instance
        '''
        self.update_stage1 = False

    def stop_processed_stage1_plot(self):
        '''
        Stops updating a callback spectrum analyzer instance
        '''
        self.update_processed_stage1 = False

    def stop_stage2_plot(self):
        '''
        Stops updating a stage 2 spectrum analyzer instance
        '''
        self.update_stage2 = False

    def stop_all(self):
        '''
        Stops any running spectrum analyzer and stops streaming
        '''
        self.update_rf = False
        self.update_stage1 = False
        self.update_processed_stage1 = False
        self.update_stage2 = False
        self.keep_streaming = False

    def stop_all_plots(self):
        '''
        Stops any running spectrum analyzer
        '''
        self.update_rf = False
        self.update_stage1 = False
        self.update_processed_stage1 = False
        self.update_stage2 = False

    def show_logs(self):
        '''
        Used in Mono FM Receiver example to show logs inside of a widget
        '''
        display(self.output)

    def clear_logs(self):
        '''
        Used in Mono FM Receiver example to clear widget logs
        '''
        self.output.clear_output()

    def _decimate(self, x, m, fs=2.4e6, stage=1):
        '''
        Private method used to decimate a signal for the Mono FM Receiver Example
        '''
        # Filter and decimate (should be polyphase)
        if (stage == 1):
            y, self.stage1_ic = signal.lfilter(self.b, self.a, x, zi=self.stage1_ic)
        else:
            y, self.stage2_ic = signal.lfilter(self.bb, self.a, x, zi=self.stage2_ic)
        z = ss.downsample(y, m)
        return z

    def _discrim(self, x):
        """
        Private method used by Mono FM Receiver example discriminate FM signal

        Mark Wickert
        """
        X = np.real(x)  # X is the real part of the received signal
        Y = np.imag(x)  # Y is the imaginary part of the received signal
        b = np.array([1, -1])  # filter coefficients for discrete derivative
        a = np.array([1, 0])  # filter coefficients for discrete derivative
        derY, self.Y_ic = signal.lfilter(b, a, Y, zi=self.Y_ic)  # derivative of Y, 
        derX, self.X_ic = signal.lfilter(b, a, X, zi=self.X_ic)  # "          X,
        disdata = (X * derY - Y * derX) / (X ** 2 + Y ** 2)
        return disdata

    def reset_data_out_queue(self):
        '''
        Clears data_sink queue
        '''
        del self.data_out_queue
        self.data_out_queue = asyncio.Queue()

    def set_fc_mhz(self, fc):
        '''
        Sets tuning center frequency value (in MHz) on the SDR
        '''
        display(Math(r'f_c:\;%.1f\;\mathrm{MHz}' % fc))
        self.fc = fc * 1e6
        self.sdr.set_center_freq(self.fc)
        with self.output:
            self.output.append_stdout(colorama.Fore.GREEN + 'Changing Center Frequency to: {} MHz\n'.format(fc))

    def set_fc(self, fc):
        '''
        Sets tuning center frequency value (in Hz) on the SDR
        '''
        self.fc = fc
        self.sdr.set_center_freq(fc)
        print(colorama.Fore.YELLOW + "Center Frequency: {}".format(self.sdr.get_center_freq()))

    def set_gain(self, gain):
        '''
        Sets receiver gain value (in dB) on the SDR
        '''
        self.gain = gain
        self.sdr.set_gain(gain)
        print(colorama.Fore.YELLOW + "Gain: {}".format(self.sdr.get_gain()))

    def set_audio_gain_db(self, gain):
        '''
        Sets the audio gain value (in dB) used to scale the audio_sink output volume
        '''
        self.audio_gain = 10 ** (gain / 20)

    def set_fs(self, fs):
        '''
        Sets the sample rate (in samples/second) to the SDR
        This should generally be left at 2.4 Msps. The radio can
        only operate at specific rates.
        '''
        self.fs = fs
        self.sdr.set_sample_rate(fs)
        print(colorama.Fore.YELLOW + "Sample Rate: {}".format(self.sdr.get_sample_rate()))

    def clear_buffer(self):
        '''
        Clears the circular buffer used by the audio sink
        '''
        self.z_out = np.zeros(self.rtl_buffer_size)
        self.rx_idx = 0
        self.audio_idx = int(self.rtl_buffer_size / 2)

    def set_rtl_buffer_size(self, rtl_buffer_size):
        '''
        Sets the circular buffer size used by the audio_sink and the data_sink.
        When the audio_sink is used, this should be set to a fairly high number
        (around 2^15). When the data_sink is used, the buffer size can be changed
        to accommodate the scenario.
        '''
        self.rtl_buffer_size = rtl_buffer_size

    def set_audio_buffsize(self, audio_buffsize):
        '''
        Sets the buffer size used by PyAudio to consume frames processed audio frames
        from the circular buffer. 
        '''
        self.audio_buffsize = audio_buffsize

    def set_audio_fs(self, audio_fs):
        '''
        Sets the audio sample rate. When the audio sink is used this should be equal
        to the radio sample rate (fs) / stage 1 decimation factor / stage 2 decimation
        factor
        '''
        self.audio_fs = audio_fs

    def set_audio_in(self, audio_in):
        '''
        Selects the audio input device. This is not used in the class, but should be
        set to a valid audio input.
        '''
        self.audio_in = audio_in

    def set_audio_out(self, audio_out):
        '''
        Selects the audio output device. Use sk_dsp_comm.rtlsdr_helper.pah.available_devices()
        to get device indices.
        '''
        self.audio_out = audio_out

    def set_audio_gain(self, gain):
        '''
        Sets the audio gain value used to scale the PyAudio volume.
        '''
        self.audio_gain = gain

    def stop(self):
        '''
        Stops a running stream.
        '''
        self.keep_streaming = False

    def set_refresh_rate(self, refresh_rate):
        '''
        Sets the refresh_rate (in Hz) of any running spectrum analyzer
        '''
        self.refresh_rate = refresh_rate

    def set_stage1_coeffs(self, b, a=[1], zi=False):
        '''
        Can be used to set the stage 1 decimation filter coefficients. This can
        be used during an active stream.

        parameters:
        -----------
        b: stage 1 filter numerator coefficients
        a: stage 1 filter denominator coefficients
        zi: stage 1 filter initial conditions
        '''
        if (type(b) == list or type(b) == np.ndarray):
            self.b = b
        else:
            raise ValueError('Numerator coefficient parameter must be list or ndarray type')
        if (type(a) == list or type(a) == np.ndarray):
            self.a = a
        else:
            raise ValueError('Denominator coefficient parameter must be list or ndarray type')
        if (type(zi) == np.ndarray or type(zi) == list):
            if (len(zi) == len(b) - 1):
                self.stage1_ic = zi
            else:
                print(colorama.Fore.RED + 'Initial conditions are not correct length')
                print('Initializing with zero vector')
                self.stage1_ic = np.zeros(len(b) - 1)
        else:
            raise ValueError('Filter initial conditions must be list or ndarray type')

    def set_stage2_coeffs(self, bb, aa=[1], zi=False):
        '''
        Can be used to set the stage 2 decimation filter coefficients. This can
        be used during an active stream.

        parameters:
        -----------
        b: stage 2 filter numerator coefficients
        a: stage 2 filter denominator coefficients
        zi: stage 2 filter initial conditions
        '''
        if (type(bb) == list or type(bb) == np.ndarray):
            self.bb = bb
        else:
            raise ValueError('Numerator coefficient parameter must be list or ndarray type')
        if (type(aa) == list or type(aa) == np.ndarray):
            self.aa = aa
        else:
            raise ValueError('Denominator coefficient parameter must be list or ndarray type')
        if (type(zi) == np.ndarray or type(zi) == list):
            if (len(zi) == len(bb) - 1):
                self.stage2_ic = zi
            else:
                print(colorama.Fore.RED + 'Initial conditions are not correct length')
                print('Initializing with zero vector')
                self.stage2_ic = np.zeros(len(bb) - 1)
        else:
            raise ValueError('Filter initial conditions must be list or ndarray type')

    def set_nfft(self, nfft):
        '''
        Sets the FFT size for any running spectrum analyzer
        '''
        self.plot_NFFT = nfft

    def toggle_invert(self):
        '''
        Toggles between black and white background of a running spectrum analyzer
        '''
        self.invert = not self.invert

    async def get_stage1_frame(self):
        '''
        Async method that can be used to get a frame of decimated data after
        the stage 1 decimation filter.

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> stage1_data_frame = await sdr_stream.get_stage1_frame()
        '''
        self.store_stage1 = True
        samples = await self.stage1_queue.get()
        return samples

    async def get_rf_frame(self):
        '''
        Async method that can be used to get a frame of incoming RF samples.

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> rf_data_frame = await sdr_stream.get_stage1_frame()
        '''
        self.store_rf = True
        samples = await self.rf_queue.get()
        return samples

    async def get_processed_stage1_frame(self):
        '''
        Async method that can be used to get a frame of decimated data after
        the callback.

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> callback_data_frame = await sdr_stream.get_stage1_frame()
        '''
        self.store_processed_stage1 = True
        samples = await self.processed_stage1_queue.get()
        return samples

    async def get_stage2_frame(self):
        '''
        Async method that can be used to get a frame of decimated data after
        the stage 2 decimation filter.

        Example:
        >>> sdr_stream = RTLSDRStream()
        >>> sdr_stream.run_user_stream(callback,10,5)
        >>> stage2_data_frame = await sdr_stream.get_stage2_frame()
        '''
        self.store_stage2 = True
        samples = await self.stage2_queue.get()
        return samples

    def get_center_freq(self):
        '''
        Returns the center frequency of the SDR
        '''
        return self.sdr.get_center_freq()

    def get_gain(self):
        '''
        Returns the receiver gain of the SDR
        '''
        return self.sdr.get_gain()

    def get_sample_rate(self):
        '''
        Returns the sample rate of the SDR
        '''
        return self.sdr.get_sample_rate()

    def get_bandwidth(self):
        '''
        returns the bandwidth of the SDR
        '''
        return self.sdr.get_bandwidth()

    def get_buffer(self):
        '''
        Returns the data_sink buffer at the current index
        '''
        if (self.buffer_exceeded):
            return np.concatenate((self.z_out[self.rx_idx:], self.z_out[:self.rx_idx]))
        else:
            return self.z_out[:self.rx_idx]
