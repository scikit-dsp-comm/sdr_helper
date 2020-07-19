from unittest import TestCase


class TestImports(TestCase):
    _multiprocess_can_split_ = True

    def test_pyaudio_helper_from(self):
        from sk_dsp_comm.sdr_helper import sdr_helper

    def test_pyaudio_helper_import(self):
        import sk_dsp_comm.sdr_helper.sdr_helper

    def test_rtlsdr_from(self):
        from sk_dsp_comm.sdr_helper import rtlsdr

    def test_rtlsdr_import(self):
        import sk_dsp_comm.sdr_helper.rtlsdr
