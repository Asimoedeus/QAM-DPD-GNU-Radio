#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: LI
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import default_epy_block_0 as epy_block_0  # embedded python block
import default_epy_block_1 as epy_block_1  # embedded python block
import default_epy_block_1_0 as epy_block_1_0  # embedded python block
import sip
import threading



class test_together(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "test_together")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 4
        self.nfilts = nfilts = 32
        self.excess_bw = excess_bw = 0.35
        self.samp_rate_0 = samp_rate_0 = 32000
        self.samp_rate = samp_rate = 32000
        self.rrc_taps_sync = rrc_taps_sync = firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), excess_bw, 11*sps*nfilts)
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1, sps, 1, excess_bw, 45)
        self.qam16 = qam16 = digital.constellation_16qam().base()
        self.qam16.set_npwr(1.0)

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_const_sink_x_0_0_0_0 = qtgui.const_sink_c(
            1024, #size
            "", #name
            3, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0_0_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0_0_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0_0_0_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_0_0.enable_autoscale(True)
        self.qtgui_const_sink_x_0_0_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(3):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_0_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_0_0_0_win, 3, 2, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.epy_block_1_0 = epy_block_1_0.PA_model_GMP(coeffs=[[1.50+0.01j, 0.05-0.02j],[0.00-0.00j, 0.00+0.00j],[-0.1-0.08j, -0.02-0.01j]])
        self.epy_block_1 = epy_block_1.PA_model_GMP(coeffs=[[1.50+0.01j, 0.05-0.02j],[0.00-0.00j, 0.00+0.00j],[-0.1-0.08j, -0.02-0.01j]])
        self.epy_block_0 = epy_block_0.MP_DPD(coeffs=[[9.138884e-01+3.676318e-02j, 4.883846e-02-9.244402e-02j],[3.472112e-02+1.207297e-01j, -4.867465e-02+3.718652e-02j],[-8.472962e-03-5.038591e-02j, 1.019886e-02+3.677094e-02j],], K=5, Q=1)
        self.digital_pfb_clock_sync_xxx_0_1_0_1 = digital.pfb_clock_sync_ccf(sps, (6.28/100.0), rrc_taps_sync, nfilts, (nfilts/2), 1.5, 1)
        self.digital_pfb_clock_sync_xxx_0_1_0_0 = digital.pfb_clock_sync_ccf(sps, (6.28/100.0), rrc_taps_sync, nfilts, (nfilts/2), 1.5, 1)
        self.digital_pfb_clock_sync_xxx_0_1_0 = digital.pfb_clock_sync_ccf(sps, (6.28/100.0), rrc_taps_sync, nfilts, (nfilts/2), 1.5, 1)
        self.digital_constellation_modulator_0_0_0_0 = digital.generic_mod(
            constellation=qam16,
            differential=False,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.blocks_throttle_0_0_0_0_1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0_0_0_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0_0_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_cc(1.0, 1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 256, 16384))), True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.digital_constellation_modulator_0_0_0_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.blocks_throttle_0_0_0_0, 0))
        self.connect((self.blocks_throttle_0_0_0_0, 0), (self.digital_pfb_clock_sync_xxx_0_1_0, 0))
        self.connect((self.blocks_throttle_0_0_0_0_0, 0), (self.digital_pfb_clock_sync_xxx_0_1_0_0, 0))
        self.connect((self.blocks_throttle_0_0_0_0_1, 0), (self.digital_pfb_clock_sync_xxx_0_1_0_1, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_0, 0), (self.blocks_throttle_0_0_0_0_1, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_0, 0), (self.epy_block_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_0, 0), (self.epy_block_1_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_1_0, 0), (self.qtgui_const_sink_x_0_0_0_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0_1_0_0, 0), (self.qtgui_const_sink_x_0_0_0_0, 1))
        self.connect((self.digital_pfb_clock_sync_xxx_0_1_0_1, 0), (self.qtgui_const_sink_x_0_0_0_0, 2))
        self.connect((self.epy_block_0, 0), (self.epy_block_1, 0))
        self.connect((self.epy_block_1, 0), (self.blocks_multiply_const_xx_0, 0))
        self.connect((self.epy_block_1_0, 0), (self.blocks_throttle_0_0_0_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "test_together")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(1, self.sps, 1, self.excess_bw, 45))
        self.set_rrc_taps_sync(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), self.excess_bw, 11*self.sps*self.nfilts))

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps_sync(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), self.excess_bw, 11*self.sps*self.nfilts))

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.set_rrc_taps(firdes.root_raised_cosine(1, self.sps, 1, self.excess_bw, 45))
        self.set_rrc_taps_sync(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), self.excess_bw, 11*self.sps*self.nfilts))

    def get_samp_rate_0(self):
        return self.samp_rate_0

    def set_samp_rate_0(self, samp_rate_0):
        self.samp_rate_0 = samp_rate_0

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_0_0_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0_0_0_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0_0_0_1.set_sample_rate(self.samp_rate)

    def get_rrc_taps_sync(self):
        return self.rrc_taps_sync

    def set_rrc_taps_sync(self, rrc_taps_sync):
        self.rrc_taps_sync = rrc_taps_sync
        self.digital_pfb_clock_sync_xxx_0_1_0.update_taps(self.rrc_taps_sync)
        self.digital_pfb_clock_sync_xxx_0_1_0_0.update_taps(self.rrc_taps_sync)
        self.digital_pfb_clock_sync_xxx_0_1_0_1.update_taps(self.rrc_taps_sync)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_qam16(self):
        return self.qam16

    def set_qam16(self, qam16):
        self.qam16 = qam16




def main(top_block_cls=test_together, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
