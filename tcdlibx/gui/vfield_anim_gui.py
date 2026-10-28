"""
GUI actions for the vector-field animation (tcdlibx.graph.vfield_anim), used by
tcdvis (the normal modes come from the fchk).

The host window must provide self.ren (vtkRenderer), self.vtkWidget, the static
molecule in self._actors['mol'], and the method _anim_source(), which opens the
AnimateDialog and returns (cube, atnum, drdq, omega, opts, gif) or None.
"""
import copy

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from tcdlibx.graph.vfield_anim import VFieldAnimator, grab_frame, save_gif


class VFieldAnimationMixin:
    """Start/stop of the moving-arrow animation in a VTK window."""

    def _init_vfield_animation(self):
        self._vfanim = None
        self._vftimer = None
        self._vfstep = 0
        self._vfopts = {}

    def start_vfield_animation(self):
        source = self._anim_source()
        if source is None:
            return
        cube, atnum, drdq, omega, opts, gif = source
        self._vfopts = opts
        self.stop_vfield_animation()
        if getattr(self, '_show_particles', False):
            self.stop_animation()
        try:
            anim = VFieldAnimator(copy.deepcopy(cube), atnum, drdq, omega, opts,
                                  molconf=self._default['molconfig'])
        except ValueError as err:
            QMessageBox.warning(self, 'Animate', str(err))
            return
        if 'mol' in self._actors:
            self.ren.RemoveActor(self._actors['mol'].actor)
        anim.add_to(self.ren)
        self._vfanim = anim
        rwin = self.vtkWidget.GetRenderWindow()
        nframes = anim.opts['nframes']
        if gif:
            frames = []
            for iframe in range(nframes):
                anim.set_phase(self.ren, 2 * np.pi * iframe / nframes)
                rwin.Render()
                frames.append(grab_frame(rwin))
            save_gif(frames, gif, anim.opts['period'])
            print(f'written {gif} ({nframes} frames)')
        self._vfstep = 0
        if self._vftimer is None:
            self._vftimer = QTimer()
            self._vftimer.timeout.connect(self._vfield_tick)
        self._vftimer.start(max(1, int(1000 * anim.opts['period'] / nframes)))
        print(f'animation: largest nuclear displacement {anim.maxdisp:.3f} A, '
              f'cap {anim.cap:.4e}, hidden below {anim.low:.4e}, {anim.nseeds} arrows')

    def _vfield_tick(self):
        if self._vfanim is None:
            return
        nframes = self._vfanim.opts['nframes']
        self._vfanim.set_phase(self.ren, 2 * np.pi * self._vfstep / nframes)
        self._vfstep = (self._vfstep + 1) % nframes
        self.vtkWidget.GetRenderWindow().Render()

    def stop_vfield_animation(self):
        if self._vftimer is not None:
            self._vftimer.stop()
        if self._vfanim is not None:
            self._vfanim.remove_from(self.ren)
            self._vfanim = None
            if 'mol' in self._actors:
                self.ren.AddActor(self._actors['mol'].actor)
            self.vtkWidget.GetRenderWindow().Render()
