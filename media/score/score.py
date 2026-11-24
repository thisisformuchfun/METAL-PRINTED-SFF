#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARROW  --  "Printed bone. Liquid marrow."
Original trailer score for the launch film, composed and rendered entirely in code.

No samples, no external audio, no generative services: every sound below is a
wavetable / additive / noise model written out in numpy, placed by an explicit
note list, mixed through send reverbs made from synthetic impulse responses and
mastered with a gain-only bus chain (so the stems null against the master).

    /opt/venv/bin/python score.py            # render everything + QA
    /opt/venv/bin/python score.py --no-qa    # audio, stems, cues.json only
    /opt/venv/bin/python score.py --only bed # just the 30 s bed

Layout of this file
    1  timebase, pitch, RNG                5  the score: event list + compositions
    2  DSP toolkit                         6  mixer (tracks -> stems, sends, cut points)
    3  instruments (functions)             7  mastering chain + BS.1770 meter
    4  reverb impulse responses            8  export (wav / mp3 / m4a / stems / cues.json)

Grid: 96 BPM, 4/4, 1 bar = 2.5 s = 120 000 samples at 48 kHz, so every bar, beat,
8th and 16th is an integer sample position; cue hits are placed sample-exact.
Key: D minor.  i - VI - III - VII (Dm - Bb - F - C), iv - V into the two big hits.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"          # shared machine: never more than two threads

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.fft as sfft
import scipy.ndimage as ndi
import scipy.signal as sig
import soundfile as sf

# =============================================================================
# 1. TIMEBASE, PITCH, RANDOMNESS
# =============================================================================
SEED = 20261123
SR = 48000
BPM = 96.0
BEAT = 60.0 / BPM            # 0.625 s
BAR = 4.0 * BEAT             # 2.5 s
N_BARS = 32
T_MUSIC = N_BARS * BAR       # 80.0 s
T_TOTAL = 84.0               # with the reverb tail
T_BED = 30.0
FFT_WORKERS = 2
HERE = Path(__file__).resolve().parent

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(p) -> float:
    """'D4' / 'Bb1' / 'C#5' -> MIDI note number (numbers pass through)."""
    if not isinstance(p, str):
        return float(p)
    pc, i = _PC[p[0]], 1
    while p[i] in "#b":
        pc += 1 if p[i] == "#" else -1
        i += 1
    return float(12 * (int(p[i:]) + 1) + pc)


def hz(p) -> float:
    return 440.0 * 2.0 ** ((midi(p) - 69.0) / 12.0)


def bt(bar: float, beat: float = 1.0) -> float:
    """Time in seconds of (1-indexed bar, 1-indexed beat). Exact in binary floats."""
    return ((bar - 1) * 4.0 + (beat - 1.0)) * BEAT


def n_of(t: float) -> int:
    return int(round(t * SR))


def rng_for(*key) -> np.random.Generator:
    """Independent, reproducible random stream per (piece, track, event...)."""
    h = hashlib.sha256(("%d|" % SEED + "|".join(str(k) for k in key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# =============================================================================
# 2. DSP TOOLKIT
# =============================================================================
def dbg(x):
    """dB -> linear gain."""
    return 10.0 ** (np.asarray(x, dtype=float) / 20.0)


def to_db(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


@lru_cache(maxsize=None)
def _butter(kind, f, order):
    return sig.butter(order, f, kind, fs=SR, output="sos")


def hp(x, fc, order=2):
    return sig.sosfilt(_butter("highpass", float(fc), order), x, axis=0)


def lp(x, fc, order=2):
    return sig.sosfilt(_butter("lowpass", float(fc), order), x, axis=0)


def bp(x, lo, hi, order=2):
    return sig.sosfilt(_butter("bandpass", (float(lo), float(hi)), order), x, axis=0)


def zlp(x, fc, order=2):
    """Zero-phase low-pass (for layers that must stay phase-aligned)."""
    return sig.sosfiltfilt(_butter("lowpass", float(fc), order), x, axis=0)


def _rbj(kind, f0, gain_db, q=0.707):
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "peak":
        b = [1 + alpha * a_, -2 * cw, 1 - alpha * a_]
        a = [1 + alpha / a_, -2 * cw, 1 - alpha / a_]
    elif kind == "loshelf":
        s = 2 * np.sqrt(a_) * alpha
        b = [a_ * ((a_ + 1) - (a_ - 1) * cw + s), 2 * a_ * ((a_ - 1) - (a_ + 1) * cw),
             a_ * ((a_ + 1) - (a_ - 1) * cw - s)]
        a = [(a_ + 1) + (a_ - 1) * cw + s, -2 * ((a_ - 1) + (a_ + 1) * cw), (a_ + 1) + (a_ - 1) * cw - s]
    elif kind == "hishelf":
        s = 2 * np.sqrt(a_) * alpha
        b = [a_ * ((a_ + 1) + (a_ - 1) * cw + s), -2 * a_ * ((a_ - 1) + (a_ + 1) * cw),
             a_ * ((a_ + 1) + (a_ - 1) * cw - s)]
        a = [(a_ + 1) - (a_ - 1) * cw + s, 2 * ((a_ - 1) - (a_ + 1) * cw), (a_ + 1) - (a_ - 1) * cw - s]
    else:
        raise ValueError(kind)
    b, a = np.asarray(b) / a[0], np.asarray(a) / a[0]
    return np.concatenate([b, a])


def eq(x, bands):
    """bands: iterable of (kind, f0, gain_dB, Q) with kind in peak / loshelf / hishelf."""
    if not bands:
        return x
    sos = np.vstack([_rbj(k, f, g, q) for (k, f, g, q) in bands])
    return sig.sosfilt(sos, x, axis=0)


def cosfade(k):
    """Raised-cosine ramp 0 -> 1 over k samples (first sample is exactly 0)."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / max(k, 1))


def env_asr(n, atk, rel):
    """Flat-top envelope of n samples with raised-cosine attack and release (ends at 0)."""
    e = np.ones(n)
    a = min(int(atk * SR), n // 2)
    r = min(int(rel * SR), n - a)
    if a > 0:
        e[:a] = cosfade(a)
    if r > 0:
        e[n - r:] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 1) / r)
    return e


def fade_tail(x, sec):
    k = min(len(x), int(sec * SR))
    if k > 0:
        w = 0.5 + 0.5 * np.cos(np.pi * (np.arange(k) + 1) / k)
        x[-k:] = x[-k:] * (w if x.ndim == 1 else w[:, None])
    return x


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def smooth(x, sec):
    """Causal-centred moving average (keeps endpoints)."""
    k = max(1, int(sec * SR))
    return ndi.uniform_filter1d(x, k, axis=0, mode="nearest")


def pan_gains(p):
    """Constant-power pan law, p in [-1 (L) .. +1 (R)]."""
    th = (np.clip(p, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(th), np.sin(th)


def pan(x, p):
    gl, gr = pan_gains(p)
    return np.stack([x * gl, x * gr], axis=1)


def noise(shape, rng):
    return rng.standard_normal(shape)


def band_noise(shape, lo, hi, rng, order=2):
    return bp(rng.standard_normal(shape), lo, hi, order)


def width(x, w):
    """Mid/side width scaling of a stereo buffer."""
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1]) * w
    return np.stack([m + s, m - s], axis=1)


# ---- band-limited wavetables -------------------------------------------------
WT_N = 4096
CR = 64                      # control-rate decimation for LFOs / pitch curves


def _cycle(amps):
    """One cycle with harmonic k at amplitude amps[k-1], sine phase (no aliasing by construction)."""
    spec = np.zeros(WT_N // 2 + 1, dtype=complex)
    spec[1:len(amps) + 1] = -0.5j * np.asarray(amps, dtype=float) * WT_N
    return sfft.irfft(spec, WT_N)


def make_tables(f0, specs, fmax=15000.0, norm="last"):
    """Stack of wavetables (dark -> bright) for a note at f0.

    specs: list of callables amp(freq_hz, k). Harmonics above min(fmax, 0.45*SR)
    are simply never written, so the oscillator is band-limited for any phase
    modulation the note can receive (3 % headroom for detune / vibrato).
    """
    kmax = max(1, min(WT_N // 2 - 1, int(min(fmax, 0.45 * SR) / (f0 * 1.03))))
    k = np.arange(1, kmax + 1, dtype=float)
    f = k * f0
    tabs = np.stack([_cycle(s(f, k)) for s in specs])
    if norm == "last":
        tabs /= np.sqrt(np.mean(tabs[-1] ** 2)) + 1e-12
    elif norm == "each":
        tabs /= np.sqrt(np.mean(tabs ** 2, axis=1, keepdims=True)) + 1e-12
    return np.ascontiguousarray(np.concatenate([tabs, tabs[:, :1]], axis=1))


def wt_play(tabs, ph, pos=None):
    """Read a (K, WT_N+1) table stack at phase `ph` (cycles), morphing along `pos` in [0, K-1]."""
    x = (ph - np.floor(ph)) * WT_N
    i = np.minimum(x.astype(np.intp), WT_N - 1)
    fr = x - i
    K = tabs.shape[0]
    if K == 1 or pos is None:
        tb = tabs[-1]
        a = tb[i]
        return a + (tb[i + 1] - a) * fr
    p = np.clip(np.broadcast_to(np.asarray(pos, dtype=float), ph.shape), 0.0, K - 1 - 1e-9)
    k0 = p.astype(np.intp)
    w = p - k0
    flat = tabs.reshape(-1)
    j = k0 * (WT_N + 1) + i
    a = flat[j]
    a = a + (flat[j + 1] - a) * fr
    j = j + (WT_N + 1)
    b = flat[j]
    b = b + (flat[j + 1] - b) * fr
    return a + (b - a) * w


def _lfo(tc, rng, lo=0.1, hi=1.3, k=3):
    """Slow wander in [-1, 1]: a few incommensurate sines (evaluated at control rate)."""
    out = np.zeros_like(tc)
    tot = 0.0
    for _ in range(k):
        a = rng.uniform(0.5, 1.0)
        out += a * np.sin(2 * np.pi * (rng.uniform(lo, hi) * tc + rng.random()))
        tot += a
    return out / tot


def ensemble(f0, n, tabs, rng, *, voices=6, detune=7.0, drift=4.0, vib_rate=5.6, vib_depth=0.0,
             vib_delay=0.25, vib_fade=0.4, pan_c=0.0, spread=0.5, pos=None, pitch=None,
             amp_jit=0.06, stagger=0.0):
    """A section of `voices` players on one pitch.

    Every voice has its own tuning offset (cents), slow drift, vibrato rate/depth/phase,
    bow-pressure wander, entry time and seat (pan). That de-correlation, not a chorus
    effect, is what gives the ensembles their width while staying mono-compatible.
    `pitch` is an optional control-rate curve in cents shared by all voices (scoops, droop).
    """
    idx = np.arange(n)
    ci = np.arange(0, n + CR, CR)
    tc = ci / SR
    out = np.zeros((n, 2))
    seats = rng.permutation(voices)
    for v in range(voices):
        c = np.full(len(tc), rng.normal(0.0, detune))
        if drift:
            c += drift * _lfo(tc, rng, 0.08, 1.2)
        if vib_depth:
            r = vib_rate * (1.0 + 0.06 * rng.standard_normal())
            d = vib_depth * rng.uniform(0.6, 1.35)
            on = np.clip((tc - vib_delay * rng.uniform(0.6, 1.4)) / vib_fade, 0.0, 1.0)
            wob = 0.12 * np.sin(2 * np.pi * (0.31 * tc + rng.random()))
            c += d * on * np.sin(2 * np.pi * (r * tc + wob + rng.random()))
        if pitch is not None:
            c += pitch
        f = f0 * np.interp(idx, ci, np.exp(c * (np.log(2.0) / 1200.0)))
        ph = np.cumsum(f) / SR + rng.random()
        s = wt_play(tabs, ph, pos)
        if amp_jit:
            s *= np.interp(idx, ci, 1.0 + amp_jit * _lfo(tc, rng, 0.3, 3.0))
        if stagger and v > 0:
            k0 = int(rng.uniform(0.0, stagger) * SR)
            k1 = min(n - k0, int(0.03 * SR))
            s[:k0] = 0.0
            s[k0:k0 + k1] *= cosfade(k1)
        gl, gr = pan_gains(pan_c + spread * ((seats[v] + 0.5) / voices * 2.0 - 1.0))
        out[:, 0] += s * gl
        out[:, 1] += s * gr
    return out / np.sqrt(voices)


def _lpm(f, fc, order=2):
    """Butterworth magnitude, used to draw spectral envelopes directly on the harmonics."""
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))


def _bumps(f, peaks):
    """Sum of log-Gaussian resonances (Hz, dB, octaves): fixed 'body' formants."""
    g = np.zeros_like(f, dtype=float)
    for fp, gdb, bw in peaks:
        g += gdb * np.exp(-0.5 * (np.log2(f / fp) / bw) ** 2)
    return 10.0 ** (g / 20.0)


def saturate_os(x, drive, bias=0.0, os=4):
    """tanh wave-shaper run at 4x oversampling so its harmonics do not fold back."""
    up = sig.resample_poly(x, os, 1, axis=0)
    up = np.tanh(drive * up + bias) - np.tanh(bias)
    return sig.resample_poly(up, 1, os, axis=0)[: len(x)]


# =============================================================================
# 3. INSTRUMENTS
#    Each returns a stereo float64 buffer (n, 2) whose first sample is the note
#    onset. Levels are nominal (about unity for vel = 1); the mixer sets balance.
# =============================================================================

# ---- 3.1 strings ---------------------------------------------------------------
STRINGS = {
    "vln1": dict(pan=-0.50, spread=0.38, fc=(1800.0, 5600.0), voices=7, vib=(5.9, 10.0),
                 body=((290, 3.0, 0.45), (480, 2.0, 0.40), (2600, 3.5, 0.50))),
    "vln2": dict(pan=-0.18, spread=0.38, fc=(1700.0, 5000.0), voices=6, vib=(5.7, 9.0),
                 body=((280, 3.0, 0.45), (460, 2.0, 0.40), (2400, 3.0, 0.50))),
    "vla": dict(pan=0.20, spread=0.36, fc=(1500.0, 4200.0), voices=6, vib=(5.5, 8.0),
                body=((230, 3.0, 0.45), (400, 2.5, 0.40), (1100, 2.0, 0.50))),
    "vc": dict(pan=0.42, spread=0.34, fc=(1200.0, 3600.0), voices=6, vib=(5.3, 7.0),
               body=((95, 2.0, 0.45), (185, 3.0, 0.40), (320, 2.0, 0.45), (1800, 2.0, 0.60))),
    "cb": dict(pan=0.26, spread=0.22, fc=(700.0, 1900.0), voices=4, vib=(5.0, 4.0),
               body=((70, 2.0, 0.45), (115, 3.0, 0.40), (180, 2.0, 0.45))),
}
_SPIC_TAU = {"vln1": 0.060, "vln2": 0.065, "vla": 0.075, "vc": 0.095, "cb": 0.125}


@lru_cache(maxsize=None)
def _string_tabs(sec, m):
    s = STRINGS[sec]
    lo, hi = s["fc"]
    specs = [lambda f, k, fc=fc: (1.0 / k) * _bumps(f, s["body"]) * _lpm(f, fc, 2) * _lpm(f, 11000.0, 2)
             for fc in (lo, math.sqrt(lo * hi), hi)]
    return make_tables(hz(m), specs)


def string_sus(m, dur, vel, sec, rng, atk=0.30, rel=0.55, shape=None, vib=1.0, voices=None):
    """Sustained bowed section: saw-family harmonics under a fixed body curve, brightness
    following the dynamic. `shape` = (start, end) hairpin or [(t, level), ...]."""
    s = STRINGS[sec]
    tabs = _string_tabs(sec, m)
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    env = env_asr(n, atk, rel)
    if shape is not None:
        if np.isscalar(shape[0]):
            env = env * np.interp(t, [0.0, dur], shape)
        else:
            ts, vs = zip(*shape)
            env = env * np.interp(t, ts, vs)
    dyn = env * vel
    pos = (0.12 + 0.88 * np.clip(dyn, 0.0, 1.0) ** 0.7) * (tabs.shape[0] - 1)
    x = ensemble(hz(m), n, tabs, rng, voices=voices or s["voices"], detune=7.5, drift=4.0,
                 vib_rate=s["vib"][0], vib_depth=s["vib"][1] * vib, vib_delay=0.22, vib_fade=0.45,
                 pan_c=s["pan"], spread=s["spread"], pos=pos, amp_jit=0.07,
                 stagger=min(0.045, 0.3 * atk))
    x += band_noise((n, 2), 1800.0, 7000.0, rng) * 0.012          # rosin / bow hair
    return x * dyn[:, None]


def string_spic(m, vel, sec, rng, length=1.0, lock=True):
    """Spiccato / staccato stroke: each desk enters a few ms apart, bright bite that
    darkens within ~70 ms, plus a short bow-noise burst. Player 1 is on the grid."""
    s = STRINGS[sec]
    tabs = _string_tabs(sec, m)
    K = tabs.shape[0]
    f0 = hz(m)
    tau = _SPIC_TAU[sec] * length * (0.85 + 0.30 * vel)
    n = int(min(1.2, tau * 7.5 + 0.03) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    nv = 4 if sec == "cb" else 5
    seats = rng.permutation(nv)
    for v in range(nv):
        off = 0 if (v == 0 and lock) else int(rng.uniform(0.0008, 0.009) * SR)
        tv = t[: n - off]
        e = (1.0 - np.exp(-tv / rng.uniform(0.003, 0.007))) * np.exp(-tv / (tau * rng.uniform(0.85, 1.15)))
        ph = f0 * 2.0 ** (rng.normal(0.0, 6.0) / 1200.0) * tv + rng.random()
        pos = (K - 1) * np.clip((0.30 + 0.70 * vel) * (0.35 + 0.65 * np.exp(-tv / 0.07)), 0.0, 1.0)
        sv = wt_play(tabs, ph, pos) * e
        gl, gr = pan_gains(s["pan"] + s["spread"] * ((seats[v] + 0.5) / nv * 2.0 - 1.0))
        out[off:, 0] += sv * gl
        out[off:, 1] += sv * gr
    out /= np.sqrt(nv)
    nb = min(n, int(0.07 * SR))
    tb = t[:nb]
    bite = band_noise((nb, 2), 1200.0, 8500.0, rng) * ((1.0 - np.exp(-tb / 0.0008)) * np.exp(-tb / 0.016))[:, None]
    gl, gr = pan_gains(s["pan"])
    out[:nb, 0] += bite[:, 0] * gl * 0.7 * (0.4 + 0.6 * vel)
    out[:nb, 1] += bite[:, 1] * gr * 0.7 * (0.4 + 0.6 * vel)
    return fade_tail(out * vel ** 1.25, 0.006)


# ---- 3.2 brass -----------------------------------------------------------------
_BRASS = {
    "horn": dict(fcs=(380.0, 650.0, 1100.0, 1900.0, 3200.0), sq=0.0,
                 body=((340, 2.0, 0.6), (700, 2.0, 0.6), (3200, -3.0, 0.8))),
    "low": dict(fcs=(260.0, 480.0, 900.0, 1700.0, 3000.0), sq=0.15,
                body=((230, 2.5, 0.6), (520, 3.0, 0.5), (1500, 2.0, 0.5))),
    "braam": dict(fcs=(190.0, 420.0, 950.0, 2000.0, 4000.0), sq=0.6,
                  body=((170, 3.0, 0.6), (520, 4.0, 0.5), (1400, 2.5, 0.5))),
}


@lru_cache(maxsize=None)
def _brass_tabs(kind, m):
    b = _BRASS[kind]
    specs = [lambda f, k, fc=fc: (1.0 / k) * (1.0 + b["sq"] * (k % 2)) * _bumps(f, b["body"])
             * _lpm(f, fc, 2) * _lpm(f, 8000.0, 3) for fc in b["fcs"]]
    return make_tables(hz(m), specs)


def brass_note(m, dur, vel, rng, kind="horn", atk=0.07, rel=0.22, pan_c=-0.28, spread=0.30,
               voices=4, shape=None, vib=3.0):
    """Horn / low-brass section note. The table morph is driven by the envelope, so the
    tone opens up ('brassier') as it gets louder - the way a real section behaves."""
    tabs = _brass_tabs(kind, m)
    K = tabs.shape[0]
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    env = env_asr(n, atk, rel)
    env = env * (0.86 + 0.14 * np.sin(np.pi * np.clip(t / max(dur, 1e-3), 0.0, 1.0)) + 0.10 * np.exp(-t / 0.12))
    if shape is not None:
        env = env * np.interp(t, [0.0, dur], shape)
    dyn = np.clip(env * vel, 0.0, 1.2)
    pos = (K - 1) * np.clip(0.08 + 0.95 * dyn ** 1.3, 0.0, 1.0)
    tc = np.arange(0, n + CR, CR) / SR
    x = ensemble(hz(m), n, tabs, rng, voices=voices, detune=5.0, drift=3.0, vib_rate=5.0, vib_depth=vib,
                 vib_delay=0.40, vib_fade=0.5, pan_c=pan_c, spread=spread, pos=pos,
                 pitch=-28.0 * np.exp(-tc / 0.035), amp_jit=0.05, stagger=0.015)
    x += band_noise((n, 2), 350.0, 2600.0, rng) * (0.05 * np.exp(-t / 0.06) + 0.008)[:, None]
    return x * (env * vel)[:, None]


def braam(m_low, dur, vel, rng, rel=1.3, droop=-16.0, bright=1.0, drive=1.15):
    """The 'braam': three octaves of detuned saw/square stacks, pushed through an
    oversampled wave-shaper. The low-pass opens slowly while the pitch sags."""
    T = dur + rel
    n = int(T * SR)
    t = np.arange(n) / SR
    env = np.interp(t, [0.0, 0.025, 0.40, 0.55 * dur, dur, T], [0.0, 0.78, 1.0, 0.86, 0.50, 0.0])
    env = smooth(env, 0.012)
    env[: int(0.004 * SR)] *= cosfade(int(0.004 * SR))
    opening = 0.10 + 0.90 * smoothstep(t / (0.5 * dur))
    closing = 1.0 - 0.65 * smoothstep((t - 0.62 * dur) / (0.38 * dur + rel))
    tc = np.arange(0, n + CR, CR) / SR
    pitch = -40.0 * np.exp(-tc / 0.03) + droop * (tc / T) ** 1.3
    x = np.zeros((n, 2))
    for octv, g, nv in ((0, 0.85, 4), (12, 1.0, 5), (24, 0.50, 4)):
        tabs = _brass_tabs("braam", m_low + octv)
        pos = (tabs.shape[0] - 1) * np.clip(bright * opening * closing * (1.0 - 0.06 * octv / 12.0), 0.0, 1.0)
        x += g * ensemble(hz(m_low + octv), n, tabs, rng, voices=nv, detune=9.0, drift=5.0,
                          pan_c=0.0, spread=0.70, pos=pos, pitch=pitch, amp_jit=0.08)
    x *= env[:, None]
    x = hp(x, 45.0, 2)
    y = saturate_os(x * (drive / 1.45), 1.0, bias=0.12)
    y = 0.8 * y + 0.2 * x / 1.45
    y += band_noise((n, 2), 300.0, 2600.0, rng) * (0.035 * env ** 2)[:, None]
    y = hp(lp(y, 6500.0, 2), 48.0, 2)
    return y * (vel / (np.max(np.abs(y)) + 1e-9))


# ---- 3.3 choir pad --------------------------------------------------------------
_VOWELS = {   # (formant Hz, level dB, bandwidth Hz)  -- 'o' (dark) and 'a' (open)
    "sop": (((450, 0, 70), (800, -11, 80), (2830, -22, 100), (3800, -22, 130), (4950, -50, 135)),
            ((800, 0, 80), (1150, -6, 90), (2900, -32, 120), (3900, -20, 130), (4950, -50, 140))),
    "alt": (((450, 0, 70), (800, -9, 80), (2830, -16, 100), (3500, -28, 130), (4950, -55, 135)),
            ((800, 0, 80), (1150, -4, 90), (2800, -20, 120), (3500, -36, 130), (4950, -60, 140))),
    "ten": (((400, 0, 40), (800, -10, 80), (2600, -12, 100), (2800, -12, 120), (3000, -26, 120)),
            ((650, 0, 80), (1080, -6, 90), (2650, -7, 120), (2900, -8, 130), (3250, -22, 140))),
    "bas": (((400, 0, 40), (750, -11, 80), (2400, -21, 100), (2600, -20, 120), (2900, -40, 120)),
            ((600, 0, 60), (1040, -7, 70), (2250, -9, 110), (2450, -9, 120), (2750, -20, 130))),
}


def _voice_type(m):
    return "sop" if m >= 69 else "alt" if m >= 60 else "ten" if m >= 52 else "bas"


@lru_cache(maxsize=None)
def _choir_tabs(m):
    def spec(formants):
        def fn(f, k):
            a = np.zeros_like(f)
            for i, (fc, lvl, bw) in enumerate(formants):
                soften = 0.0 if i < 2 else -7.0            # restrained: tame the singer's formant
                a += 10.0 ** ((lvl + soften) / 20.0) * (1.0 + ((f - fc) / (1.5 * bw)) ** 2) ** -0.75
            return a + 0.06 / k ** 1.5
        return fn
    return make_tables(hz(m), [spec(v) for v in _VOWELS[_voice_type(m)]], fmax=9000.0, norm="each")


def choir_note(m, dur, vel, rng, atk=0.7, rel=1.1, vowel=0.45, voices=8, shape=None, spread=0.9):
    """Wordless choir: harmonic source drawn under vowel formants ('o' -> 'a'), eight
    singers per pitch with independent vibrato, plus a little breath."""
    tabs = _choir_tabs(m)
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    env = env_asr(n, atk, rel)
    if shape is not None:
        env = env * np.interp(t, [0.0, dur], shape)
    pos = np.clip(vowel + 0.4 * (env * vel - 0.5), 0.0, 1.0)
    x = ensemble(hz(m), n, tabs, rng, voices=voices, detune=11.0, drift=8.0, vib_rate=5.2, vib_depth=13.0,
                 vib_delay=0.35, vib_fade=0.7, pan_c=0.0, spread=spread, pos=pos, amp_jit=0.12, stagger=0.12)
    x += (band_noise((n, 2), 500.0, 1400.0, rng) * 0.6 + band_noise((n, 2), 2300.0, 3600.0, rng) * 0.25) * 0.04
    return x * (env * vel)[:, None]


# ---- 3.4 keys: felt-piano / plucked motif voice ---------------------------------
def piano_note(m, vel, rng, dur=4.5, pan_c=0.0):
    """Additive piano-like tone: stretched (inharmonic) partials, strike-position comb,
    two detuned strings per partial (slow beating), two-stage decay, felt thump."""
    f0 = hz(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    B = 3.0e-4 * (f0 / 261.63) ** 1.3
    k = np.arange(1, 48, dtype=float)
    fk = k * f0 * np.sqrt(1.0 + B * k * k)
    keep = fk < 0.42 * SR
    k, fk = k[keep], fk[keep]
    hard = 1500.0 + 4200.0 * vel
    a = k ** -1.05 * np.exp(-fk / hard) * (0.30 + 0.70 * np.abs(np.sin(np.pi * k / 7.7)))
    tau1 = 2.4 * (261.63 / f0) ** 0.55
    tauk = tau1 / (1.0 + 0.25 * (k - 1.0))
    out = np.zeros((n, 2))
    for i in range(len(k)):
        if a[i] < 1e-4 * a[0]:
            continue
        e = 0.62 * np.exp(-t / (0.22 * tauk[i])) + 0.38 * np.exp(-t / tauk[i])
        w = 2 * np.pi * fk[i] * t
        s = (np.sin(w) + 0.85 * np.sin(w * 1.00055)) * (a[i] * e)
        gl, gr = pan_gains(pan_c + 0.14 * ((i % 3) - 1))
        out[:, 0] += s * gl
        out[:, 1] += s * gr
    out *= (1.0 - np.exp(-t / 0.0015))[:, None]
    nb = int(0.08 * SR)
    th = lp(rng.standard_normal(nb), 900.0, 2) * np.exp(-t[:nb] / 0.014) * (1.0 - np.exp(-t[:nb] / 0.001))
    out /= np.max(np.abs(out)) + 1e-9
    out[:nb] += (0.05 * th)[:, None]
    return fade_tail(out * vel ** 1.4, 0.35)


def ping_pong(x, delay, fb=0.42, mix=0.5, n_taps=7, damp=3200.0):
    """Tempo-locked ping-pong echo built as explicit taps (each repeat a little darker)."""
    d = n_of(delay)
    out = x.copy()
    echo = lp(0.5 * (x[:, 0] + x[:, 1]), damp, 1)
    g = mix
    for i in range(1, n_taps + 1):
        if i * d >= len(x):
            break
        side = i % 2
        out[i * d:, side] += g * echo[: len(x) - i * d]
        out[i * d:, 1 - side] += 0.25 * g * echo[: len(x) - i * d]
        echo = lp(echo, damp, 1)
        g *= fb
    return out


# ---- 3.5 drums ------------------------------------------------------------------
TAIKO = {
    "odaiko": dict(f0=55.00, decay=0.42, bright=0.80, click=0.7),      # A1
    "low": dict(f0=73.42, decay=0.30, bright=0.90, click=0.9),         # D2
    "mid": dict(f0=110.00, decay=0.20, bright=1.00, click=1.0),        # A2
    "high": dict(f0=196.00, decay=0.11, bright=1.20, click=1.2),       # G3
}
_MEMBRANE = (1.000, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.600, 4.060)   # Bessel-zero ratios
_MEMB_AMP = (1.00, 0.60, 0.42, 0.30, 0.26, 0.20, 0.16, 0.13, 0.10, 0.08)


def taiko(kind, vel, rng):
    """Big skin drum: circular-membrane modes whose pitch falls as the head relaxes
    (the classic sine sweep), skin 'slap' noise and a stick click, lightly saturated."""
    p = TAIKO[kind]
    f0 = p["f0"] * (1.0 + 0.006 * rng.standard_normal())
    dec = p["decay"] * (0.75 + 0.35 * vel)
    n = int(min(4.0, dec * 7.0) * SR)
    t = np.arange(n) / SR
    bend = (0.55 + 0.45 * vel) * np.exp(-t / 0.026) + 0.05 * np.exp(-t / 0.22)
    ib = (np.cumsum(bend) - bend[0]) / SR
    x = np.zeros(n)
    for r, a in zip(_MEMBRANE, _MEMB_AMP):
        rr = r * (1.0 + 0.004 * rng.standard_normal())
        amp = a * (p["bright"] * (0.55 + 0.45 * vel)) ** np.log2(r)
        tau = dec / (1.0 + 1.3 * (r - 1.0))
        x += amp * np.sin(2 * np.pi * f0 * rr * (t + (1.0 if r == 1.0 else 0.3) * ib)) * np.exp(-t / tau)
    on = 1.0 - np.exp(-t / 0.00025)
    skin = bp(rng.standard_normal(n), 110.0, 1300.0, 2) * np.exp(-t / (0.035 + 0.02 * vel)) * 1.3
    stick = bp(rng.standard_normal(n), 1700.0, 9000.0, 2) * np.exp(-t / 0.007) * 0.85 * p["click"]
    x = x + (skin + stick * vel ** 0.5) * on * (0.5 + 0.5 * vel)
    x = np.tanh(1.4 * x / 2.2) / np.tanh(1.4)
    return fade_tail(x * vel ** 1.3, 0.05)


def sub_boom(f_end, vel, tau=1.1, sweep=1.5, tau_p=0.045):
    """Sub drop: sine falling onto the chord root (36-65 Hz), exponential decay, soft clip.
    Its phase settles to 2*pi*f_end*(t + sweep*tau_p): the sub-bass is started in step."""
    n = int(min(7.0, tau * 6.5) * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f_end * (t + sweep * tau_p * (1.0 - np.exp(-t / tau_p)))
    x = np.sin(ph) * np.exp(-t / tau)
    x = np.tanh(1.8 * x) / np.tanh(1.8)
    return fade_tail(x * vel, 0.2)


def heart(vel, rng, f=50.0, tau=0.075):
    """One half of the distant 'lub-dub' pulse: a soft, low, click-free thump."""
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * (t + 0.35 * 0.03 * (1.0 - np.exp(-t / 0.03)))
    x = np.sin(ph) * np.exp(-t / tau)
    x += lp(rng.standard_normal(n), 170.0, 2) * np.exp(-t / 0.04) * 0.35
    x *= smoothstep(t / 0.006)
    return fade_tail(lp(x, 240.0, 2) * vel, 0.05)


def tick(vel, rng, f_lo=5200.0, f_hi=11500.0):
    """Machine tick: a few ms of band-limited hiss (the film's 'printer' voice)."""
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    x = band_noise(n, f_lo * rng.uniform(0.9, 1.1), f_hi, rng, 2)
    x *= (1.0 - np.exp(-t / 0.0004)) * np.exp(-t / (0.009 + 0.012 * vel))
    return fade_tail(x * vel, 0.005)


def clack(vel, rng):
    """Stick on the drum rim / frame: two damped wood modes plus a band of noise."""
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    x = band_noise(n, 700.0, 3600.0, rng, 2) * np.exp(-t / 0.014) * 0.8
    for f, a, tau in ((1150.0 * rng.uniform(0.98, 1.02), 0.9, 0.022), (1830.0 * rng.uniform(0.98, 1.02), 0.6, 0.016),
                      (410.0, 0.5, 0.03)):
        x += a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    x *= 1.0 - np.exp(-t / 0.0003)
    return fade_tail(x * vel ** 1.2 * 0.6, 0.01)


# ---- 3.6 low end ----------------------------------------------------------------
@lru_cache(maxsize=None)
def _pulse_tabs(m):
    specs = [lambda f, k, fc=fc: (1.0 / k) * (1.0 + 0.5 * (k % 2)) * _lpm(f, fc, 2) * _lpm(f, 6000.0, 2)
             for fc in (110.0, 240.0, 520.0, 1100.0, 2400.0)]
    return make_tables(hz(m), specs)


def pulse_note(m, vel, rng, open_=0.4, length=1.0):
    """Synth bass pulse: saw/square pluck whose filter snaps shut in ~70 ms."""
    tabs = _pulse_tabs(m)
    K = tabs.shape[0]
    n = int(0.42 * length * SR)
    t = np.arange(n) / SR
    env = (1.0 - np.exp(-t / 0.003)) * np.exp(-t / (0.15 * length))
    pos = (K - 1) * np.clip(open_ * (0.22 + 0.78 * np.exp(-t / 0.07)) * (0.6 + 0.4 * vel), 0.0, 1.0)
    out = np.zeros((n, 2))
    for det, p in ((-4.0, -0.18), (4.5, 0.18)):
        ph = hz(m) * 2.0 ** ((det + rng.normal(0, 1.0)) / 1200.0) * t + rng.random()
        s = wt_play(tabs, ph, pos) * env
        gl, gr = pan_gains(p)
        out[:, 0] += s * gl
        out[:, 1] += s * gr
    return fade_tail(out * (0.7 * vel), 0.02)


def sub_note(m, dur, vel, phase0=0.0, atk=0.03, rel=0.12, shape=None):
    """Sub-bass: sine on the root with a little 2nd/3rd harmonic so it reads on small speakers."""
    f = hz(m)
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * t + phase0
    x = np.sin(ph) + 0.30 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    env = env_asr(n, atk, rel)
    if shape is not None:
        env = env * np.interp(t, [0.0, dur], shape)
    return pan(x * env * vel, 0.0)


@lru_cache(maxsize=None)
def _drone_tabs(m):
    specs = [lambda f, k, fc=fc: (1.0 / k) * _lpm(f, fc, 2) for fc in (140.0, 260.0, 480.0, 900.0)]
    return make_tables(hz(m), specs)


def drone(m, dur, vel, rng, shape=None, rel=1.5):
    """Low D drone: two sub sines beating very slowly, an octave sine, and a dark saw
    layer whose filter breathes. `shape` = [(t, level), ...]."""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    f = hz(m)
    sub = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * (f + 0.07) * t + 1.0)
    octv = 0.45 * np.sin(2 * np.pi * 2 * f * t + 0.5) * (1.0 + 0.25 * np.sin(2 * np.pi * 0.11 * t))
    fifth = 0.10 * np.sin(2 * np.pi * 3 * f * t + 0.2) * (1.0 + 0.5 * np.sin(2 * np.pi * 0.07 * t + 2.0))
    tabs = _drone_tabs(m + 12)
    tc = np.arange(0, n + CR, CR) / SR
    breathe = np.interp(np.arange(n), np.arange(0, n + CR, CR), 0.5 + 0.5 * _lfo(tc, rng, 0.04, 0.22))
    pos = (tabs.shape[0] - 1) * np.clip(0.15 + 0.7 * breathe, 0.0, 1.0)
    saw = ensemble(f * 2, n, tabs, rng, voices=5, detune=6.0, drift=6.0, pan_c=0.0, spread=0.7, pos=pos, amp_jit=0.15)
    x = pan(sub + octv + fifth, 0.0) * 0.55 + saw * 0.32
    env = env_asr(n, 0.5, rel)
    if shape is not None:
        ts, vs = zip(*shape)
        env = env * np.interp(t, ts, vs)
    return x * (env * vel)[:, None]


# ---- 3.7 hits, metal, air -------------------------------------------------------
def impact(vel, rng, size=1.0, dark=False):
    """Trailer hit body: low thud + mid 'chest' + (unless dark) a crack and an airy splash."""
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    on = (1.0 - np.exp(-t / 0.0003))[:, None]
    thud = pan(band_noise(n, 38.0, 170.0, rng, 2) * np.exp(-t / (0.20 * size)), 0.0) * 1.3
    body = band_noise((n, 2), 140.0, 900.0, rng, 2) * np.exp(-t / (0.10 * size))[:, None] * 0.9
    x = thud + body
    if dark:
        x = lp(x, 900.0, 2) * 2.2 + band_noise((n, 2), 250.0, 2400.0, rng, 2) * np.exp(-t / 0.014)[:, None] * 0.9
    else:
        x += band_noise((n, 2), 1800.0, 9500.0, rng, 2) * np.exp(-t / 0.014)[:, None] * 0.55
        x += band_noise((n, 2), 1400.0, 9000.0, rng, 2) * np.exp(-t / 0.28)[:, None] * 0.10
    return fade_tail(x * on * vel, 0.1)


def crash(vel, rng, dur=4.5):
    """Cymbal-ish splash: bands of noise with frequency-dependent decay + a cloud of
    inharmonic partials for the metallic 'clang'."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros((n, 2))
    for lo, hi, tau, g in ((1800.0, 4000.0, 1.10, 0.8), (4000.0, 8000.0, 0.80, 0.8), (8000.0, 15000.0, 0.45, 0.6),
                           (350.0, 1800.0, 0.60, 0.35)):
        x += band_noise((n, 2), lo, hi, rng, 2) * (g * np.exp(-t / tau))[:, None]
    for _ in range(36):
        f = 10.0 ** rng.uniform(np.log10(900.0), np.log10(9000.0))
        s = np.sin(2 * np.pi * f * t + rng.random() * 6.28) * np.exp(-t / rng.uniform(0.4, 1.3)) * 0.035
        gl, gr = pan_gains(rng.uniform(-0.8, 0.8))
        x[:, 0] += s * gl
        x[:, 1] += s * gr
    x *= (1.0 - np.exp(-t / 0.0008))[:, None]
    return fade_tail(x * vel * 0.6, 0.3)


def tamtam(vel, rng, dur=7.0, f_lo=170.0, f_hi=5200.0, n_part=64):
    """Tam-tam: dense inharmonic partials; the highs bloom late and die early."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros((n, 2))
    fs = np.sort(10.0 ** rng.uniform(np.log10(f_lo), np.log10(f_hi), n_part))
    for f in fs:
        u = np.log(f / f_lo) / np.log(f_hi / f_lo)
        ta = 0.012 + 0.75 * u ** 1.5
        td = 3.6 * (1.0 - 0.75 * u)
        a = (1.0 - 0.6 * u) * rng.uniform(0.5, 1.0)
        s = a * np.sin(2 * np.pi * f * t + rng.random() * 6.28) * (1.0 - np.exp(-t / ta)) * np.exp(-t / td)
        gl, gr = pan_gains(rng.uniform(-0.9, 0.9))
        x[:, 0] += s * gl
        x[:, 1] += s * gr
    return fade_tail(x * (vel / np.sqrt(n_part)) * 1.6, 0.5)


def bell(m, vel, rng, dur=8.0):
    """Low bell on the tonic: hum, prime, minor-third 'tierce', quint, nominal, and a minor
    (not major) tenth so it agrees with D minor; each partial is a slowly beating pair."""
    f0 = hz(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros((n, 2))
    parts = ((0.5, 0.60, 6.0), (1.0, 1.00, 5.0), (1.2, 0.80, 4.0), (1.5, 0.45, 3.0), (2.0, 0.90, 2.6),
             (2.4, 0.35, 1.8), (3.01, 0.40, 1.4), (4.07, 0.25, 1.0), (5.2, 0.15, 0.7), (6.3, 0.10, 0.5))
    for r, a, tau in parts:
        f = f0 * r
        s = a * (np.sin(2 * np.pi * f * t) + 0.7 * np.sin(2 * np.pi * (f + rng.uniform(0.15, 0.6)) * t)) * np.exp(-t / tau)
        gl, gr = pan_gains(rng.uniform(-0.5, 0.5))
        x[:, 0] += s * gl
        x[:, 1] += s * gr
    x *= (1.0 - np.exp(-t / 0.002))[:, None]
    x += band_noise((n, 2), 800.0, 4200.0, rng, 2) * (np.exp(-t / 0.03) * 0.5 * (1.0 - np.exp(-t / 0.0005)))[:, None]
    return fade_tail(x * vel * 0.3, 0.4)


@lru_cache(maxsize=None)
def _riser_tabs(m):
    specs = [lambda f, k, fc=fc: (1.0 / k) * _lpm(f, fc, 2) * _lpm(f, 9000.0, 2) for fc in (500.0, 1200.0, 2800.0, 6000.0)]
    return make_tables(hz(m), specs, fmax=12000.0)


def riser(dur, vel, rng, m_from=57.0, m_to=81.0, f_lo=220.0, f_hi=9000.0, curve=3.2):
    """Riser: a noise band swept upward through a filter bank + a detuned tone cluster
    gliding up with accelerating tremolo; gain rises exponentially to the cut."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    gain = (np.exp(curve * u) - 1.0) / (np.exp(curve) - 1.0)
    w = rng.standard_normal((n, 2))
    centre = f_lo * (f_hi / f_lo) ** (u ** 1.25)
    nz = np.zeros((n, 2))
    for fc in np.geomspace(160.0, 13500.0, 15):
        band = bp(w, fc / 1.20, fc * 1.20, 2)
        nz += band * np.exp(-0.5 * (np.log2(fc / centre) / 0.75) ** 2)[:, None]
    nz *= 0.55
    # tone: table chosen for the top of the glide so it stays band-limited throughout
    tabs = _riser_tabs(m_to)
    tc = np.arange(0, n + CR, CR) / SR
    glide = (m_from - m_to) * 100.0 * (1.0 - (tc / dur) ** 1.6)
    pos = (tabs.shape[0] - 1) * np.clip(0.15 + 0.85 * u ** 1.5, 0.0, 1.0)
    tone = ensemble(hz(m_to), n, tabs, rng, voices=7, detune=14.0, drift=10.0, pan_c=0.0, spread=0.9,
                    pos=pos, pitch=glide, amp_jit=0.1)
    trem_rate = 3.0 + 13.0 * u ** 2
    trem = 1.0 - 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(trem_rate) / SR)) * u
    x = (nz + tone * 0.30 * trem[:, None]) * gain[:, None]
    k = int(0.004 * SR)
    x[-k:] *= (1.0 - cosfade(k))[:, None]
    return x * vel


def swell(dur, vel, rng, lo=500.0, hi=7000.0, curve=4.0):
    """Reverse-style swell into a downbeat (filtered noise with exponential rise, hard stop)."""
    n = int(dur * SR)
    u = np.arange(n) / n
    x = band_noise((n, 2), lo, hi, rng, 2)
    x = x * ((np.exp(curve * u) - 1.0) / (np.exp(curve) - 1.0))[:, None]
    k = int(0.008 * SR)
    x[-k:] *= (1.0 - cosfade(k))[:, None]
    return x * vel * 0.5


def rev_crash(dur, vel, rng):
    """A crash played backwards: the classic suction into a hit."""
    x = crash(1.0, rng, dur=dur)[::-1].copy()
    k = int(0.008 * SR)
    x[-k:] *= (1.0 - cosfade(k))[:, None]
    x[: int(0.3 * SR)] *= cosfade(int(0.3 * SR))[:, None]
    return x * vel


def air(dur, vel, rng):
    """Room tone / breath: band-limited noise whose level and colour wander slowly."""
    n = int(dur * SR)
    idx = np.arange(n)
    ci = np.arange(0, n + CR, CR)
    tc = ci / SR
    hi_ = band_noise((n, 2), 2200.0, 9000.0, rng, 1) * np.interp(idx, ci, 0.55 + 0.45 * _lfo(tc, rng, 0.05, 0.35))[:, None]
    mid = band_noise((n, 2), 300.0, 1200.0, rng, 2) * np.interp(idx, ci, 0.5 + 0.5 * _lfo(tc, rng, 0.04, 0.25))[:, None]
    x = hi_ * 0.6 + mid * 0.5
    return x * (env_asr(n, min(2.0, dur / 3), min(2.0, dur / 3)) * vel)[:, None]


def droplet(m, vel, rng, pan_c=0.0):
    """Liquid ping: a sine that chirps upward as it rings (drop into a cavity) + a glassy partial."""
    f = hz(m)
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * (t + 0.30 * (t - 0.02 * (1.0 - np.exp(-t / 0.02))))     # f -> 1.3 f in ~40 ms
    x = np.sin(ph) * np.exp(-t / 0.16) + 0.25 * np.sin(2.76 * ph) * np.exp(-t / 0.06)
    x *= 1.0 - np.exp(-t / 0.0012)
    return fade_tail(pan(x * vel * 0.5, pan_c), 0.1)


# =============================================================================
# 4. REVERB: SYNTHETIC IMPULSE RESPONSES
#    Decaying noise with a frequency-dependent RT60, a diffusion 'bloom', a few
#    early reflections and a high-pass (the low end of the mix stays dry).
# =============================================================================
IR_SPECS = {
    "hall": dict(rt60={125: 3.6, 250: 4.0, 500: 4.4, 1000: 4.3, 2000: 3.7, 4000: 2.7, 8000: 1.6, 16000: 0.9},
                 dur=6.0, predelay=0.028, bloom=0.030, hp=170.0, er=0.05, er_span=0.085),
    "room": dict(rt60={125: 1.5, 250: 1.4, 500: 1.3, 1000: 1.2, 2000: 1.1, 4000: 0.9, 8000: 0.6, 16000: 0.4},
                 dur=2.4, predelay=0.006, bloom=0.005, hp=110.0, er=0.22, er_span=0.045),
}


@lru_cache(maxsize=None)
def get_ir(name):
    sp = IR_SPECS[name]
    rng = rng_for("ir", name)
    n = int(sp["dur"] * SR)
    t = np.arange(n) / SR
    W = sfft.rfft(rng.standard_normal((n, 2)), axis=0)
    lf = np.log2(np.maximum(sfft.rfftfreq(n, 1.0 / SR), 1.0))
    fcs = sorted(sp["rt60"])
    cs = np.log2(np.array(fcs, dtype=float))
    late = np.zeros((n, 2))
    for i, c in enumerate(cs):                       # triangular (in log f) partition of unity
        w = np.zeros_like(lf)
        if i == 0:
            w[lf <= c] = 1.0
        else:
            m = (lf > cs[i - 1]) & (lf <= c)
            w[m] = (lf[m] - cs[i - 1]) / (c - cs[i - 1])
        if i == len(cs) - 1:
            w[lf > c] = 1.0
        else:
            m = (lf > c) & (lf <= cs[i + 1])
            w[m] = (cs[i + 1] - lf[m]) / (cs[i + 1] - c)
        band = sfft.irfft(W * w[:, None], n, axis=0)
        late += band * np.exp(-6.9078 * t / sp["rt60"][fcs[i]])[:, None]
    late *= (1.0 - np.exp(-t / sp["bloom"]))[:, None]
    late /= np.sqrt(np.sum(late ** 2) / 2.0)
    er = np.zeros((n, 2))
    for ch in range(2):
        for _ in range(14):
            d = rng.uniform(0.004, sp["er_span"])
            er[int(d * SR), ch] += rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.0) / (1.0 + d / 0.02)
    er = lp(er, 5500.0, 1)
    er *= math.sqrt(sp["er"]) / np.sqrt(np.sum(er ** 2) / 2.0)
    ir = late * math.sqrt(1.0 - sp["er"]) + er
    ir = np.concatenate([np.zeros((int(sp["predelay"] * SR), 2)), ir])[:n]
    return fade_tail(hp(ir, sp["hp"], 2), 0.15 * sp["dur"])


_IR_FFT: dict = {}


def reverb(send, name, n_out):
    """FFT convolution of a stereo send with a stereo IR (unit energy per channel)."""
    ir = get_ir(name)
    nfft = sfft.next_fast_len(send.shape[0] + ir.shape[0] - 1, real=True)
    key = (name, nfft)
    if key not in _IR_FFT:
        _IR_FFT[key] = sfft.rfft(ir, nfft, axis=0, workers=FFT_WORKERS)
    x = np.stack([0.8 * send[:, 0] + 0.2 * send[:, 1], 0.2 * send[:, 0] + 0.8 * send[:, 1]], axis=1)
    y = sfft.irfft(sfft.rfft(x, nfft, axis=0, workers=FFT_WORKERS) * _IR_FFT[key], nfft, axis=0, workers=FFT_WORKERS)
    return y[:n_out]


# =============================================================================
# 5. THE SCORE
# =============================================================================
@dataclass
class Ev:
    track: str
    t: float                 # onset, seconds
    dur: float = 0.0         # nominal length for sustained voices
    pitch: float = 0.0       # MIDI note number
    vel: float = 1.0
    lock: bool = False       # on the cue grid: never humanised, player 1 sample-exact
    kw: dict = field(default_factory=dict)
    idx: int = 0             # running index on its track (seeds its random stream)


class Score:
    def __init__(self, name, length, bounds, gaps=()):
        self.name, self.length = name, float(length)
        self.bounds = tuple(bounds)          # hard cut points: nothing rings across them
        self.gaps = tuple(gaps)              # (start, end) of the silences before the cuts
        self.events: list[Ev] = []
        self.auto: dict[str, list] = {}      # track -> [(t, dB), ...]
        self.ducks: list[tuple] = []         # (t, amount) of big hits, for side-chain ducking
        self.cues: list[dict] = []
        self.sections: list[dict] = []
        self.chords: dict[int, str] = {}
        self._n: dict[str, int] = {}

    def add(self, track, t, dur=0.0, pitch=0.0, vel=1.0, lock=False, **kw):
        i = self._n.get(track, 0)
        self._n[track] = i + 1
        self.events.append(Ev(track, float(t), float(dur), midi(pitch), float(vel), bool(lock), kw, i))

    def cue(self, t, kind, label, strength=1.0):
        self.cues.append(dict(t=float(t), kind=kind, label=label, strength=round(float(strength), 3)))

    def section(self, name, bar0, bar1, label):
        self.sections.append(dict(name=name, bars=[bar0, bar1], start=bt(bar0), end=bt(bar1 + 1), label=label))


# Voicings. Upper parts move by step or hold common tones (A-Bb-A-G on top of
# i-VI-III-VII); the ostinato keeps D/F or C as pivots so only one note changes
# per chord:  D-A | D-Bb | C-A | C-G.
CHORD = {
    "Dm": dict(sub="D1", pulse="D2", chug=("D2", "D2"), low=("D2", "D3", "A3"), braam="D1",
               ost=("D3", "A3", "D4", "A3", "F3", "A3", "D4", "A3"), pad=("A3", "D4", "F4"),
               pad_up=("D4", "F4", "A4"), choir=("A4", "D5", "F5"), choir_up=("D5", "F5", "A5"), men=("D3", "A3")),
    "Bb": dict(sub="Bb1", pulse="Bb1", chug=("Bb1", "Bb2"), low=("Bb1", "Bb2", "F3"), braam="Bb0",
               ost=("D3", "Bb3", "D4", "Bb3", "F3", "Bb3", "D4", "Bb3"), pad=("Bb3", "D4", "F4"),
               pad_up=("D4", "F4", "Bb4"), choir=("Bb4", "D5", "F5"), choir_up=("D5", "F5", "Bb5"), men=("Bb2", "F3")),
    "F": dict(sub="F1", pulse="F2", chug=("F2", "F2"), low=("F2", "F3", "C4"), braam="F1",
              ost=("C3", "A3", "C4", "A3", "F3", "A3", "C4", "A3"), pad=("A3", "C4", "F4"),
              pad_up=("C4", "F4", "A4"), choir=("A4", "C5", "F5"), choir_up=("C5", "F5", "A5"), men=("F3", "C4")),
    "C": dict(sub="C2", pulse="C2", chug=("C2", "C3"), low=("C2", "C3", "G3"), braam="C1",
              ost=("C3", "G3", "C4", "G3", "E3", "G3", "C4", "G3"), pad=("G3", "C4", "E4"),
              pad_up=("C4", "E4", "G4"), choir=("G4", "C5", "E5"), choir_up=("C5", "E5", "G5"), men=("C3", "G3")),
    "Gm": dict(sub="G1", pulse="G2", chug=("G1", "G2"), low=("G1", "G2", "D3"), braam="G1",
               ost=("D3", "Bb3", "D4", "Bb3", "G3", "Bb3", "D4", "Bb3"), pad=("Bb3", "D4", "G4"),
               pad_up=("Bb3", "D4", "G4"), choir=("Bb4", "D5", "G5"), choir_up=("D5", "G5", "Bb5"), men=("G2", "D3")),
    # V: bars 14 and 30 hold the D of iv as a 4-3 suspension (Asus4 -> A) before the hit
    "A": dict(sub="A1", pulse="A2", chug=("A1", "A2"), low=("A1", "A2", "E3"), braam="A1",
              ost=("E3", "A3", "D4", "A3", "E3", "C#4", "E4", "C#4"), pad=("A3", "C#4", "E4"),
              pad_up=("A3", "C#4", "E4"), choir=("A4", "C#5", "E5"), choir_up=("C#5", "E5", "A5"), men=("A2", "E3")),
}
PROG = {1: "D5", 2: "D5", 3: "D5", 4: "Dm", 5: "Dm", 6: "Bb/D", 7: "Bb/D", 8: "C/D",
        9: "Dm", 10: "Dm", 11: "Bb", 12: "Bb", 13: "Gm", 14: "A",
        15: "Dm", 16: "Dm", 17: "Bb", 18: "Bb", 19: "F", 20: "F", 21: "C", 22: "C", 23: "Bb", 24: "C",
        25: "Dm", 26: "Bb", 27: "F", 28: "C", 29: "Gm", 30: "A", 31: "Dm", 32: "D5"}

# The theme: (bar, beat, length in beats, pitch, dynamic). Concert pitch of the violins; horns and
# celli sound an octave lower. Four 2-bar phrases - A (D E F A), its answer B (Bb A F D), A a third
# higher, then the peak: D5 as a 9-8 suspension over C that resolves on beat 3 - and a rising turnaround.
THEME = ((15, 1, 3, "D4", 0.90), (15, 4, 1, "E4", 0.80), (16, 1, 2, "F4", 0.90), (16, 3, 2, "A4", 1.00),
         (17, 1, 3, "Bb4", 1.00), (17, 4, 1, "A4", 0.85), (18, 1, 2, "F4", 0.90), (18, 3, 2, "D4", 0.82),
         (19, 1, 3, "F4", 0.92), (19, 4, 1, "G4", 0.85), (20, 1, 2, "A4", 0.95), (20, 3, 2, "C5", 1.00),
         (21, 1, 2, "D5", 1.05), (21, 3, 2, "C5", 0.95), (22, 1, 2, "G4", 0.92), (22, 3, 2, "E4", 0.85),
         (23, 1, 1, "F4", 0.85), (23, 2, 1, "Bb4", 0.90), (23, 3, 1, "D5", 0.96), (23, 4, 1, "F5", 1.00),
         (24, 1, 4, "E5", 1.00))
# second lift: the same four phrases in diminution, an octave up, then iv - V
LIFT = ((25, 1, 1.5, "D5", 1.00), (25, 2.5, 0.5, "E5", 0.90), (25, 3, 1, "F5", 0.95), (25, 4, 1, "A5", 1.00),
        (26, 1, 1.5, "Bb5", 1.05), (26, 2.5, 0.5, "A5", 0.90), (26, 3, 1, "F5", 0.95), (26, 4, 1, "D5", 0.90),
        (27, 1, 1.5, "F5", 0.98), (27, 2.5, 0.5, "G5", 0.90), (27, 3, 1, "A5", 1.00), (27, 4, 1, "C6", 1.05),
        (28, 1, 1.5, "D6", 1.10), (28, 2.5, 0.5, "C6", 0.95), (28, 3, 1, "G5", 0.98), (28, 4, 1, "E5", 0.92),
        (29, 1, 2, "G5", 1.00), (29, 3, 2, "Bb5", 1.05), (30, 1, 4, "A5", 1.00))
MOTIF_RHYTHM = ((0, 3.0), (0, 4.5), (1, 2.0), (1, 3.5))       # dotted-quarter lilt across two bars
MOTIF_A = ("D5", "E5", "F5", "A5")                            # the four notes; also the head of the theme
MOTIF_B = ("Bb5", "A5", "F5", "D5")
ACC_STRAIGHT = (1.0, .50, .68, .50, .86, .50, .68, .52, .94, .50, .68, .50, .86, .52, .70, .56)
ACC_332 = tuple(1.0 if i in (0, 3, 6, 8, 11, 14) else 0.55 for i in range(16))
ACC_8 = (1.0, .55, .75, .55, .90, .55, .75, .60)
ACC_8_332 = (1.0, .60, .60, .95, .60, .60, .90, .65)
BOOM_SWEEP, BOOM_TAUP = 1.5, 0.045
PRE = 0.045            # swells stop this long before the hit they lead into
ENS = {   # taiko ensemble layers: (drum, pan, flam offset s, level). Layer 1 is exactly on the grid.
    "big": (("odaiko", 0.0, 0.0, 1.0), ("low", -0.50, 0.004, 0.90), ("low", 0.50, 0.009, 0.85),
            ("mid", -0.25, 0.006, 0.70), ("mid", 0.30, 0.012, 0.70), ("high", 0.60, 0.003, 0.50)),
    "med": (("odaiko", 0.0, 0.0, 0.90), ("low", -0.45, 0.005, 0.85), ("mid", 0.35, 0.009, 0.70)),
    "small": (("low", -0.20, 0.0, 0.90), ("mid", 0.30, 0.007, 0.60)),
}


class _Builder:
    """Shared phrase generators for the theme and the bed."""

    def __init__(self, S: Score, cue_bars=()):
        self.S, self.cue_bars = S, set(cue_bars)

    # -- drums ---------------------------------------------------------------
    def ens(self, t, vel, size="med", lock=True):
        for kind, p, dt, g in ENS[size]:
            self.S.add("taiko", t + dt, 0, 0, vel * g, lock=lock, kind=kind, pan=p)

    def hit(self, t, lvl, root, size="med", tau=0.6, crash_=0.0, imp=0.5, bell_=0.0, tam=0.0, duck=True):
        S = self.S
        self.ens(t, lvl, size)
        S.add("boom", t, 0, root, lvl, lock=True, tau=tau)
        if imp:
            S.add("impact", t, 0, 0, lvl * imp, lock=True)
        if crash_:
            S.add("crash", t, 0, 0, lvl * crash_, lock=True)
        if bell_:
            S.add("bell", t, 0, "D3", lvl * bell_, lock=True)
        if tam:
            S.add("tamtam", t, 0, 0, lvl * tam, lock=True)
        if duck:
            S.ducks.append((t, lvl))

    def soft_hit(self, t, lvl, root="D1", tau=1.0):
        S = self.S
        S.add("boom", t, 0, root, lvl, lock=True, tau=tau)
        S.add("impact", t, 0, 0, lvl * 0.9, lock=True, dark=True, size=1.3)
        S.ducks.append((t, 0.5 * lvl))

    def drums(self, bar, pat, level=1.0):
        for st, kind, vel, p in pat:
            self.S.add("taiko", bt(bar) + st * BEAT / 4, 0, 0, vel * level, kind=kind, pan=p)

    def ticks(self, bar, level, div=16, acc=ACC_STRAIGHT):
        for st in range(0, 16, 16 // div):
            self.S.add("tick", bt(bar) + st * BEAT / 4, 0, 0, level * acc[st], pan=(-0.55 if (st // (16 // div)) % 2 else 0.55))

    def clacks(self, bar, steps, level):
        for i, st in enumerate(steps):
            self.S.add("clack", bt(bar) + st * BEAT / 4, 0, 0, level * (1.0 if st % 4 == 0 else 0.8), pan=(-0.3, 0.35)[i % 2])

    # -- strings -------------------------------------------------------------
    def ost(self, track, sec, bar, chord, level, div=8, octave=0, acc=ACC_STRAIGHT, cresc=(1.0, 1.0), last_step=15):
        notes = CHORD[chord]["ost"]
        for st in range(0, 16, 16 // div):
            if st > last_step:
                break
            v = level * acc[st] * (cresc[0] + (cresc[1] - cresc[0]) * st / 16.0)
            self.S.add(track, bt(bar) + st * BEAT / 4, 0, midi(notes[st // 2]) + octave, v,
                       lock=(st == 0 and bar in self.cue_bars), sec=sec)

    def chug(self, bar, chord, level, div=8, acc=ACC_8_332, cresc=(1.0, 1.0), last_step=15):
        lo, hi = CHORD[chord]["chug"]
        for i, st in enumerate(range(0, 16, 16 // div)):
            if st > last_step:
                break
            a = acc[i] if len(acc) == div else acc[st]
            v = level * a * (cresc[0] + (cresc[1] - cresc[0]) * st / 16.0)
            lock = st == 0 and bar in self.cue_bars
            self.S.add("chug", bt(bar) + st * BEAT / 4, 0, lo, v, lock=lock, sec="cb", length=0.8)
            self.S.add("chug", bt(bar) + st * BEAT / 4, 0, hi, v * 0.9, lock=lock, sec="vc", length=0.9)

    def sus(self, track, bar, nbars, parts, vel, t_off=0.0, **kw):
        """parts: iterable of (section, pitch)."""
        for sec, p in parts:
            self.S.add(track, bt(bar) + t_off, nbars * BAR - t_off, p, vel, sec=sec, **kw)

    # -- bass ----------------------------------------------------------------
    def pulse(self, bar, pitch, level, open_, div=8, acc=ACC_8, last_step=15):
        for i, st in enumerate(range(0, 16, 16 // div)):
            if st > last_step:
                break
            a = acc[i] if len(acc) == div else acc[st]
            self.S.add("pulse", bt(bar) + st * BEAT / 4, 0, pitch, level * a, open=open_,
                       lock=(st == 0 and bar in self.cue_bars), length=1.0 if div == 8 else 0.7)

    def sub(self, bar, nbars, pitch, vel, boom=False, **kw):
        ph = 2 * np.pi * hz(pitch) * BOOM_SWEEP * BOOM_TAUP if boom else 0.0    # in step with the sub drop
        self.S.add("sub", bt(bar), nbars * BAR, pitch, vel, lock=True, phase0=ph, **kw)

    # -- motif ---------------------------------------------------------------
    def motif(self, bar, notes, vel, shadow=0.45):
        for (db, beat), p, v in zip(MOTIF_RHYTHM, notes, (1.0, 0.82, 0.92, 1.0)):
            self.S.add("piano", bt(bar + db, beat), 0, p, vel * v)
            if shadow:
                self.S.add("piano", bt(bar + db, beat) + 0.005, 0, midi(p) - 12, vel * v * shadow)


def spans(bar0, bar1):
    """Merge consecutive bars that share a chord: [(first_bar, n_bars, chord), ...]."""
    out = []
    for b in range(bar0, bar1 + 1):
        if out and out[-1][2] == PROG[b] and out[-1][0] + out[-1][1] == b:
            out[-1][1] += 1
        else:
            out.append([b, 1, PROG[b]])
    return [tuple(o) for o in out]


def compose_theme() -> Score:
    """MARROW main theme: 32 bars at 96 BPM (80.0 s) + tail to 84 s."""
    GAP1, GAP2 = (34.85, 35.0), (74.88, 75.0)
    S = Score("theme", T_TOTAL, bounds=(0.0, 35.0, 75.0, T_TOTAL), gaps=(GAP1, GAP2))
    S.chords = dict(PROG)
    B = _Builder(S, cue_bars=(1, 4, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31))
    for name, b0, b1, label in (("I", 1, 3, "near-silence, low D drone, distant pulse, air"),
                                ("II", 4, 8, "pulse establishes, four-note motif, low strings"),
                                ("III", 9, 14, "build: ostinato, growing percussion, riser"),
                                ("IV", 15, 24, "the drop: full theme, braams, choir"),
                                ("V", 25, 30, "second lift: octave doubling, full-time taikos"),
                                ("VI", 31, 32, "final hit rings out (tail to 84 s)")):
        S.section(name, b0, b1, label)

    # ---------------------------------------------------------------- I  bars 1-3
    S.cue(0.0, "sub_hit", "soft sub hit, drone begins", 0.35)
    S.add("boom", 0.0, 0, "D1", 0.36, lock=True, tau=1.2)
    S.add("impact", 0.0, 0, 0, 0.28, lock=True, dark=True, size=1.3)
    S.add("drone", 0.0, 20.0, "D1", 1.0, rel=1.5,
          shape=[(0.0, 0.0), (0.8, 0.16), (7.5, 0.60), (12.5, 0.75), (20.0, 0.95), (21.5, 0.0)])
    S.add("air", 0.0, 21.0, 0, 1.0)
    for bar in range(2, 9):                                   # distant lub-dub, walking closer
        for beat in (1, 3):
            v = float(np.interp(bar + (beat - 1) / 4, [2, 4, 8.75], [0.22, 0.45, 0.70]))
            S.add("heart", bt(bar, beat), 0, 0, v, f=50.0, lock=(beat == 1 and bar in B.cue_bars))
            S.add("heart", bt(bar, beat + 0.5), 0, 0, v * 0.66, f=58.0, tau=0.06)
    for t, p, v, pn in ((3.93, "A6", 0.50, -0.5), (5.47, "D7", 0.40, 0.45), (6.70, "F6", 0.55, 0.1),
                        (9.21, "A6", 0.45, 0.5), (11.02, "E7", 0.35, -0.4), (14.36, "D7", 0.45, -0.2),
                        (16.05, "A6", 0.40, 0.55), (18.60, "F7", 0.35, -0.55)):
        S.add("drop", t, 0, p, v, pan=pn)

    # --------------------------------------------------------------- II  bars 4-8
    # harmony moves over a D pedal: Dm | Dm | Bb/D | Bb/D | C/D
    S.cue(7.5, "soft_impact", "soft impact, pulse and motif enter", 0.45)
    S.cue(12.5, "soft_impact", "soft impact, harmony opens to Bb/D", 0.5)
    B.soft_hit(7.5, 0.46)
    B.soft_hit(12.5, 0.68)
    S.add("swell", 12.5 - 1.25, 1.25 - 2 * PRE, 0, 0.12, lo=400.0, hi=3500.0)
    for bar in range(4, 9):
        u = (bar - 4) / 4.0
        B.pulse(bar, "D2", 0.55 + 0.35 * u, 0.20 + 0.30 * u)
    S.add("pad_lo", bt(4), 5 * BAR, "D2", 0.55, sec="cb", atk=1.6, rel=0.9, shape=(0.45, 1.0))
    B.sus("pad_lo", 4, 2, (("vc", "A2"), ("vc", "D3"), ("vla", "F3"), ("vla", "A3")), 0.6, atk=1.4, rel=0.7, shape=(0.35, 0.60))
    B.sus("pad_lo", 6, 2, (("vc", "Bb2"), ("vc", "D3"), ("vla", "F3"), ("vla", "Bb3")), 0.6, atk=0.45, rel=0.7, shape=(0.60, 0.75))
    B.sus("pad_lo", 8, 1, (("vc", "C3"), ("vla", "E3"), ("vla", "G3"), ("vln2", "C4")), 0.6, atk=0.45, rel=0.7, shape=(0.75, 0.95))
    B.motif(4, MOTIF_A, 0.55)
    B.motif(6, MOTIF_B, 0.62)
    S.add("piano", bt(8, 2.0), 0, "E5", 0.60)
    S.add("piano", bt(8, 3.5), 0, "G5", 0.66)
    S.add("swell", 20.0 - 2.5, 2.5 - PRE, 0, 0.55, rev=True)

    # -------------------------------------------------------------- III  bars 9-14
    S.cue(20.0, "hit", "build begins: ostinato + taikos", 0.55)
    S.cue(25.0, "hit", "percussion grows, violins join (Bb)", 0.68)
    S.cue(30.0, "hit", "16ths, riser starts (Gm -> Asus4 -> A)", 0.80)
    B.hit(20.0, 0.55, "D1", "small", imp=0.8, crash_=0.3)
    B.hit(25.0, 0.68, "Bb1", "med", imp=0.8, crash_=0.3)
    B.hit(30.0, 0.80, "G1", "med", imp=0.9, crash_=0.45)
    S.add("piano", bt(9), 0, "A5", 0.70, lock=True)
    B.motif(9, MOTIF_A, 0.70)
    B.motif(11, MOTIF_B, 0.75)
    for bar, n, ch in spans(9, 14):
        u0, u1 = (bar - 9) / 6.0, (bar + n - 9) / 6.0
        c = CHORD[ch]
        B.sub(bar, n, c["sub"], 0.62 + 0.3 * u0, boom=bar in (9, 11, 13), shape=(1.0, 1.0 + 0.3 * (u1 - u0)))
        B.sus("pad_lo", bar, n, (("cb", c["chug"][0]), ("vc", c["chug"][1] if ch != "Dm" else "A2")), 0.6,
              atk=0.35, rel=0.6, shape=(0.55 + 0.4 * u0, 0.55 + 0.4 * u1))
        if ch != "A":
            B.sus("pad_mid", bar, n, (("vla", c["pad"][0]), ("vla", c["pad"][1]), ("vln2", c["pad"][2])), 0.62,
                  atk=0.35, rel=0.6, shape=(0.45 + 0.5 * u0, 0.45 + 0.5 * u1))
    # bar 14: the D of Gm is held as a 4-3 suspension over A, resolving on beat 3
    B.sus("pad_mid", 14, 1, (("vla", "A3"), ("vln2", "E4")), 0.62, atk=0.35, rel=0.3, shape=(0.86, 1.0))
    S.add("pad_mid", bt(14), 2 * BEAT, "D4", 0.62, sec="vla", atk=0.3, rel=0.35, shape=(0.86, 0.93))
    S.add("pad_mid", bt(14, 3), 2 * BEAT, "C#4", 0.62, sec="vla", atk=0.2, rel=0.3, shape=(0.93, 1.0))
    for bar, n, ps, v0, v1 in ((9, 2, ("A4",), 0.35, 0.50), (11, 2, ("Bb4",), 0.50, 0.65),
                               (13, 1, ("Bb4", "D5"), 0.65, 0.80), (14, 1, ("A4", "E5"), 0.80, 1.0)):
        B.sus("pad_hi", bar, n, [("vln1", p) for p in ps], 0.6, atk=0.4, rel=0.5 if bar < 14 else 0.3, shape=(v0, v1))
    for bar, n, ps, v0, v1 in ((11, 2, ("Bb3", "D4", "F4"), 0.30, 0.48), (13, 1, ("Bb3", "D4", "G4"), 0.48, 0.62)):
        for p in ps:
            S.add("choir", bt(bar), n * BAR, p, 0.7, vowel=0.15, atk=0.9, rel=0.8, shape=(v0, v1))
    for p, d, b3 in (("A3", 4, 1), ("E4", 4, 1), ("D4", 2, 1), ("C#4", 2, 3)):
        S.add("choir", bt(14, b3), d * BEAT, p, 0.7, vowel=0.30, atk=0.5, rel=0.3, shape=(0.62 if b3 == 1 else 0.72, 0.85))
    S.add("lowbrass", bt(13), BAR, "G2", 0.55, kind="low", pan=0.2, atk=0.5, rel=0.4, shape=(0.35, 0.60))
    S.add("lowbrass", bt(13), BAR, "D3", 0.55, kind="low", pan=0.3, atk=0.5, rel=0.4, shape=(0.35, 0.60))
    for p in ("A1", "A2", "E3"):
        S.add("lowbrass", bt(14), BAR, p, 0.8, kind="low", pan=0.2, atk=0.4, rel=0.2, shape=(0.55, 1.0))
    S.add("horns", bt(14, 3), 2 * BEAT, "C#4", 0.75, atk=0.25, rel=0.2, shape=(0.6, 1.1))        # leading tone
    S.add("horns", bt(14, 3), 2 * BEAT, "A3", 0.65, atk=0.25, rel=0.2, shape=(0.6, 1.1))
    # strings: 8ths -> 8ths in two octaves -> 16ths
    for bar in (9, 10):
        B.ost("ost_lo", "vc", bar, PROG[bar], 0.50 + 0.05 * (bar - 9))
        B.pulse(bar, "D2", 0.82, 0.46 + 0.04 * (bar - 9))
    for bar in (11, 12):
        B.ost("ost_lo", "vc", bar, PROG[bar], 0.66 + 0.05 * (bar - 11))
        B.ost("ost_hi", "vln2", bar, PROG[bar], 0.46 + 0.05 * (bar - 11), octave=12)
        B.chug(bar, PROG[bar], 0.50 + 0.06 * (bar - 11), acc=ACC_8)
        B.pulse(bar, "Bb1", 0.88, 0.56 + 0.04 * (bar - 11))
    for bar in (13, 14):
        last = 15 if bar == 13 else 14
        c0 = 0.78 if bar == 13 else 0.90
        B.ost("ost_lo", "vc", bar, PROG[bar], 1.0, div=16, cresc=(c0, c0 + 0.12), last_step=last)
        B.ost("ost_hi", "vln2", bar, PROG[bar], 0.9, div=16, octave=12, cresc=(c0 - 0.1, c0 + 0.08), last_step=last)
        B.chug(bar, PROG[bar], 0.9, div=8 if bar == 13 else 16, acc=ACC_8 if bar == 13 else ACC_STRAIGHT,
               cresc=(c0 - 0.1, c0 + 0.1), last_step=last)
        B.pulse(bar, CHORD[PROG[bar]]["pulse"], 0.95, 0.68 + 0.1 * (bar - 13), last_step=last)
    # percussion grows every two bars
    B.drums(9, ((8, "mid", 0.32, 0.3),))
    B.drums(10, ((0, "low", 0.50, -0.2), (8, "mid", 0.34, 0.3), (14, "mid", 0.30, -0.3)))
    for bar in (9, 10):
        B.ticks(bar, 0.35, div=8, acc=ACC_STRAIGHT)
    B.drums(11, ((3, "mid", 0.38, 0.3), (6, "mid", 0.42, -0.3), (8, "low", 0.52, 0.2), (11, "mid", 0.40, 0.3), (14, "mid", 0.44, -0.3)))
    B.drums(12, ((0, "low", 0.62, -0.2), (0.25, "mid", 0.45, 0.3), (3, "mid", 0.40, 0.3), (6, "mid", 0.44, -0.3),
                 (8, "low", 0.56, 0.2), (11, "mid", 0.42, 0.3), (14, "low", 0.50, -0.3), (15, "mid", 0.46, 0.3)))
    for bar in (11, 12):
        B.ticks(bar, 0.45)
        B.clacks(bar, (4, 12), 0.35)
    B.drums(13, tuple((st, "low" if st % 4 == 0 else "mid", 0.42 + 0.02 * st, (-0.3, 0.3)[(st // 2) % 2]) for st in range(2, 16, 2)))
    roll = [(st, "low" if st % 2 == 0 else "mid", 0.55 + 0.03 * st, (-0.35, 0.35)[st % 2]) for st in range(0, 12)]
    roll += [(12 + 0.5 * i, ("low", "mid", "high")[i % 3], 0.88 + 0.04 * i, (-0.4, 0.0, 0.4)[i % 3]) for i in range(6)]   # to step 14.5
    B.drums(14, tuple(roll))
    for bar in (13, 14):
        B.ticks(bar, 0.55 + 0.1 * (bar - 13))
        B.clacks(bar, tuple(range(0, 15, 2)), 0.42 + 0.1 * (bar - 13))
    S.add("riser", 30.0, GAP1[0] - 30.0, 0, 1.0, m_from=midi("A3"), m_to=midi("A5"))
    S.add("swell", GAP1[0] - 2.6, 2.6, 0, 0.8, rev=True)

    # --------------------------------------------------------------- IV  bars 15-24
    S.cue(35.0, "drop", "THE DROP: braam + sub boom + taikos, theme enters", 1.0)
    B.hit(35.0, 1.12, "D1", "big", tau=0.95, crash_=1.0, imp=1.15, bell_=0.7, tam=0.6)
    for bar, lvl in ((17, 0.82), (19, 0.86), (21, 0.90), (23, 0.86)):
        S.cue(bt(bar), "braam_hit", "braam + hit on %s" % PROG[bar], lvl)
        B.hit(bt(bar), lvl, CHORD[PROG[bar]]["sub"], "med", crash_=0.6, imp=1.0)
    for bar, dur, v in ((15, 3.4, 1.0), (17, 3.2, 0.86), (19, 3.2, 0.88), (21, 3.2, 0.92), (23, 2.0, 0.88)):
        S.add("braam", bt(bar), dur, CHORD[PROG[bar]]["braam"], v, lock=True, rel=1.3 if bar < 23 else 0.6)
    for trk in ("braam", "horns", "mel_str", "lowbrass", "choir", "pad_mid", "taiko"):
        S.auto[trk] = [(35.0, 1.5), (36.2, 1.5), (38.0, 0.0), (74.9, 0.0), (75.0, 1.2), (77.0, 1.2)]     # push the two big downbeats
    for bar, n, ch in spans(15, 24):
        c = CHORD[ch]
        B.sub(bar, n, c["sub"], 0.9, boom=bar % 2 == 1)
        for i, p in enumerate(c["low"]):
            S.add("lowbrass", bt(bar), n * BAR, p, 0.75, kind="low", pan=(0.15, 0.25, 0.35)[i], atk=0.08, rel=0.35,
                  shape=(1.0, 0.8) if bar != 24 else (0.75, 1.05))
        B.sus("pad_mid", bar, n, (("vla", c["pad"][0]), ("vln2", c["pad"][1]), ("vln2", c["pad"][2])), 0.8,
              atk=0.10 if bar == 15 else 0.3, rel=0.55)
        for p in c["choir"]:
            S.add("choir", bt(bar), n * BAR, p, 0.70 if bar < 19 else 0.85, vowel=0.45, atk=0.25 if bar == 15 else 0.6, rel=1.0)
    for bar, beat, beats, p, v in THEME:
        d = beats * BEAT
        t = bt(bar, beat)
        shape = (0.9, 1.15) if bar == 24 else None            # the last note leans into the lift
        S.add("horns", t, d, midi(p) - 12, 0.92 * v, lock=(beat == 1 and bar in B.cue_bars), shape=shape)
        S.add("mel_str", t, d, midi(p) - 12, 0.80 * v, sec="vc", atk=0.09, rel=0.25, vib=1.3, shape=shape)
        S.add("mel_str", t, d, p, 0.86 * v, sec="vln1", atk=0.09, rel=0.25, vib=1.3, shape=shape)
        S.add("mel_str", t, d, p, 0.70 * v, sec="vla", atk=0.10, rel=0.25, vib=1.2, shape=shape)
    for bar in range(15, 25):
        ch = PROG[bar]
        B.ost("ost_lo", "vc", bar, ch, 0.86)
        B.ost("ost_hi", "vln2", bar, ch, 0.62 if bar < 19 else 0.74, div=16, octave=12)
        B.chug(bar, ch, 0.92)
        B.pulse(bar, CHORD[ch]["pulse"], 0.9, 0.70)
        B.ticks(bar, 0.60)
        B.clacks(bar, (4, 12), 0.50)
        if bar % 2 == 1:      # braam bars: downbeat comes from hit(); half-time answer on beat 3
            B.drums(bar, ((8, "low", 0.78, -0.4), (8.06, "mid", 0.50, 0.35), (11, "mid", 0.42, 0.3),
                          (14, "low", 0.50, 0.4), (15, "mid", 0.46, -0.3)))
        else:
            B.ens(bt(bar), 0.74, "med", lock=False)
            B.drums(bar, ((6, "mid", 0.45, 0.3), (8, "low", 0.80, 0.4), (8.06, "mid", 0.50, -0.3),
                          (12, "mid", 0.50, 0.3), (14, "low", 0.62, -0.4), (15, "low", 0.70, 0.4), (15.5, "mid", 0.55, -0.2)))
            if bar in (18, 22):       # phrase ends: a short run down the drums
                B.drums(bar, ((10, "high", 0.42, 0.5), (11, "high", 0.46, 0.3), (13, "mid", 0.50, 0.0)))
            if bar == 24:             # turnaround: 16ths building into the lift
                B.drums(bar, tuple((st, ("mid", "high")[st % 2], 0.36 + 0.035 * (st - 1), (0.35, -0.35)[st % 2])
                                   for st in (1, 2, 3, 4, 5, 7, 9, 10, 11, 13)))
    S.add("swell", 60.0 - 2.5, 2.5 - PRE, 0, 0.7, rev=True)
    for t in (40.0, 45.0, 50.0, 55.0, 65.0, 70.0):
        S.add("swell", t - 1.25, 1.25 - PRE, 0, 0.22, lo=700.0, hi=6500.0)

    # ---------------------------------------------------------------- V  bars 25-30
    S.cue(60.0, "lift", "second lift: theme in octaves, full-time taikos", 0.97)
    S.cue(65.0, "braam_hit", "braam + hit on F", 0.86)
    S.cue(70.0, "braam_hit", "braam + hit on Gm, riser starts", 0.90)
    B.hit(60.0, 0.93, "D1", "big", tau=0.8, crash_=0.9, imp=1.0, bell_=0.6)
    B.hit(65.0, 0.85, "F1", "med", crash_=0.9, imp=1.2)
    B.hit(70.0, 0.88, "G1", "med", crash_=0.9, imp=1.2)
    for bar, dur, v, rel in ((25, 2.3, 1.0, 1.0), (27, 2.3, 0.90, 1.0), (29, 2.2, 0.92, 0.4), (30, 2.3, 0.92, 0.2)):
        S.add("braam", bt(bar), dur, CHORD[PROG[bar]]["braam"], v, lock=True, rel=rel)
    for bar in range(25, 31):
        ch = PROG[bar]
        c = CHORD[ch]
        u = (bar - 25) / 5.0
        last = 14 if bar == 30 else 15
        B.sub(bar, 1, c["sub"], 0.95, boom=bar % 2 == 1)
        for i, p in enumerate(c["low"]):
            S.add("lowbrass", bt(bar), BAR, p, 0.85, kind="low", pan=(0.15, 0.25, 0.35)[i], atk=0.07, rel=0.3,
                  shape=(1.0, 0.85) if bar < 29 else (0.85, 1.1))
        for p in c["men"]:
            S.add("choir", bt(bar), BAR, p, 0.75, vowel=0.55, atk=0.3, rel=0.6)
        if ch != "A":
            B.sus("pad_mid", bar, 1, (("vla", c["pad_up"][0]), ("vln2", c["pad_up"][1]), ("vln2", c["pad_up"][2])), 0.9, atk=0.2, rel=0.45)
            for p in c["choir_up"]:
                S.add("choir", bt(bar), BAR, p, 0.84, vowel=0.62, atk=0.35, rel=0.8)
        B.ost("ost_lo", "vc", bar, ch, 0.95 + 0.05 * u, div=16, acc=ACC_332, last_step=last)
        B.ost("ost_hi", "vln2", bar, ch, 0.84 + 0.08 * u, div=16, octave=12, acc=ACC_332, last_step=last)
        B.chug(bar, ch, 1.0, div=16, acc=ACC_332, last_step=last)
        B.pulse(bar, c["pulse"], 0.9, 0.85, div=16, acc=ACC_332, last_step=last)
        B.ticks(bar, 0.80, acc=ACC_332)
        B.clacks(bar, tuple(range(0, last + 1, 2)), 0.62)
        if bar % 2 == 0:
            B.ens(bt(bar), 0.80, "med", lock=False)
        if bar < 30:          # full-time: lows on the 3+3+2 accents, mids fill the 16ths
            pat = [(3, "low", 0.70, -0.4), (6, "low", 0.75, 0.4), (8, "odaiko", 0.80, 0.0), (8.05, "low", 0.70, -0.3),
                   (11, "low", 0.70, 0.4), (14, "low", 0.80, -0.4)]
            pat += [(st, "mid", 0.34 + 0.05 * (st % 2), (0.3, -0.3)[st % 2]) for st in (1, 2, 4, 5, 7, 9, 10, 12, 13, 15)]
            B.drums(bar, tuple(pat), level=1.0 + 0.1 * u)
    # bar 30: Asus4 -> A, everything crescendos into the cut
    B.sus("pad_mid", 30, 1, (("vla", "A3"), ("vln2", "E4")), 0.95, atk=0.2, rel=0.2, shape=(0.9, 1.1))
    S.add("pad_mid", bt(30), 2 * BEAT, "D4", 0.95, sec="vln2", atk=0.2, rel=0.3)
    S.add("pad_mid", bt(30, 3), 2 * BEAT, "C#4", 0.95, sec="vln2", atk=0.15, rel=0.2, shape=(1.0, 1.1))
    for p, d, b3 in (("E5", 4, 1), ("A5", 4, 1), ("D5", 2, 1), ("C#5", 2, 3)):
        S.add("choir", bt(30, b3), d * BEAT, p, 0.95, vowel=0.7, atk=0.3, rel=0.2, shape=(0.9, 1.1))
    S.add("horns", bt(30, 3), 2 * BEAT, "C#4", 0.9, atk=0.12, rel=0.2, shape=(0.8, 1.15))         # leading tone
    roll = [(st, "low" if st % 2 == 0 else "mid", 0.62 + 0.03 * st, (-0.35, 0.35)[st % 2]) for st in range(1, 12)]
    roll += [(12 + 0.5 * i, ("low", "mid", "high")[i % 3], 0.92 + 0.03 * i, (-0.4, 0.0, 0.4)[i % 3]) for i in range(6)]
    B.drums(30, tuple(roll))
    for bar, beat, beats, p, v in LIFT:
        d = beats * BEAT
        t = bt(bar, beat)
        lock = beat == 1 and bar in B.cue_bars
        shape = (0.9, 1.15) if bar == 30 else None
        S.add("horns", t, d, midi(p) - 12, 1.0 * v, lock=lock, shape=shape)
        S.add("horns", t, d, midi(p) - 24, 0.85 * v, lock=lock, shape=shape, kind="low", pan=0.1)
        S.add("mel_str", t, d, p, 0.95 * v, sec="vln1", atk=0.07, rel=0.2, vib=1.4, shape=shape)
        S.add("mel_str", t, d, midi(p) - 12, 0.82 * v, sec="vln2", atk=0.07, rel=0.2, vib=1.3, shape=shape)
        S.add("mel_str", t, d, midi(p) - 12, 0.70 * v, sec="vla", atk=0.08, rel=0.2, vib=1.2, shape=shape)
    S.add("riser", 70.0, GAP2[0] - 70.0, 0, 1.0, m_from=midi("D4"), m_to=midi("A5"))
    S.add("swell", GAP2[0] - 2.6, 2.6, 0, 0.9, rev=True)

    # --------------------------------------------------------------- VI  bars 31-32 + tail
    S.cue(75.0, "final_hit", "final hit: D minor rings out, thins to an open fifth", 1.0)
    B.hit(75.0, 1.1, "D1", "big", tau=1.5, crash_=1.0, imp=1.1, bell_=1.0, tam=1.0)
    S.add("braam", 75.0, 4.2, "D1", 1.0, lock=True, rel=3.0, droop=-22.0)
    S.add("sub", 75.0, 4.0, "D1", 0.9, lock=True, phase0=2 * np.pi * hz("D1") * BOOM_SWEEP * BOOM_TAUP, rel=2.5, shape=(1.0, 0.25))
    S.add("drone", 75.0, 5.5, "D1", 0.8, rel=3.0, shape=[(0.0, 0.9), (5.5, 0.35), (8.5, 0.0)])
    for i, p in enumerate(("D2", "D3", "A3")):
        S.add("lowbrass", 75.0, 3.6, p, 0.9, kind="low", pan=(0.15, 0.25, 0.35)[i], atk=0.06, rel=2.4, shape=(1.0, 0.28))
    for p in ("A3", "D4"):
        S.add("horns", 75.0, 3.6, p, 0.95, lock=True, rel=2.4, shape=(1.0, 0.28))
    S.add("horns", 75.0, 2.4, "F4", 0.85, lock=True, rel=1.6, shape=(1.0, 0.4))
    B.sus("pad_lo", 31, 2, (("cb", "D2"), ("vc", "A2"), ("vc", "D3")), 0.9, atk=0.08, rel=2.8, shape=(1.0, 0.25))
    B.sus("pad_mid", 31, 2, (("vla", "A3"), ("vln2", "D4")), 0.9, atk=0.08, rel=2.8, shape=(1.0, 0.25))
    S.add("pad_mid", 75.0, 2.6, "F4", 1.0, sec="vln2", atk=0.08, rel=1.8, shape=(1.0, 0.4))       # the third leaves first
    S.add("pad_mid", 75.0, 2.6, "F3", 0.9, sec="vla", atk=0.08, rel=1.8, shape=(1.0, 0.4))
    B.sus("pad_hi", 31, 2, (("vln2", "A4"), ("vln1", "D5"), ("vln1", "D6")), 0.8, atk=0.08, rel=3.0, shape=(1.0, 0.22))
    for p, d, r in (("D3", 4.6, 2.4), ("A3", 4.6, 2.4), ("F4", 2.4, 1.8), ("A4", 4.8, 2.6), ("D5", 4.8, 2.6), ("F5", 2.4, 1.8), ("A5", 4.2, 2.4)):
        S.add("choir", 75.0, d, p, 0.9, vowel=0.6, atk=0.2, rel=r, shape=(1.0, 0.28))
    for (db, beat), p, v in zip(MOTIF_RHYTHM, MOTIF_A, (0.62, 0.50, 0.55, 0.60)):                    # the motif, once more
        S.add("piano", bt(31 + db, beat), 0, p, v)
        S.add("piano", bt(31 + db, beat) + 0.005, 0, midi(p) - 12, v * 0.4)
    S.add("air", 75.5, 8.5, 0, 0.8)
    S.add("drop", 80.62, 0, "A6", 0.35, pan=-0.4)
    S.add("drop", 82.05, 0, "D7", 0.25, pan=0.4)
    return S


def compose_bed() -> Score:
    """30.0 s engineering-explainer bed: same key, tempo and motif; no big drums.
    Dm (3 bars) | Bb (2) | F (2) | C (2) | Bb | C | soft hit on D at 27.5 s."""
    S = Score("bed", T_BED, bounds=(0.0, T_BED), gaps=())
    prog = {1: "Dm", 2: "Dm", 3: "Dm", 4: "Bb", 5: "Bb", 6: "F", 7: "F", 8: "C", 9: "C", 10: "Bb", 11: "C", 12: "D5"}
    S.chords = dict(prog)
    B = _Builder(S, cue_bars=(1, 12))
    S.section("in", 1, 1, "drone, pulse and pad fade in")
    S.section("body", 2, 9, "motif and its three answers over Dm - Bb - F - C")
    S.section("swell", 10, 11, "Bb - C, gentle swell")
    S.section("out", 12, 12, "soft end hit at 27.5 s, ring-out")
    S.cue(0.0, "start", "drone, pulse and pad fade in", 0.2)
    S.cue(27.5, "soft_end_hit", "soft end hit (D, open fifth), rings to 30.0 s", 0.5)
    lv = lambda bar: float(np.interp(bar, [1, 2, 8, 11, 12], [0.50, 0.62, 0.74, 0.95, 1.0]))   # one long hairpin
    # floor: D pedal under the first phrase, roots afterwards
    S.add("drone", 0.0, 3 * BAR, "D1", 0.9, rel=1.5, shape=[(0.0, 0.0), (0.7, 0.55), (7.5, 0.70), (9.0, 0.0)])
    S.add("air", 0.0, 27.5, 0, 1.0)
    for bar in range(1, 12):
        for beat in (1, 3):
            S.add("heart", bt(bar, beat), 0, 0, 0.45 + 0.2 * lv(bar), f=50.0)
            S.add("heart", bt(bar, beat + 0.5), 0, 0, (0.45 + 0.2 * lv(bar)) * 0.66, f=58.0, tau=0.06)
    for bar in range(1, 12):
        ch = prog[bar]
        B.pulse(bar, CHORD[ch]["pulse"], 0.52 + 0.33 * lv(bar), 0.22 + 0.30 * lv(bar))
    for bar, n, ch in ((4, 2, "Bb"), (6, 2, "F"), (8, 2, "C"), (10, 1, "Bb"), (11, 1, "C")):
        S.add("sub", bt(bar), n * BAR, CHORD[ch]["sub"], 0.6, atk=0.25, rel=0.4, shape=(1.0, 1.15) if bar == 11 else None)
    # pad: low strings with the theme's voice-leading (A-Bb-A-G on top)
    for bar, n, ch in ((1, 3, "Dm"), (4, 2, "Bb"), (6, 2, "F"), (8, 2, "C"), (10, 1, "Bb"), (11, 1, "C")):
        c = CHORD[ch]
        low = (("cb", "D2"), ("vc", "A2")) if ch == "Dm" else (("cb", c["chug"][0] if ch != "F" else "F2"), ("vc", c["low"][1]))
        B.sus("pad_lo", bar, n, low, 0.55, atk=0.9 if bar == 1 else 0.5, rel=0.8, shape=(lv(bar), lv(bar + n)))
        B.sus("pad_mid", bar, n, (("vla", c["pad"][0]), ("vla", c["pad"][1]), ("vln2", c["pad"][2])), 0.55,
              atk=0.9 if bar == 1 else 0.5, rel=0.8, shape=(lv(bar), lv(bar + n)))
    for bar, n, ps in ((6, 2, ("A4", "C5")), (8, 2, ("G4", "C5")), (10, 1, ("Bb4", "D5")), (11, 1, ("C5", "E5"))):
        for p in ps:
            S.add("choir", bt(bar), n * BAR, p, 0.55, vowel=0.15, atk=0.9, rel=0.9, shape=(lv(bar) * 0.8, lv(bar + n) * 0.8))
    for bar in range(6, 12):                     # a whisper of the ostinato for motion
        B.ost("ost_lo", "vc", bar, prog[bar], 0.26 + 0.035 * (bar - 6))
    # the motif and its three answers (the four phrase-heads of the theme), then two rising notes
    B.motif(2, MOTIF_A, 0.58)
    B.motif(4, MOTIF_B, 0.62)
    B.motif(6, ("F5", "G5", "A5", "C6"), 0.66)
    B.motif(8, ("D6", "C6", "G5", "E5"), 0.68)
    for bar, notes in ((10, ("F5", "Bb5")), (11, ("G5", "C6"))):
        S.add("piano", bt(bar, 2.0), 0, notes[0], 0.60)
        S.add("piano", bt(bar, 3.5), 0, notes[1], 0.66)
    for t, p, v, pn in ((1.31, "A6", 0.45, -0.5), (6.02, "D7", 0.40, 0.45), (11.4, "F6", 0.40, 0.2),
                        (16.7, "A6", 0.40, -0.35), (21.55, "E7", 0.35, 0.5)):
        S.add("drop", t, 0, p, v, pan=pn)
    S.add("swell", 27.5 - 2.5, 2.5 - 2 * PRE, 0, 0.32, lo=400.0, hi=5000.0)
    # soft end hit at 27.5 s: sub + dark impact + D open fifth in strings and piano
    B.soft_hit(27.5, 0.74, "D1", tau=0.8)
    S.add("heart", 27.5, 0, 0, 0.8, f=50.0, lock=True)
    for sec, p in (("cb", "D2"), ("vc", "A2"), ("vc", "D3"), ("vla", "A3"), ("vln2", "D4")):
        S.add("pad_lo" if sec in ("cb", "vc") else "pad_mid", 27.5, 1.2, p, 0.6, sec=sec, atk=0.05, rel=1.2, shape=(1.0, 0.5))
    for p, v in (("D5", 0.62), ("A5", 0.55), ("D4", 0.35), ("D3", 0.40)):
        S.add("piano", 27.5, 0, p, v, lock=True)
    return S


# =============================================================================
# 6. MIXER
#    track (events -> dry buffer -> inserts -> automation/ducking -> gain)
#      -> stem bus (dry + hall send + room send)  ->  stems
#    The piece is rendered in segments split at the two cut points, so nothing
#    (notes or reverb tails) can ring through the silences before the big hits.
# =============================================================================
STEMS = ("drones", "strings", "perc", "braams", "choir_pad", "fx", "keys")


@dataclass
class Track:
    stem: str
    fn: object                    # (event, velocity, rng) -> buffer
    gain: float = 0.0             # dB
    hall: float | None = None     # send level, dB (None = dry)
    room: float | None = None
    hp: float = 0.0               # Hz (0 = off). Everything that is not bass is cut below ~120 Hz.
    lp: float = 0.0
    eq: tuple = ()
    duck: float = 0.0             # dB of side-chain ducking under the big hits (recovers in ~0.2 s)
    breath: float = 0.0           # dB dip in the last ~80 ms before a big hit: the section 'breathes' before the downbeat
    human: tuple = (0.0, 0.0)     # (timing sigma in s, velocity sigma) for un-locked events
    post: object = None           # insert effect on the summed track


def _r_sus(e, v, rng):
    k = e.kw
    return string_sus(e.pitch, e.dur, v, k.get("sec", "vc"), rng, atk=k.get("atk", 0.30), rel=k.get("rel", 0.55),
                      shape=k.get("shape"), vib=k.get("vib", 1.0))


def _r_spic(e, v, rng):
    return string_spic(e.pitch, v, e.kw.get("sec", "vc"), rng, length=e.kw.get("length", 1.0), lock=e.lock)


def _r_brass(e, v, rng):
    k = e.kw
    return brass_note(e.pitch, e.dur, v, rng, kind=k.get("kind", "horn"), atk=k.get("atk", 0.07), rel=k.get("rel", 0.22),
                      pan_c=k.get("pan", -0.28), shape=k.get("shape"))


def _r_braam(e, v, rng):
    return braam(e.pitch, e.dur, v, rng, rel=e.kw.get("rel", 1.3), droop=e.kw.get("droop", -16.0))


def _r_choir(e, v, rng):
    k = e.kw
    return choir_note(e.pitch, e.dur, v, rng, atk=k.get("atk", 0.7), rel=k.get("rel", 1.1), vowel=k.get("vowel", 0.45),
                      shape=k.get("shape"))


def _r_piano(e, v, rng):
    return piano_note(e.pitch, v, rng, pan_c=float(np.clip((e.pitch - 74.0) / 40.0, -0.3, 0.3)))


def _r_taiko(e, v, rng):
    return pan(taiko(e.kw.get("kind", "low"), min(v, 1.15), rng), e.kw.get("pan", 0.0))


def _r_boom(e, v, rng):
    return pan(sub_boom(hz(e.pitch), v, tau=e.kw.get("tau", 0.6), sweep=BOOM_SWEEP, tau_p=BOOM_TAUP), 0.0)


def _r_heart(e, v, rng):
    return pan(heart(v, rng, f=e.kw.get("f", 50.0), tau=e.kw.get("tau", 0.075)), 0.0)


def _r_tick(e, v, rng):
    return pan(tick(v, rng), e.kw.get("pan", 0.0))


def _r_clack(e, v, rng):
    return pan(clack(v, rng), e.kw.get("pan", 0.0))


def _r_pulse(e, v, rng):
    return pulse_note(e.pitch, v, rng, open_=e.kw.get("open", 0.4), length=e.kw.get("length", 1.0))


def _r_sub(e, v, rng):
    k = e.kw
    return sub_note(e.pitch, e.dur, v, phase0=k.get("phase0", 0.0), atk=k.get("atk", 0.03), rel=k.get("rel", 0.12),
                    shape=k.get("shape"))


def _r_drone(e, v, rng):
    return drone(e.pitch, e.dur, v, rng, shape=e.kw.get("shape"), rel=e.kw.get("rel", 1.5))


def _r_impact(e, v, rng):
    return impact(v, rng, size=e.kw.get("size", 1.0), dark=e.kw.get("dark", False))


def _r_crash(e, v, rng):
    return crash(v, rng)


def _r_tamtam(e, v, rng):
    return tamtam(v, rng)


def _r_bell(e, v, rng):
    return bell(e.pitch, v, rng)


def _r_riser(e, v, rng):
    return riser(e.dur, v, rng, m_from=e.kw.get("m_from", 57.0), m_to=e.kw.get("m_to", 81.0))


def _r_swell(e, v, rng):
    if e.kw.get("rev"):
        return rev_crash(e.dur, v, rng)
    return swell(e.dur, v, rng, lo=e.kw.get("lo", 500.0), hi=e.kw.get("hi", 7000.0))


def _r_air(e, v, rng):
    return air(e.dur, v, rng)


def _r_drop(e, v, rng):
    return droplet(e.pitch, v, rng, pan_c=e.kw.get("pan", 0.0))


def _drum_sat(x, knee=1.2):
    """Soft-clip the drum group (oversampled): shaves the tallest ensemble peaks, adds density."""
    return knee * saturate_os(x / knee, 1.0)


def _echo_keys(x):
    return ping_pong(x, 0.75 * BEAT, fb=0.42, mix=0.34)          # dotted-8th


def _echo_drop(x):
    return ping_pong(x, 0.75 * BEAT, fb=0.5, mix=0.5, damp=5000.0)


TRACKS = {
    # low end (the only parts allowed below ~100 Hz, together with drums and the braam)
    "drone": Track("drones", _r_drone, gain=-16.0, hall=-14.0, hp=28.0, duck=2.0, breath=2.0),
    "sub": Track("drones", _r_sub, gain=-14.0, hp=25.0, lp=400.0, duck=7.0, breath=7.0),
    "pulse": Track("drones", _r_pulse, gain=-13.0, hall=-20.0, room=-14.0, hp=45.0, duck=2.0, human=(0.002, 0.06)),
    # strings
    "pad_lo": Track("strings", _r_sus, gain=-24.0, hall=-6.0, hp=50.0, duck=2.5, breath=5.0),
    "pad_mid": Track("strings", _r_sus, gain=-28.0, hall=-5.0, hp=120.0, duck=2.5, breath=5.0, eq=(("peak", 380.0, -2.5, 0.8),)),
    "pad_hi": Track("strings", _r_sus, gain=-26.0, hall=-4.0, hp=180.0, duck=2.5, breath=5.0),
    "mel_str": Track("strings", _r_sus, gain=-21.5, hall=-5.0, hp=140.0, breath=4.0, human=(0.008, 0.04)),
    "ost_lo": Track("strings", _r_spic, gain=-10.0, hall=-13.0, room=-8.0, hp=100.0, human=(0.004, 0.07)),
    "ost_hi": Track("strings", _r_spic, gain=-9.0, hall=-12.0, room=-8.0, hp=180.0, human=(0.0045, 0.08)),
    "chug": Track("strings", _r_spic, gain=-10.0, hall=-16.0, room=-9.0, hp=50.0, human=(0.004, 0.06)),
    # percussion
    "taiko": Track("perc", _r_taiko, gain=-2.0, hall=-15.0, room=-7.0, hp=35.0, human=(0.003, 0.08), post=_drum_sat),
    "boom": Track("perc", _r_boom, gain=-7.0, hp=24.0, lp=320.0),
    "heart": Track("perc", _r_heart, gain=-12.0, hall=-8.0, room=-10.0, hp=30.0, human=(0.0, 0.05)),
    "tick": Track("perc", _r_tick, gain=-21.0, hall=-18.0, room=-12.0, hp=2500.0, human=(0.003, 0.12)),
    "clack": Track("perc", _r_clack, gain=-14.0, hall=-14.0, room=-9.0, hp=300.0, human=(0.003, 0.10)),
    # brass
    "braam": Track("braams", _r_braam, gain=-6.0, hall=-10.0, hp=48.0, breath=5.0, eq=(("peak", 420.0, -2.0, 0.8),)),
    "horns": Track("braams", _r_brass, gain=-16.0, hall=-4.0, hp=110.0, breath=4.0, human=(0.008, 0.04)),
    "lowbrass": Track("braams", _r_brass, gain=-23.0, hall=-7.0, hp=50.0, duck=2.0, breath=5.0, eq=(("peak", 320.0, -2.5, 0.8),)),
    # voices
    "choir": Track("choir_pad", _r_choir, gain=-25.0, hall=-3.0, hp=170.0, duck=2.0, breath=5.0, eq=(("peak", 1000.0, -2.5, 1.0),)),
    # effects
    "impact": Track("fx", _r_impact, gain=-8.0, hall=-8.0, room=-12.0, hp=30.0),
    "crash": Track("fx", _r_crash, gain=-14.0, hall=-8.0, hp=300.0),
    "tamtam": Track("fx", _r_tamtam, gain=-21.0, hall=-8.0, hp=120.0),
    "bell": Track("fx", _r_bell, gain=-18.0, hall=-6.0, hp=60.0),
    "riser": Track("fx", _r_riser, gain=-10.0, hall=-10.0, hp=150.0),
    "swell": Track("fx", _r_swell, gain=-18.0, hall=-10.0, hp=250.0),
    "air": Track("fx", _r_air, gain=-41.0, hall=-12.0, hp=250.0),
    "drop": Track("fx", _r_drop, gain=-30.0, hall=-2.0, hp=500.0, post=_echo_drop),
    # keys
    "piano": Track("keys", _r_piano, gain=-11.0, hall=-4.0, hp=150.0, human=(0.010, 0.07), post=_echo_keys),
}


def _duck_curves(score, a, L):
    """(breath, duck) depth curves in 0..1 around every big hit.
    breath: eases in over the 85 ms before the hit, released in 6 ms at the hit.
    duck:   8 ms attack at the hit, ~0.2 s recovery."""
    pre, post = np.zeros(L), np.zeros(L)
    kp, kr = int(0.085 * SR), int(0.006 * SR)
    for t, amt in score.ducks:
        i = n_of(t) - a
        if 0 < i <= L:
            j = max(0, i - kp)
            pre[j:i] = np.maximum(pre[j:i], amt * smoothstep((np.arange(j, i) - (i - kp)) / (0.060 * SR)))
        if 0 <= i < L:
            k = min(L - i, kr)
            pre[i:i + k] = np.maximum(pre[i:i + k], amt * (1.0 - smoothstep(np.arange(k) / kr)))
            k = min(L - i, int(1.2 * SR))
            tt = np.arange(k) / SR
            post[i:i + k] = np.maximum(post[i:i + k], amt * smoothstep(tt / 0.008) * np.exp(-tt / 0.20))
    return pre, post


def render(score: Score, tracks=TRACKS, log=print, stats=None):
    """Render a Score to pre-master stems {name: (N, 2)}."""
    N = n_of(score.length)
    stems = {s: np.zeros((N, 2)) for s in STEMS}
    by_track: dict = {}
    for e in score.events:
        by_track.setdefault(e.track, []).append(e)
    missing = set(by_track) - set(tracks)
    if missing:
        raise KeyError("events on unknown tracks: %s" % sorted(missing))
    for si in range(len(score.bounds) - 1):
        t0, t1 = score.bounds[si], score.bounds[si + 1]
        a, b = n_of(t0), n_of(t1)
        L = b - a
        tabs = t0 + np.arange(L) / SR
        pre, post = _duck_curves(score, a, L)
        dry = {s: np.zeros((L, 2)) for s in STEMS}
        hall = {s: None for s in STEMS}
        room = {s: None for s in STEMS}
        for name, tr in tracks.items():
            evs = [e for e in by_track.get(name, ()) if t0 <= e.t < t1]
            if not evs:
                continue
            tic = time.time()
            buf = np.zeros((L, 2))
            for e in evs:
                rng = rng_for(score.name, name, e.idx)
                t, v = e.t, e.vel
                if not e.lock:
                    st, sv = tr.human
                    if st:
                        t = min(max(t + float(np.clip(rng.normal(0.0, st), -2.5 * st, 2.5 * st)), t0), t1 - 1.0 / SR)
                    if sv:
                        v *= float(np.clip(1.0 + rng.normal(0.0, sv), 0.6, 1.4))
                x = tr.fn(e, v, rng)
                i0 = n_of(t) - a
                k = min(len(x), L - i0)
                if k > 0:
                    buf[i0:i0 + k] += x[:k]
            if tr.hp:
                buf = hp(buf, tr.hp, 2)
            if tr.lp:
                buf = lp(buf, tr.lp, 2)
            if tr.eq:
                buf = eq(buf, tr.eq)
            if tr.post is not None:
                buf = tr.post(buf)
            g = np.full(L, float(dbg(tr.gain)))
            if name in score.auto:
                ts, vs = zip(*score.auto[name])
                g *= dbg(np.interp(tabs, ts, vs))
            if tr.duck or tr.breath:
                g *= dbg(-(tr.duck * post + tr.breath * pre))
            buf *= g[:, None]
            dry[tr.stem] += buf
            if tr.hall is not None:
                hall[tr.stem] = buf * dbg(tr.hall) if hall[tr.stem] is None else hall[tr.stem] + buf * dbg(tr.hall)
            if tr.room is not None:
                room[tr.stem] = buf * dbg(tr.room) if room[tr.stem] is None else room[tr.stem] + buf * dbg(tr.room)
            if stats is not None:
                stats.setdefault(name, np.zeros(N // 4800 + 1))
                p = np.sum(k_weight(buf) ** 2, axis=1)
                nb = L // 4800
                stats[name][a // 4800: a // 4800 + nb] += p[: nb * 4800].reshape(nb, 4800).mean(axis=1)
                pk = stats.setdefault("_peak", {})
                pk[name] = max(pk.get(name, 0.0), float(np.max(np.abs(buf))))
            log("    seg %d  %-9s %4d ev  %5.1fs" % (si, name, len(evs), time.time() - tic))
        for s in STEMS:
            out = dry[s]
            if hall[s] is not None:
                out = out + reverb(hall[s], "hall", L)
            if room[s] is not None:
                out = out + reverb(room[s], "room", L)
            for g0, g1 in score.gaps:                 # silence before the cut: fade 12 ms, then digital black
                if abs(g1 - t1) < 1e-9:
                    i0, k = n_of(g0) - a, int(0.012 * SR)
                    out[i0 - k:i0] *= (1.0 - cosfade(k))[:, None]
                    out[i0:] = 0.0
            stems[s][a:b] = out
    return stems


# =============================================================================
# 7. MASTERING
#    Everything on the master bus is either linear (EQ, mono-below-110 Hz) or a
#    broadband time-varying gain (glue compression, true-peak limiter, loudness
#    trim, end fade). The same processing is therefore applied to each stem, and
#    the delivered stems sum to the delivered master.
# =============================================================================
_K1 = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
       np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K2 = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))


def k_weight(x):
    """ITU-R BS.1770 K-weighting (48 kHz coefficients)."""
    return sig.lfilter(_K2[0], _K2[1], sig.lfilter(_K1[0], _K1[1], x, axis=0), axis=0)


def block_power(x, win=0.4, hop=0.1):
    y = k_weight(x)
    p = np.sum(y * y, axis=1) if y.ndim == 2 else y * y
    cs = np.concatenate([[0.0], np.cumsum(p)])
    w, h = int(win * SR), int(hop * SR)
    st = np.arange(0, len(p) - w + 1, h)
    return (cs[st + w] - cs[st]) / w


def lufs(x):
    """Integrated loudness, BS.1770-4 (400 ms blocks, 75 % overlap, -70 LUFS and -10 LU gates)."""
    z = block_power(x)
    l = -0.691 + 10.0 * np.log10(z + 1e-20)
    keep = l > -70.0
    if not np.any(keep):
        return -120.0
    rel = -0.691 + 10.0 * np.log10(np.mean(z[keep])) - 10.0
    keep &= l > rel
    return float(-0.691 + 10.0 * np.log10(np.mean(z[keep])))


def true_peak_db(x, os=4):
    return float(to_db(np.max(np.abs(sig.resample_poly(x, os, 1, axis=0)))))


def tp_envelope(x, os=4):
    """Per-sample inter-sample-peak envelope (max over channels and the 4x oversampled neighbours)."""
    up = np.max(np.abs(sig.resample_poly(x, os, 1, axis=0)), axis=1)
    return ndi.maximum_filter1d(up.reshape(-1, os).max(axis=1), 3)


def glue_curve(mix, thr=-18.0, ratio=1.6, knee=10.0, atk=0.030, rel=0.28):
    """Bus compressor gain (linear, per sample). RMS detector with a 90 Hz side-chain
    high-pass so the sub does not pump the mix; 1 ms control rate with real ballistics."""
    hop = 48
    sc = hp(mix, 90.0, 2)
    p = np.mean(sc * sc, axis=1)
    nb = len(p) // hop
    lvl = 10.0 * np.log10(p[: nb * hop].reshape(nb, hop).mean(axis=1) + 1e-12)
    ca, cr = math.exp(-0.001 / atk), math.exp(-0.001 / rel)
    env = np.empty(nb)
    e = -120.0
    for i in range(nb):
        l = lvl[i]
        c = ca if l > e else cr
        e = c * e + (1.0 - c) * l
        env[i] = e
    over = env - thr
    slope = 1.0 / ratio - 1.0
    gr = np.where(over <= -knee / 2, 0.0, np.where(over >= knee / 2, slope * over, slope * (over + knee / 2) ** 2 / (2 * knee)))
    return dbg(np.interp(np.arange(len(mix)), (np.arange(nb) + 0.5) * hop, gr))


def limiter_gain(peak, ceiling, look=0.005, hold=0.015, rel_db_s=28.0):
    """Look-ahead true-peak limiter as a pure gain curve (no latency, nothing else touched).

    required = ceiling / peak  ->  hold (sliding minimum over [n-hold, n+look))
    -> release at a constant dB/s (running-min of a tilted curve) -> look-ahead
    box smoothing. By construction gain[n] <= required[n] for every sample.
    """
    r = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    L, H = int(look * SR), int(hold * SR)
    size = L + H
    pad = np.concatenate([np.ones(size), r, np.ones(size)])
    c = ndi.minimum_filter1d(pad, size, mode="nearest")           # c[n] = min pad[n - size//2 .. n - size//2 + size - 1]
    d = size // 2 - H
    m = c[size + d: size + d + len(r)]                            # m[n] = min r[n-H .. n+L-1]
    md = 20.0 * np.log10(m)
    ramp = (rel_db_s / SR) * np.arange(len(md))
    md = np.minimum.accumulate(md - ramp) + ramp
    g = np.concatenate([np.ones(L - 1), dbg(md)])
    cs = np.concatenate([[0.0], np.cumsum(g)])
    return (cs[L:] - cs[:-L]) / L                                 # g[n] = mean m[n-L+1 .. n]


# Tilt correction measured against the reference curve in QA.md: the summed pads/brass/choir sat
# ~3 dB proud around 300-600 Hz and the top octave ~2.5 dB shy.
MASTER_EQ = (("peak", 420.0, -2.5, 0.5), ("hishelf", 6500.0, 2.5, 0.7))


def _bus_linear(x):
    y = hp(x, 22.0, 2)                                             # subsonic / DC
    mid, side = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])
    side = hp(side, 110.0, 2)                                      # mono low end
    return eq(np.stack([mid + side, mid - side], axis=1), MASTER_EQ)


def master(stems, target_lufs=-14.0, ceiling_db=-1.0, fade=None, bounds=None, gaps=(), log=print):
    """Returns (master, mastered stems, report).

    The filters run separately on each segment between cut points and the silences are
    re-gated afterwards, so no filter ringing leaks into them: the gaps stay digital black."""
    N = len(next(iter(stems.values())))
    edges = [n_of(t) for t in (bounds or (0.0, N / SR))]
    proc = {}
    for k, x in stems.items():
        y = np.concatenate([_bus_linear(x[a:b]) for a, b in zip(edges[:-1], edges[1:])])
        for g0, g1 in gaps:
            i0, i1, kf = n_of(g0), n_of(g1), int(0.003 * SR)
            y[i0 - kf:i0] *= (1.0 - cosfade(kf))[:, None]
            y[i0:i1] = 0.0
        proc[k] = y
    mix = sum(proc.values())
    g = np.ones(N)
    if fade is not None:
        i0, i1 = n_of(fade[0]), min(N, n_of(fade[1]))
        g[i0:i1] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(i1 - i0) + 1) / (i1 - i0))
        g[i1:] = 0.0
    ref = float(dbg(-18.0 - lufs(mix * g[:, None])))               # fixed operating level for the compressor
    gc = glue_curve(mix * ref)
    g = g * ref * gc
    pre = mix * g[:, None]
    env = tp_envelope(pre)
    trim = float(dbg(target_lufs - lufs(pre)))
    ceil = float(dbg(ceiling_db - 0.12))
    for _ in range(4):                                             # outer loop: verify true peak at 8x
        for _ in range(8):                                         # inner loop: hit the loudness target
            gl = limiter_gain(env * trim, ceil)
            out = pre * (trim * gl)[:, None]
            err = target_lufs - lufs(out)
            if abs(err) < 0.02:
                break
            trim *= float(dbg(err))
        tp = true_peak_db(out, 8)
        if tp <= ceiling_db - 0.02:
            break
        ceil *= float(dbg(ceiling_db - 0.05 - tp))
    total = g * trim * gl
    out_stems = {k: v * total[:, None] for k, v in proc.items()}
    out = sum(out_stems.values())
    gr_lim = to_db(gl)
    rep = dict(lufs=round(lufs(out), 2), true_peak_dbtp=round(true_peak_db(out, 8), 2),
               glue_max_gr_db=round(float(-to_db(gc.min())), 2), limiter_max_gr_db=round(float(-gr_lim.min()), 2),
               limiter_time_over_1db_pct=round(float(100.0 * np.mean(gr_lim < -1.0)), 2),
               makeup_db=round(float(to_db(ref * trim)), 2))
    log("    master: %s" % rep)
    return out, out_stems, rep


# =============================================================================
# 8. EXPORT
# =============================================================================
def write_wav(path: Path, x, key):
    """48 kHz / 24-bit PCM with seeded TPDF dither."""
    rng = rng_for("dither", key)
    lsb = 1.0 / (1 << 23)
    y = np.clip(x + (rng.random(x.shape) - rng.random(x.shape)) * lsb, -1.0, 1.0 - lsb)
    sf.write(str(path), y, SR, subtype="PCM_24")


def encode(src: Path, dst: Path):
    codec = {".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
             ".m4a": ["-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"]}[dst.suffix]
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(src),
                    "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", *codec, str(dst)], check=True)


def cue_sheet(theme: Score, bed: Score, reports: dict) -> dict:
    def sm(t):
        return dict(t=round(float(t), 6), sample=n_of(t))

    def piece(S: Score, n_bars):
        bars = [dict(bar=b, **sm(bt(b)), chord=S.chords.get(b, ""),
                     beats=[round(bt(b, k), 6) for k in (1, 2, 3, 4)]) for b in range(1, n_bars + 1)]
        cues = []
        for c in sorted(S.cues, key=lambda c: c["t"]):
            bar = int(c["t"] // BAR) + 1
            cues.append(dict(**sm(c["t"]), bar=bar, beat=round((c["t"] - bt(bar)) / BEAT + 1, 3), kind=c["kind"],
                             label=c["label"], strength=c["strength"]))
        return dict(length_s=S.length, length_samples=n_of(S.length), bars=bars,
                    sections=[dict(name=x["name"], bars=x["bars"], start=sm(x["start"]), end=sm(min(x["end"], S.length)),
                                   label=x["label"]) for x in S.sections],
                    cues=cues, silences=[dict(start=sm(a), end=sm(b)) for a, b in S.gaps])

    return dict(
        title="MARROW - Printed bone. Liquid marrow.", key="D minor", bpm=BPM, time_signature="4/4",
        sample_rate=SR, beat_s=BEAT, bar_s=BAR, bar_samples=n_of(BAR), seed=SEED,
        note="All cue times are exact sample positions at 48 kHz; bar b starts at (b-1)*2.5 s.",
        theme=dict(file="marrow_theme.wav", music_end=sm(T_MUSIC), tail_end=sm(T_TOTAL), **piece(theme, N_BARS),
                   accent_grid_s=[c["t"] for c in sorted(theme.cues, key=lambda c: c["t"])],
                   master=reports.get("theme", {})),
        bed=dict(file="marrow_bed_30s.wav", **piece(bed, 12), master=reports.get("bed", {})),
        stems=["stems/%s.wav" % s for s in STEMS],
    )


def print_stats(stats, windows, log=print):
    names = [k for k in stats if not k.startswith("_")]
    log("    %-9s %6s " % ("track", "peak") + " ".join("%7s" % ("%g-%g" % w) for w in windows))
    for nme in names:
        row = []
        for w0, w1 in windows:
            z = stats[nme][int(w0 * 10): int(w1 * 10)]
            row.append("%7.1f" % (-0.691 + 10 * np.log10(np.mean(z) + 1e-20)) if np.any(z > 0) else "      -")
        log("    %-9s %6.1f " % (nme, to_db(stats["_peak"][nme])) + " ".join(row))


def build(which, out: Path, do_encode=True, show_stats=False, log=print):
    tic = time.time()
    score = compose_theme() if which == "theme" else compose_bed()
    log("[%s] %d events on %d tracks" % (which, len(score.events), len({e.track for e in score.events})))
    stats = {} if show_stats else None
    stems = render(score, log=log if show_stats else (lambda *a: None), stats=stats)
    if show_stats:
        wins = ((0, 7.5), (7.5, 20), (20, 35), (35, 60), (60, 75), (75, 84)) if which == "theme" else ((0, 7.5), (7.5, 27.5), (27.5, 30))
        print_stats(stats, wins, log)
    log("[%s] rendered in %.1fs" % (which, time.time() - tic))
    if which == "theme":
        # ceiling 0.5 dB under the -1 dBTP limit: leaves room for MP3 / AAC decoder overshoot
        mst, mstems, rep = master(stems, target_lufs=-14.0, ceiling_db=-1.5, fade=(81.0, T_TOTAL),
                                  bounds=score.bounds, gaps=score.gaps, log=log)
        base = out / "marrow_theme"
        (out / "stems").mkdir(exist_ok=True)
        for s in STEMS:
            write_wav(out / "stems" / ("%s.wav" % s), mstems[s], "stem-" + s)
    else:
        mst, mstems, rep = master(stems, target_lufs=-16.0, ceiling_db=-1.5, fade=(28.6, T_BED),
                                  bounds=score.bounds, gaps=score.gaps, log=log)
        base = out / "marrow_bed_30s"
    write_wav(base.with_suffix(".wav"), mst, which)
    if do_encode:
        for ext in ((".mp3", ".m4a") if which == "theme" else (".m4a",)):
            encode(base.with_suffix(".wav"), base.with_suffix(ext))
    log("[%s] done in %.1fs" % (which, time.time() - tic))
    return score, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=("theme", "bed"), help="render just one piece")
    ap.add_argument("--out", default=str(HERE), help="output directory (default: next to this file)")
    ap.add_argument("--no-qa", action="store_true", help="skip qa.py (measurements, plots, QA.md)")
    ap.add_argument("--no-encode", action="store_true", help="skip mp3 / m4a")
    ap.add_argument("--stats", action="store_true", help="print per-track loudness by section")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reports, scores = {}, {}
    for which in ("theme", "bed"):
        if a.only in (None, which):
            scores[which], reports[which] = build(which, out, do_encode=not a.no_encode, show_stats=a.stats)
    if a.only is None:
        (out / "cues.json").write_text(json.dumps(cue_sheet(scores["theme"], scores["bed"], reports), indent=2) + "\n")
        if not a.no_qa:
            import qa
            qa.main(out)


if __name__ == "__main__":
    main()
