#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA for the MARROW score: nobody listened to this render, so everything is measured.

Reads the delivered files (never the synthesis code), measures them with independent
implementations where possible, draws the plots and writes QA.md.

    /opt/venv/bin/python qa.py            # after score.py has rendered

Checks: format/duration, integrated loudness (own BS.1770-4 + pyloudnorm + ffmpeg ebur128),
true peak (8x oversampling + ffmpeg), clipping, DC, dynamic arc per section, spectral
balance against a reference tilt, sub energy at the hits, stereo correlation / mono
fold-down, blind onset detection against the cue grid (two detectors), the silences
before the cut points, stems null test, lossy encodes, file hashes.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import hashlib
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import soundfile as sf
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
SR = 48000

# ---- chart style (light surface, recessive chrome, one or two series per chart) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list(          # sequential: one hue, light -> dark
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#071d3b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def db(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def pw(x, floor=1e-20):
    return 10.0 * np.log10(np.maximum(x, floor))


# =============================================================================
# measurements
# =============================================================================
_K1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
_K2 = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def k_weight(x):
    return sig.lfilter(_K2[0], _K2[1], sig.lfilter(_K1[0], _K1[1], x, axis=0), axis=0)


def k_blocks(x, win, hop):
    y = k_weight(x)
    p = np.sum(y * y, axis=1)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    w, h = int(round(win * SR)), int(round(hop * SR))
    st = np.arange(0, len(p) - w + 1, h)
    return st / SR + win / 2.0, (cs[st + w] - cs[st]) / w


def lufs_integrated(x):
    """ITU-R BS.1770-4 gated integrated loudness (own implementation)."""
    _, z = k_blocks(x, 0.4, 0.1)
    l = -0.691 + pw(z)
    keep = l > -70.0
    rel = -0.691 + pw(np.mean(z[keep])) - 10.0
    keep &= l > rel
    return float(-0.691 + pw(np.mean(z[keep])))


def lufs_ungated(x):
    y = k_weight(x)
    return float(-0.691 + pw(np.mean(np.sum(y * y, axis=1))))


def loudness_range(x):
    """EBU Tech 3342 LRA from 3 s short-term loudness."""
    _, z = k_blocks(x, 3.0, 0.1)
    l = -0.691 + pw(z)
    l = l[l > -70.0]
    if len(l) == 0:
        return 0.0
    rel = -0.691 + pw(np.mean(10 ** ((l + 0.691) / 10))) - 20.0
    l = l[l > rel]
    return float(np.percentile(l, 95) - np.percentile(l, 10))


def true_peak_db(x, os=8):
    return float(db(np.max(np.abs(sig.resample_poly(x, os, 1, axis=0)))))


def ffmpeg_ebur128(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-threads", "2", "-i", str(path), "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    g = lambda pat: float(re.search(pat, tail).group(1))
    return dict(I=g(r"I:\s+(-?[\d.]+) LUFS"), LRA=g(r"LRA:\s+(-?[\d.]+) LU"), TP=g(r"Peak:\s+(-?[\d.]+) dBFS"))


def decode(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(path), "-f", "f64le",
                        "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype="<f8").reshape(-1, 2)


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=codec_name,sample_rate,channels,bit_rate,bits_per_raw_sample:format=duration,bit_rate",
                        "-of", "json", str(path)], capture_output=True, text=True)
    j = json.loads(r.stdout)
    st, fm = j["streams"][0], j["format"]
    return dict(codec=st.get("codec_name"), sr=int(st.get("sample_rate", 0)), ch=st.get("channels"),
                kbps=round(float(fm.get("bit_rate", 0)) / 1000.0), duration=float(fm.get("duration", 0)))


def octave_bands(x, a, b):
    """Octave-band power (dB, arbitrary ref) of the mono sum between a and b seconds."""
    seg = x[int(a * SR): int(b * SR)].mean(axis=1)
    f, p = sig.welch(seg, SR, nperseg=16384)
    cs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    return cs, np.array([pw(p[(f >= c / 2 ** 0.5) & (f < c * 2 ** 0.5)].sum() * (f[1] - f[0])) for c in cs])


def third_octaves(x, a, b):
    seg = x[int(a * SR): int(b * SR)].mean(axis=1)
    f, p = sig.welch(seg, SR, nperseg=32768)
    cs = 1000.0 * 2.0 ** (np.arange(-16, 14) / 3.0)          # 25 Hz .. 20 kHz
    return cs, np.array([pw(p[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() * (f[1] - f[0])) for c in cs])


def band_rms_db(x, lo, hi):
    sos = sig.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
    y = sig.sosfilt(sos, x, axis=0)
    return float(db(np.sqrt(np.mean(y * y))))


def correlation_series(x, win=0.4, hop=0.2):
    w, h = int(win * SR), int(hop * SR)
    t, c = [], []
    for i in range(0, len(x) - w, h):
        l, r = x[i:i + w, 0], x[i:i + w, 1]
        d = np.sqrt(np.sum(l * l) * np.sum(r * r))
        if d > 1e-7:                                    # skip digital silence
            t.append((i + w / 2) / SR)
            c.append(np.sum(l * r) / d)
    return np.array(t), np.array(c)


# ---- harmony and rhythm -------------------------------------------------------
PCS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
CHORD_TONES = {"Dm": "D F A", "Bb": "Bb D F", "F": "F A C", "C": "C E G", "Gm": "G Bb D", "A": "A C# E D",
               "D5": "D A", "Bb/D": "Bb D F", "C/D": "C E G D"}
_KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma(seg, fmin=110.0, fmax=2200.0, nfft=32768):
    """Share of spectral energy per pitch class (A = 440 Hz equal temperament, bins weighted by how
    close they sit to a tempered pitch). 110 Hz and up: below that FFT bins are wider than a semitone."""
    f, _, z = sig.stft(seg, SR, window="hann", nperseg=nfft, noverlap=nfft * 3 // 4, boundary=None, padded=False)
    p = (np.abs(z) ** 2).mean(axis=1)
    k = (f >= fmin) & (f <= fmax)
    m = 69.0 + 12.0 * np.log2(f[k] / 440.0)
    c = np.zeros(12)
    np.add.at(c, np.round(m).astype(int) % 12, p[k] * np.exp(-0.5 * ((m - np.round(m)) / 0.25) ** 2))
    return c / c.sum()


def estimate_key(c):
    """Krumhansl-Kessler key-profile correlation; returns [(r, name), ...] best first."""
    cand = [(float(np.corrcoef(np.roll(_KK_MAJ, k), c)[0, 1]), PCS[k] + " major") for k in range(12)]
    cand += [(float(np.corrcoef(np.roll(_KK_MIN, k), c)[0, 1]), PCS[k] + " minor") for k in range(12)]
    return sorted(cand, reverse=True)


def modulation_peaks(x, a, b, rates=(1.6, 3.2, 6.4)):
    """Prominence (dB over the median 1-12 Hz floor) of the envelope modulation at the beat, 8th and
    16th-note rates: is there a pulse in the texture, or just a sustained wash?"""
    sos = sig.butter(2, 150.0, "highpass", fs=SR, output="sos")
    y = sig.sosfilt(sos, x[int(a * SR): int(b * SR)], axis=0)
    env = np.sqrt(np.convolve(np.sum(y * y, axis=1), np.ones(480) / 480.0, mode="same"))[::480]      # 100 Hz envelope
    env = (env - env.mean()) * np.hanning(len(env))
    f, p = sig.periodogram(env, 100.0)
    floor = pw(np.median(p[(f > 1.0) & (f < 12.0)]))
    out = []
    for r in rates:
        k = int(np.argmin(np.abs(f - r)))
        out.append(float(pw(p[k - 1: k + 2].max()) - floor))
    return out


# ---- onset detection --------------------------------------------------------
ONSET_BANDS = (("high (>2 kHz)", ("highpass", 2000.0), 0.001), ("mid (0.3-2 kHz)", ("bandpass", (300.0, 2000.0)), 0.002),
               ("full (>25 Hz)", ("highpass", 25.0), 0.005))


def band_novelty(x, spec, wfast, pad):
    """Leading-edge novelty of one causal band: fast-window energy (dB) minus the energy of the
    20 ms that ended 1 ms before the fast window. No look-ahead, so it cannot fire early."""
    kind, fc = spec
    sos = sig.butter(2, fc, kind, fs=SR, output="sos")
    y = sig.sosfilt(sos, np.concatenate([np.zeros((pad, x.shape[1])), x]), axis=0)
    p = np.sum(y * y, axis=1)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    n = np.arange(len(p))
    wf, ws, g = int(wfast * SR), int(0.020 * SR), int(0.001 * SR)
    fast = (cs[n + 1] - cs[np.maximum(n + 1 - wf, 0)]) / wf
    slow = (cs[np.maximum(n + 1 - wf - g, 0)] - cs[np.maximum(n + 1 - wf - g - ws, 0)]) / ws
    eps = 10 ** (-90 / 10)
    return pw(fast + eps) - pw(slow + eps), pw(fast + eps)


def detect_onsets(x, theta=8.0, refractory=0.030):
    """Detector A (blind): an onset is the first sample where any band's novelty crosses `theta` dB.
    Returns [(time, strength_dB, band)], merged across bands within 15 ms (earliest time wins)."""
    pad = int(0.1 * SR)
    ev, nov = [], {}
    for name, spec, wfast in ONSET_BANDS:
        d, e = band_novelty(x, spec, wfast, pad)
        nov[name] = (d[pad:], e[pad:])
        above = d >= theta
        idx = np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))
        last = -10 ** 9
        for i in idx:
            if i - last >= refractory * SR:
                ev.append(((i - pad) / SR, float(d[i: i + int(0.02 * SR)].max()), name))
            last = i
    ev.sort()
    merged = []
    for t, s, b in ev:
        if merged and t - merged[-1][0] < 0.015:
            if s > merged[-1][1]:
                merged[-1] = (merged[-1][0], s, merged[-1][2])
        else:
            merged.append((t, s, b))
    return merged, nov


def spectral_flux_onsets(x, nfft=256, hop=24, lag=4):
    """Detector B (blind, textbook): log-magnitude spectral flux on a 5.3 ms Hann STFT with a 0.5 ms
    hop; peaks = local maxima (+/-25 ms) at least 6 scaled-MADs above the 1 s running median."""
    m = np.concatenate([np.zeros(nfft), x.mean(axis=1), np.zeros(nfft)])
    f, t, Z = sig.stft(m, SR, window="hann", nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    L = np.log10(1.0 + 1000.0 * np.abs(Z))
    flux = np.maximum(L[:, lag:] - L[:, :-lag], 0.0).sum(axis=0)
    tt = t[lag:] - nfft / SR - (lag * hop / SR) / 2.0       # frame centre, referred to the middle of the difference
    k = int(0.5 * SR / hop)
    from scipy.ndimage import maximum_filter1d, median_filter
    med = median_filter(flux, size=2 * k + 1, mode="nearest")
    mad = median_filter(np.abs(flux - med), size=2 * k + 1, mode="nearest") * 1.4826 + 1e-9
    z = (flux - med) / mad
    r = int(0.025 * SR / hop)
    peaks = np.flatnonzero((flux == maximum_filter1d(flux, 2 * r + 1)) & (z >= 6.0))
    return tt[peaks], z[peaks]


# =============================================================================
# plots
# =============================================================================
def _spectrogram(ax, x, t0=0.0, nfft=4096, hop=512, fmin=25.0, fmax=20000.0, dyn=80.0, vmax=None):
    m = x.mean(axis=1)
    f, t, s = sig.spectrogram(m, SR, window="hann", nperseg=nfft, noverlap=nfft - hop, mode="magnitude")
    s = db(s, 1e-9)
    k = (f >= fmin) & (f <= fmax)
    top = s.max() if vmax is None else vmax
    im = ax.pcolormesh(t + t0, f[k], s[k], vmin=top - dyn, vmax=top, cmap=SEQ, shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(fmin, fmax)
    ax.set_yticks([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_yticklabels(["31", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax.minorticks_off()
    ax.set_ylabel("frequency (Hz)")
    return im, top


def _mark_sections(ax, sections, y=1.015, lines=True):
    for s in sections:
        a, b = s["start"]["t"], s["end"]["t"]
        if lines and a > 0:
            ax.axvline(a, color=AXIS, lw=0.8, zorder=1)
        ax.text((a + b) / 2, y, "%s  bars %d-%d" % (s["name"], s["bars"][0], s["bars"][1]) if b - a > 6 else s["name"],
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color=INK2)


def plot_overview(x, cues, path, title):
    t_m, z_m = k_blocks(x, 0.4, 0.1)
    t_s, z_s = k_blocks(x, 3.0, 0.1)
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.28))
    ax = axs[0]
    n = len(x)
    step = 480
    env = np.abs(x[: n // step * step]).max(axis=1).reshape(-1, step).max(axis=1)
    tt = (np.arange(len(env)) + 0.5) * step / SR
    ax.fill_between(tt, -env, env, color=BLUE, lw=0, alpha=0.55)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("sample peak (full scale = 1)")
    ax.set_title(title + "   waveform envelope (10 ms peak)", pad=18)
    ax.grid(axis="y")
    _mark_sections(ax, cues["sections"])
    ax = axs[1]
    ax.plot(t_m, -0.691 + pw(z_m), color=NEUTRAL, lw=1.2, label="momentary (400 ms)")
    ax.plot(t_s, -0.691 + pw(z_s), color=BLUE, lw=2.0, label="short-term (3 s)")
    ax.set_ylim(-60, -4)
    ax.set_ylabel("loudness (LUFS)")
    ax.set_xlabel("time (s)")
    ax.set_title("Loudness over time", pad=6)
    ax.grid(axis="y")
    for c in cues["cues"]:
        ax.axvline(c["t"], color=GRID, lw=0.8, zorder=0)
    for s in cues["sections"]:
        if s["start"]["t"] > 0:
            ax.axvline(s["start"]["t"], color=AXIS, lw=0.8, zorder=1)
    ax.legend(loc="lower center", ncol=2)
    ax.set_xlim(0, len(x) / SR)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram(x, cues, path, title):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    im, _ = _spectrogram(ax, x, nfft=4096, hop=1024)
    ax.set_xlabel("time (s)")
    ax.set_title(title, pad=18)
    _mark_sections(ax, cues["sections"], lines=False)
    for c in cues["cues"]:
        ax.plot([c["t"]], [1.0], marker="v", color=INK2, ms=4, transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("magnitude (dB re. peak bin)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    ax.set_xlim(0, len(x) / SR)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_zooms(x, windows, path):
    fig, axs = plt.subplots(len(windows), 1, figsize=(13, 3.5 * len(windows)), gridspec_kw=dict(hspace=0.42))
    for ax, (name, a, b) in zip(np.atleast_1d(axs), windows):
        _spectrogram(ax, x[int(a * SR): int(b * SR)], t0=a, nfft=2048, hop=128, fmin=30.0, fmax=16000.0, dyn=70.0)
        ax.set_title(name + "   [70 dB below this panel's own peak]")
        ax.set_xlim(a, b)
        for bar_t in np.arange(np.ceil(a / 2.5) * 2.5, b + 1e-9, 2.5):
            ax.plot([bar_t], [1.0], marker="v", color=INK2, ms=4, transform=ax.get_xaxis_transform(), clip_on=False)
    np.atleast_1d(axs)[-1].set_xlabel("time (s)   -   markers: bar lines (2.5 s)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_sections(rows, path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for key, col, lab, dy in (("rms", ORANGE, "RMS (dBFS)", -1.0), ("lufs", BLUE, "K-weighted loudness (LUFS, ungated)", 1.0)):
        for i, r in enumerate(rows):
            ax.plot([r["t0"], r["t1"]], [r[key], r[key]], color=col, lw=2.6, label=lab if i == 0 else None)
            ax.text((r["t0"] + r["t1"]) / 2, r[key] + dy * 1.1, "%.1f" % r[key], ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=8, color=INK2)
    for r in rows:
        if r["t0"] > 0:
            ax.axvline(r["t0"], color=GRID, lw=0.8, zorder=0)
        ax.text((r["t0"] + r["t1"]) / 2, 1.02, r["name"], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.set_xlim(0, rows[-1]["t1"])
    lo = min(min(r["rms"], r["lufs"]) for r in rows)
    ax.set_ylim(lo - 5, -6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("level (dB)")
    ax.set_title("Dynamic arc: level per section", pad=20)
    ax.grid(axis="y")
    ax.legend(loc="lower center", ncol=2)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(curves, target, tol, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cs = np.array(sorted(target))
    tv = np.array([target[c] for c in cs])
    ax.fill_between(cs, tv - tol, tv + tol, color=GRID, lw=0, label="reference tilt +/- %g dB" % tol, zorder=1)
    ax.plot(cs, tv, color=MUTED, lw=1.2, zorder=2)
    for (name, f3, v3), col in zip(curves, (BLUE, ORANGE)):
        ax.plot(f3, v3, color=col, lw=2.0, label=name, zorder=3)
        ax.text(f3[-1] * 1.04, v3[-1], name.split(" ")[0], color=INK2, fontsize=8, va="center")
    ax.axvspan(2000, 5000, color="#f0efec", lw=0, zorder=0)
    ax.text(3160, 0.02, "2-5 kHz", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(22, 26000)
    ax.set_xticks([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(["31", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax.minorticks_off()
    ax.set_ylim(-42, 12)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("octave-band level re. 125 Hz octave (dB)")
    ax.set_title("Spectral balance of the loud sections vs. reference trailer tilt")
    ax.grid(axis="y")
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(t, c, cues, path):
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axhline(0.0, color=AXIS, lw=1.0, zorder=1)
    ax.plot(t, c, color=BLUE, lw=1.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, cues["length_s"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L/R correlation (400 ms)")
    ax.set_title("Stereo correlation (+1 = mono, 0 = fully wide, < 0 = cancels in mono)", pad=18)
    ax.grid(axis="y")
    _mark_sections(ax, cues["sections"])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_onsets(x, rows, nov, path):
    n = len(rows)
    cols = 5
    fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(13, 2.5 * ((n + cols - 1) // cols)), sharey=True,
                            gridspec_kw=dict(hspace=0.62, wspace=0.08))
    e_hi = nov["high (>2 kHz)"][1]
    for ax in axs.ravel():
        ax.set_visible(False)
    for ax, r in zip(axs.ravel(), rows):
        ax.set_visible(True)
        i0 = int(round(r["cue"] * SR))
        a, b = i0 - int(0.012 * SR), i0 + int(0.012 * SR)
        idx = np.arange(max(a, 0), min(b, len(e_hi)))
        ax.axvline(0.0, color=AXIS, lw=1.0, zorder=1)
        ax.plot((idx - i0) / SR * 1000.0, e_hi[idx], color=BLUE, lw=1.6, zorder=3)
        if r["err_a"] is not None and abs(r["err_a"]) < 12:
            j = int(round((r["cue"] + r["err_a"] / 1000.0) * SR))
            ax.plot([r["err_a"]], [e_hi[min(max(j, 0), len(e_hi) - 1)]], marker="o", ms=7, color=ORANGE, mec=SURFACE, mew=1.5, zorder=4)
        ax.set_title("%.1f s   %+.2f ms" % (r["cue"], r["err_a"] if r["err_a"] is not None else float("nan")), fontsize=9)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-95, 0)
        ax.grid(axis="y")
        ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("ms from cue")
    for ax in axs[:, 0]:
        ax.set_ylabel(">2 kHz energy, 1 ms (dBFS)", fontsize=8)
    fig.suptitle("Cue hits: high-band energy around each cue time (line at the grid position, dot = detected onset, title = error)",
                 x=0.125, y=0.995, ha="left", fontsize=10.5, fontweight="bold")
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def plot_stems(stems, cues, path):
    names = list(stems)
    fig, axs = plt.subplots(len(names), 1, figsize=(13, 1.35 * len(names) + 0.8), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.25))
    for ax, nme in zip(axs, names):
        t, z = k_blocks(stems[nme], 1.0, 0.25)
        ax.plot(t, -0.691 + pw(z), color=BLUE, lw=1.6)
        ax.set_ylim(-62, -8)
        ax.set_yticks([-50, -30, -10])
        ax.grid(axis="y")
        ax.text(0.006, 0.84, nme, transform=ax.transAxes, fontsize=9, color=INK, fontweight="bold")
        for s in cues["sections"]:
            if s["start"]["t"] > 0:
                ax.axvline(s["start"]["t"], color=GRID, lw=0.8, zorder=0)
    axs[0].set_title("Stems: 1 s loudness (LUFS) of each delivered stem", pad=6)
    axs[-1].set_xlabel("time (s)")
    axs[-1].set_xlim(0, cues["length_s"])
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def plot_harmony(C, bars, path):
    fig, ax = plt.subplots(figsize=(13, 4.4))
    order = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1]                     # D at the bottom: the tonic is the floor
    im = ax.imshow(C[:, order].T * 100.0, origin="lower", aspect="auto", cmap=SEQ, vmin=0, vmax=60,
                   extent=(0.5, len(bars) + 0.5, -0.5, 11.5), interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels([PCS[i] for i in order])
    ax.set_xticks(range(1, len(bars) + 1))
    ax.tick_params(axis="x", labelsize=7)
    for b in bars:
        ax.text(b["bar"], 1.015, b["chord"], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK2,
                rotation=0 if len(b["chord"]) < 3 else 90)
    for xg in np.arange(1.5, len(bars), 1.0):
        ax.axvline(xg, color=SURFACE, lw=1.0)
    for yg in np.arange(0.5, 11, 1.0):
        ax.axhline(yg, color=SURFACE, lw=1.0)
    ax.set_xlabel("bar")
    ax.set_ylabel("pitch class")
    ax.set_title("Harmony measured from the audio: energy per pitch class in each bar (intended chord above)", pad=30)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("share of 110 Hz-2.2 kHz energy (%)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_bed(x, cues, path):
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5], hspace=0.2))
    t_m, z_m = k_blocks(x, 0.4, 0.1)
    t_s, z_s = k_blocks(x, 3.0, 0.1)
    ax = axs[0]
    ax.plot(t_m, -0.691 + pw(z_m), color=NEUTRAL, lw=1.2, label="momentary (400 ms)")
    ax.plot(t_s, -0.691 + pw(z_s), color=BLUE, lw=2.0, label="short-term (3 s)")
    ax.set_ylim(-60, -4)
    ax.set_ylabel("loudness (LUFS)")
    ax.grid(axis="y")
    ax.legend(loc="lower center", ncol=2)
    ax.set_title("marrow_bed_30s.wav   loudness and spectrogram (soft end hit at 27.5 s)")
    for c in cues["cues"]:
        ax.axvline(c["t"], color=AXIS, lw=0.8, zorder=0)
    im, _ = _spectrogram(axs[1], x, nfft=4096, hop=512)
    axs[1].set_xlabel("time (s)")
    axs[1].set_xlim(0, len(x) / SR)
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# report
# =============================================================================
TARGET = {31.5: -2.0, 63: 1.0, 125: 0.0, 250: -2.0, 500: -4.5, 1000: -7.5, 2000: -10.5, 4000: -14.0, 8000: -19.0, 16000: -28.0}
TOL = 4.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_size(n):
    return "%.1f MB" % (n / 1e6) if n >= 1e6 else "%.0f kB" % (n / 1e3)


def audio_stats(x):
    peak = float(np.max(np.abs(x)))
    full = np.abs(x) >= 1.0 - 2.0 ** -22
    run = 0
    if full.any():
        f = full.any(axis=1).astype(int)
        d = np.diff(np.concatenate([[0], f, [0]]))
        run = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())
    w = SR
    nb = len(x) // w
    dc_win = np.abs(x[: nb * w].reshape(nb, w, 2).mean(axis=1)).max()
    return dict(sample_peak_db=float(db(peak)), clipped_samples=int(full.sum()), longest_fullscale_run=run,
                dc=[float(v) for v in x.mean(axis=0)], dc_db=float(db(np.abs(x.mean(axis=0)).max())),
                dc_worst_1s_db=float(db(dc_win)))


def main(out: Path = HERE):
    out = Path(out)
    cues = json.loads((out / "cues.json").read_text())
    th, bd = cues["theme"], cues["bed"]
    res = {}
    checks = []                                   # (name, target, measured, pass)

    def check(name, target, measured, ok):
        checks.append((name, target, measured, bool(ok)))
        return ok

    x, sr = sf.read(str(out / "marrow_theme.wav"), always_2d=True)
    info = sf.info(str(out / "marrow_theme.wav"))
    assert sr == SR

    # ---- 1. format ------------------------------------------------------------
    check("Theme format", "48 kHz / 24-bit / stereo / 84.000 s",
          "%d Hz / %s / %d ch / %.3f s (%d samples)" % (info.samplerate, info.subtype, info.channels, len(x) / SR, len(x)),
          info.samplerate == 48000 and info.subtype == "PCM_24" and info.channels == 2 and len(x) == 84 * SR)

    # ---- 2. loudness and peaks ---------------------------------------------------
    import pyloudnorm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l_pyln = float(pyloudnorm.Meter(SR).integrated_loudness(x))
    l_own = lufs_integrated(x)
    ff = ffmpeg_ebur128(out / "marrow_theme.wav")
    tp = true_peak_db(x, 8)
    st = audio_stats(x)
    lra = loudness_range(x)
    _, zm = k_blocks(x, 0.4, 0.1)
    _, zs = k_blocks(x, 3.0, 0.1)
    res["loud"] = dict(own=l_own, pyln=l_pyln, ff=ff, tp=tp, lra=lra, max_m=float(-0.691 + pw(zm.max())),
                       max_s=float(-0.691 + pw(zs.max())), **st)
    check("Integrated loudness", "-14 LUFS +/- 0.5",
          "%.2f (own BS.1770-4) / %.2f (pyloudnorm) / %.1f (ffmpeg ebur128)" % (l_own, l_pyln, ff["I"]),
          all(abs(v + 14.0) <= 0.5 for v in (l_own, l_pyln, ff["I"])))
    check("True peak", "<= -1.0 dBTP", "%.2f dBTP (8x oversampled) / %.1f (ffmpeg)" % (tp, ff["TP"]), tp <= -1.0 and ff["TP"] <= -1.0)
    check("Clipping", "0 full-scale samples", "%d full-scale samples, sample peak %.2f dBFS" % (st["clipped_samples"], st["sample_peak_db"]),
          st["clipped_samples"] == 0)
    check("DC offset", "< -60 dBFS", "L %+.1e, R %+.1e (%.0f dBFS); worst 1 s window %.0f dBFS" % (st["dc"][0], st["dc"][1], st["dc_db"], st["dc_worst_1s_db"]),
          st["dc_db"] < -60 and st["dc_worst_1s_db"] < -50)

    # ---- 3. dynamic arc -----------------------------------------------------------
    rows = []
    for s in th["sections"]:
        a, b = s["start"]["t"], min(s["end"]["t"], 80.0)
        seg = x[int(a * SR): int(b * SR)]
        t_, z_ = k_blocks(seg, 0.4, 0.1)
        rows.append(dict(name="%s (bars %d-%d)" % (s["name"], s["bars"][0], s["bars"][1]), t0=a, t1=b, label=s["label"],
                         rms=float(db(np.sqrt(np.mean(seg ** 2)))), lufs=lufs_ungated(seg), max_m=float(-0.691 + pw(z_.max())),
                         peak=float(db(np.max(np.abs(seg))))))
    seg = x[80 * SR:]
    rows.append(dict(name="tail", t0=80.0, t1=84.0, label="reverb tail", rms=float(db(np.sqrt(np.mean(seg ** 2)))),
                     lufs=lufs_ungated(seg), max_m=float(-0.691 + pw(k_blocks(seg, 0.4, 0.1)[1].max())), peak=float(db(np.max(np.abs(seg))))))
    res["sections"] = rows
    r_ = [r["rms"] for r in rows]
    check("Dynamic arc (RMS per section)", "I < II < III < IV < V, VI and tail fall away",
          " / ".join("%.1f" % v for v in r_) + " dBFS",
          r_[0] < r_[1] < r_[2] < r_[3] < r_[4] and r_[6] < r_[5] - 6 and r_[4] - r_[3] >= 1.0 and r_[3] - r_[0] >= 12)
    last = x[int(83.5 * SR):]
    check("Tail ends in silence", "last 0.5 s < -60 dBFS RMS", "%.1f dBFS RMS, final sample %.1e" % (db(np.sqrt(np.mean(last ** 2))), np.abs(x[-1]).max()),
          db(np.sqrt(np.mean(last ** 2))) < -60)

    # ---- 4. silences before the cuts ------------------------------------------------
    sil = []
    for g in th["silences"]:
        seg = x[g["start"]["sample"]: g["end"]["sample"]]
        sil.append(dict(a=g["start"]["t"], b=g["end"]["t"], ms=1000 * (g["end"]["t"] - g["start"]["t"]),
                        rms=float(db(np.sqrt(np.mean(seg ** 2)), 1e-9)), peak=float(db(np.max(np.abs(seg)), 1e-9))))
    res["silences"] = sil
    check("Silence before the drop", "~150 ms gap ending at 35.000 s", "%.0f ms at %.1f dBFS RMS (peak %.1f)" % (sil[0]["ms"], sil[0]["rms"], sil[0]["peak"]),
          abs(sil[0]["ms"] - 150) <= 15 and sil[0]["peak"] < -100)

    # ---- 5. spectral balance ---------------------------------------------------------
    spec = {}
    for name, a, b in (("theme (35-60 s)", 35, 60), ("lift (60-75 s)", 60, 75)):
        cs, lv = octave_bands(x, a, b)
        spec[name] = dict(cs=cs, rel=lv - lv[2])
    f3, v3 = third_octaves(x, 35, 75)
    k = (f3 >= 250) & (f3 <= 8000)
    slope, icpt = np.polyfit(np.log2(f3[k]), v3[k], 1)
    resid = v3 - (slope * np.log2(f3) + icpt)
    pres = (f3 >= 2000) & (f3 <= 5000)
    cs, lv = octave_bands(x, 35, 75)
    rel = lv - lv[2]
    dev = np.array([rel[i] - TARGET[c] for i, c in enumerate(cs)])
    res["spec"] = dict(bands=spec, slope=float(slope), presence_max=float(resid[pres].max()), rel=rel, dev=dev, cs=cs)
    check("Spectral balance, 35-75 s", "every octave within +/- %g dB of the reference tilt" % TOL,
          "worst deviation %+.1f dB at %g Hz; tilt 250 Hz-8 kHz %.1f dB/oct" % (dev[np.argmax(np.abs(dev))], cs[int(np.argmax(np.abs(dev)))], slope),
          np.all(np.abs(dev) <= TOL))
    check("No 2-5 kHz build-up", "third-octaves 2-5 kHz <= +2 dB over the fitted tilt", "max %+.1f dB" % resid[pres].max(), resid[pres].max() <= 2.0)

    # ---- 6. hits: momentary loudness and sub content -----------------------------------
    hits = []
    for c in th["cues"]:
        i0 = c["sample"]
        seg = x[i0: i0 + int(0.4 * SR)]
        seg1 = x[i0: i0 + int(1.0 * SR)]
        y = k_weight(np.concatenate([x[max(0, i0 - SR): i0], seg]))[-len(seg):]
        hits.append(dict(t=c["t"], kind=c["kind"], label=c["label"], m=float(-0.691 + pw(np.mean(np.sum(y * y, axis=1)))),
                         rms=float(db(np.sqrt(np.mean(seg ** 2)))), peak=float(db(np.max(np.abs(seg)))),
                         sub=band_rms_db(seg1, 30.0, 80.0), full=float(db(np.sqrt(np.mean(seg1 ** 2))))))
    res["hits"] = hits
    big = [h for h in hits if h["t"] >= 35.0]
    check("Sub at the hits (30-80 Hz, first 1 s)", "within 6 dB of the full-band RMS at every hit from the drop on",
          "sub minus full-band: %.1f ... %.1f dB" % (min(h["sub"] - h["full"] for h in big), max(h["sub"] - h["full"] for h in big)),
          all(h["sub"] - h["full"] >= -6.0 for h in big))
    drop = next(h for h in hits if h["kind"] == "drop")
    others = [h for h in hits if h["t"] < 75.0 and h["kind"] != "drop"]
    check("The drop is the biggest hit before the finale", "35.0 s momentary loudness >= every other hit before 75 s",
          "%.1f LUFS at 35.0 s vs. max %.1f elsewhere (final hit %.1f)" % (drop["m"], max(h["m"] for h in others), hits[-1]["m"]),
          drop["m"] >= max(h["m"] for h in others) - 0.05)

    # ---- 7. stereo ------------------------------------------------------------------------
    tc, cc = correlation_series(x)
    mono = x.mean(axis=1)
    l_mono = lufs_integrated(np.stack([mono, mono], axis=1))
    side_lo = band_rms_db(0.5 * (x[:, :1] - x[:, 1:]), 25.0, 100.0) - band_rms_db(mono[:, None], 25.0, 100.0)
    res["stereo"] = dict(mean=float(cc.mean()), min=float(cc.min()), tmin=float(tc[cc.argmin()]), p5=float(np.percentile(cc, 5)),
                         pos=float(100 * np.mean(cc > 0)), mono_loss=float(l_own - l_mono), side_lo=float(side_lo))
    check("Stereo correlation", "mostly positive (>= 95 % of 400 ms windows > 0)",
          "mean %+.2f, 5th percentile %+.2f, min %+.2f at %.1f s, %.1f %% > 0" % (cc.mean(), np.percentile(cc, 5), cc.min(), tc[cc.argmin()], 100 * np.mean(cc > 0)),
          np.mean(cc > 0) >= 0.95 and cc.mean() > 0.3)
    check("Mono fold-down", "loses < 3 LU; side below 100 Hz < -15 dB re. mid", "%.2f LU lost; low side %.1f dB" % (l_own - l_mono, side_lo),
          (l_own - l_mono) < 3.0 and side_lo < -15.0)

    # ---- 8. cue timing --------------------------------------------------------------------
    ev, nov = detect_onsets(x)
    ta = np.array([e[0] for e in ev])
    sa = np.array([e[1] for e in ev])
    rank = {int(i): r + 1 for r, i in enumerate(np.argsort(-sa))}
    tb, zb = spectral_flux_onsets(x)
    orows = []
    for c in th["cues"]:
        i = int(np.argmin(np.abs(ta - c["t"])))
        j = int(np.argmin(np.abs(tb - c["t"])))
        orows.append(dict(cue=c["t"], kind=c["kind"], err_a=float(1000 * (ta[i] - c["t"])), str_a=float(sa[i]), band=ev[i][2],
                          rank=rank[i], err_b=float(1000 * (tb[j] - c["t"])), z_b=float(zb[j])))
    res["onsets"] = dict(rows=orows, n_a=len(ev), n_b=len(tb))
    wa = max(abs(r["err_a"]) for r in orows)
    wb = max(abs(r["err_b"]) for r in orows)
    check("Cue hits on the grid (detector A, leading edge)", "all 15 within +/- 5 ms", "worst |error| %.2f ms, mean %.2f ms" % (wa, np.mean([abs(r["err_a"]) for r in orows])), wa <= 5.0)
    check("Cue hits on the grid (detector B, spectral flux)", "all 15 within +/- 5 ms", "worst |error| %.2f ms, mean %.2f ms" % (wb, np.mean([abs(r["err_b"]) for r in orows])), wb <= 5.0)

    # ---- 8b. harmony and rhythm, measured from the audio -------------------------------------
    mono_ = x.mean(axis=1)
    hrows = []
    C = np.zeros((len(th["bars"]), 12))
    for i, b in enumerate(th["bars"]):
        c = chroma(mono_[b["sample"]: b["sample"] + int(2.5 * SR)])
        C[i] = c
        want = [PCS.index(n) for n in CHORD_TONES[b["chord"]].split()]
        top = np.argsort(-c)[:3]
        hrows.append(dict(bar=b["bar"], t=b["t"], chord=b["chord"], top=[(PCS[j], float(c[j])) for j in top], share=float(c[want].sum())))
    keys = estimate_key(chroma(mono_[int(7.5 * SR): int(80 * SR)]))
    mod = {name: modulation_peaks(x, a, b_) for name, a, b_ in (("build (20-34.8 s)", 20.0, 34.8), ("theme (35-60 s)", 35.0, 60.0), ("lift (60-74.8 s)", 60.0, 74.8))}
    res["harmony"] = dict(rows=hrows, keys=keys[:3], mod=mod)
    n_h = sum(1 for r in hrows if r["share"] >= 0.5)
    check("Key (Krumhansl-Kessler on the audio)", "D minor", "%s (r = %.2f); next: %s (r = %.2f)" % (keys[0][1], keys[0][0], keys[1][1], keys[1][0]),
          keys[0][1] == "D minor")
    check("Chords follow the score", "chord tones carry >= 50 % of the pitched energy in >= 28 of 32 bars",
          "%d of 32 bars; median %.0f %%, minimum %.0f %% (bar %d)" % (n_h, 100 * np.median([r["share"] for r in hrows]), 100 * min(r["share"] for r in hrows),
                                                                       min(hrows, key=lambda r: r["share"])["bar"]), n_h >= 28)
    check("Rhythmic pulse in the texture", "8th- or 16th-note modulation >= 10 dB over the floor in build, theme and lift",
          "; ".join("%s %.0f / %.0f dB" % (k_.split(" ")[0], v[1], v[2]) for k_, v in mod.items()), all(max(v[1], v[2]) >= 10.0 for v in mod.values()))

    # ---- 9. stems ---------------------------------------------------------------------------
    stems = {}
    srows = []
    for p in cues["stems"]:
        y, _ = sf.read(str(out / p), always_2d=True)
        nme = Path(p).stem
        stems[nme] = y
        srows.append(dict(name=nme, n=len(y), peak=float(db(np.max(np.abs(y)))), lufs=lufs_ungated(y), size=(out / p).stat().st_size))
    null = float(db(np.max(np.abs(sum(stems.values()) - x))))
    res["stems"] = dict(rows=srows, null=null)
    check("Stems", "%d stems, 84.000 s each, sum nulls against the master" % len(stems),
          "residual peak %.1f dBFS" % null, null < -100 and all(r["n"] == len(x) for r in srows))

    # ---- 10. lossy encodes ---------------------------------------------------------------------
    lossy = []
    for name, ref_path in (("marrow_theme.mp3", "marrow_theme.wav"), ("marrow_theme.m4a", "marrow_theme.wav"), ("marrow_bed_30s.m4a", "marrow_bed_30s.wav")):
        if not (out / name).exists():
            continue
        ref, _ = sf.read(str(out / ref_path), always_2d=True)
        y = decode(out / name)
        pr = ffprobe(out / name)
        k = 8 * SR if "theme" in name else 4 * SR
        a0 = 34 * SR if "theme" in name else 5 * SR
        a_ = ref[a0: a0 + k].mean(axis=1)
        lagmax = 4096
        b_ = y[a0 - lagmax: a0 + k + lagmax].mean(axis=1)
        xc = sig.correlate(b_, a_, mode="valid")
        off = int(np.argmax(xc)) - lagmax
        lossy.append(dict(name=name, **pr, n=len(y), offset=off, lufs=lufs_integrated(y), tp=true_peak_db(y, 8), size=(out / name).stat().st_size))
    res["lossy"] = lossy
    if lossy:
        check("Lossy encodes", "same loudness (+/- 0.3 LU), sample-aligned, true peak <= -1.0 dBTP",
              "; ".join("%s %.2f LUFS, %+d samples, %.2f dBTP" % (r["name"].split(".")[-1], r["lufs"], r["offset"], r["tp"]) for r in lossy if "theme" in r["name"]),
              all(abs(r["offset"]) <= 1 and r["tp"] <= -1.0 for r in lossy) and all(abs(r["lufs"] + 14.0) <= 0.5 for r in lossy if "theme" in r["name"]))

    # ---- 11. bed --------------------------------------------------------------------------------
    xb, _ = sf.read(str(out / "marrow_bed_30s.wav"), always_2d=True)
    ib = sf.info(str(out / "marrow_bed_30s.wav"))
    evb, novb = detect_onsets(xb)
    tbb = np.array([e[0] for e in evb])
    ib_hit = int(np.argmin(np.abs(tbb - 27.5)))
    stb = audio_stats(xb)
    t_s, z_s = k_blocks(xb, 3.0, 0.5)
    ls = -0.691 + pw(z_s)
    res["bed"] = dict(n=len(xb), lufs=lufs_integrated(xb), ff=ffmpeg_ebur128(out / "marrow_bed_30s.wav"), tp=true_peak_db(xb, 8), lra=loudness_range(xb),
                      hit_err=float(1000 * (tbb[ib_hit] - 27.5)), hit_str=float(evb[ib_hit][1]), **stb,
                      st_start=float(ls[1]), st_peak=float(ls.max()), t_peak=float(t_s[ls.argmax()]),
                      max_m=float(-0.691 + pw(k_blocks(xb, 0.4, 0.1)[1].max())),
                      end_rms=float(db(np.sqrt(np.mean(xb[int(29.8 * SR):] ** 2)))))
    rb = res["bed"]
    check("Bed format", "48 kHz / 24-bit / stereo / exactly 30.000 s", "%d Hz / %s / %.3f s" % (ib.samplerate, ib.subtype, len(xb) / SR),
          len(xb) == 30 * SR and ib.subtype == "PCM_24")
    check("Bed loudness / peak", "-16 LUFS +/- 0.5, <= -1.0 dBTP", "%.2f LUFS (ffmpeg %.1f), %.2f dBTP, LRA %.1f LU" % (rb["lufs"], rb["ff"]["I"], rb["tp"], rb["lra"]),
          abs(rb["lufs"] + 16.0) <= 0.5 and rb["tp"] <= -1.0 and rb["clipped_samples"] == 0)
    check("Bed end hit", "onset at 27.500 s +/- 5 ms", "%+.2f ms (strength %.0f dB)" % (rb["hit_err"], rb["hit_str"]), abs(rb["hit_err"]) <= 5.0)

    # ---- plots -------------------------------------------------------------------------------------
    plot_overview(x, th, out / "qa_overview.png", "marrow_theme.wav")
    plot_spectrogram(x, th, out / "qa_spectrogram.png", "marrow_theme.wav   spectrogram of the full mix (L+R, log frequency)")
    plot_zooms(x, (("Bars 3-8: drone, pulse, four-note motif (5-20 s)", 5.0, 20.0),
                   ("Bars 13-18: riser, silence, the drop, first theme phrases (30-45 s)", 30.0, 45.0),
                   ("Bars 24-32: turnaround, second lift, final hit (57.5-80 s)", 57.5, 80.0)), out / "qa_spectrogram_zoom.png")
    plot_sections(rows, out / "qa_sections.png")
    curves = []
    for name, a, b in (("theme (35-60 s)", 35, 60), ("lift (60-75 s)", 60, 75)):
        cs_, lv_ = octave_bands(x, a, b)
        curves.append((name, np.array(cs_), lv_ - lv_[2]))
    plot_spectrum(curves, TARGET, TOL, out / "qa_spectrum.png")
    plot_correlation(tc, cc, th, out / "qa_correlation.png")
    plot_onsets(x, orows, nov, out / "qa_onsets.png")
    plot_stems(stems, th, out / "qa_stems.png")
    plot_harmony(C, th["bars"], out / "qa_harmony.png")
    plot_bed(xb, bd, out / "qa_bed.png")

    write_report(out, cues, res, checks, rows, orows)
    n_ok = sum(1 for c in checks if c[3])
    print("[qa] %d / %d checks pass" % (n_ok, len(checks)))
    for name, target, measured, ok in checks:
        print("   %s  %-48s %s" % ("PASS" if ok else "FAIL", name, measured))
    return all(c[3] for c in checks)


def write_report(out, cues, res, checks, rows, orows):
    th = cues["theme"]
    L = res["loud"]
    md = []
    w = md.append
    w("# MARROW score - QA by measurement\n")
    w("Nobody listened to this render: the machine that built it has no ears. Everything below is measured from the "
      "delivered files by `qa.py` (which reads the audio, not the synthesis code) and regenerated on every run of `score.py`.\n")
    n_ok = sum(1 for c in checks if c[3])
    w("**%d of %d checks pass.**\n" % (n_ok, len(checks)))
    w("| Check | Target | Measured | Result |")
    w("|---|---|---|---|")
    for name, target, measured, ok in checks:
        w("| %s | %s | %s | %s |" % (name, target, measured, "pass" if ok else "**FAIL**"))
    w("")

    w("## 1. Files\n")
    w("| File | Format | Duration | Size | SHA-256 (first 12) |")
    w("|---|---|---|---|---|")
    files = ["marrow_theme.wav", "marrow_theme.mp3", "marrow_theme.m4a", "marrow_bed_30s.wav", "marrow_bed_30s.m4a"] + cues["stems"]
    lossy = {r["name"]: r for r in res["lossy"]}
    for f in files:
        p = out / f
        if not p.exists():
            continue
        if f.endswith(".wav"):
            i = sf.info(str(p))
            fm, du = "WAV %d Hz %s %d ch" % (i.samplerate, i.subtype.replace("PCM_", "") + "-bit", i.channels), "%.3f s" % (i.frames / i.samplerate)
        else:
            r = lossy[f]
            fm, du = "%s %d kb/s %d Hz" % (r["codec"].upper(), r["kbps"], r["sr"]), "%.3f s decoded" % (r["n"] / SR)
        w("| `%s` | %s | %s | %s | `%s` |" % (f, fm, du, fmt_size(p.stat().st_size), sha256(p)[:12]))
    w("\nAlso: `score.py` (source), `qa.py` (this report), `cues.json` (grid and cue sheet), `qa_*.png` (plots).\n")

    w("## 2. Loudness, peaks, clipping, DC\n")
    w("| Measure | Value |")
    w("|---|---|")
    w("| Integrated loudness, own BS.1770-4 implementation | %.2f LUFS |" % L["own"])
    w("| Integrated loudness, pyloudnorm | %.2f LUFS |" % L["pyln"])
    w("| Integrated loudness, ffmpeg `ebur128` | %.1f LUFS |" % L["ff"]["I"])
    w("| Loudness range (EBU 3342), own / ffmpeg | %.1f / %.1f LU |" % (L["lra"], L["ff"]["LRA"]))
    w("| Max short-term (3 s) / max momentary (400 ms) | %.1f / %.1f LUFS |" % (L["max_s"], L["max_m"]))
    w("| True peak, 8x oversampled / ffmpeg | %.2f / %.1f dBTP |" % (L["tp"], L["ff"]["TP"]))
    w("| Sample peak | %.2f dBFS |" % L["sample_peak_db"])
    w("| Full-scale samples / longest full-scale run | %d / %d |" % (L["clipped_samples"], L["longest_fullscale_run"]))
    w("| DC offset L / R | %+.1e / %+.1e (%.0f dBFS) |" % (L["dc"][0], L["dc"][1], L["dc_db"]))
    m = th.get("master", {})
    if m:
        w("| Bus compressor, max gain reduction | %.1f dB |" % m.get("glue_max_gr_db", float("nan")))
        w("| Limiter, max gain reduction / time above 1 dB | %.1f dB / %.2f %% |" % (m.get("limiter_max_gr_db", float("nan")), m.get("limiter_time_over_1db_pct", float("nan"))))
    w("\n![waveform and loudness](qa_overview.png)\n")

    w("## 3. Dynamic arc\n")
    w("| Section | Time | RMS (dBFS) | Loudness (LUFS, ungated) | Max momentary (LUFS) | Peak (dBFS) | What happens |")
    w("|---|---|---|---|---|---|---|")
    for r in rows:
        w("| %s | %.1f-%.1f s | %.1f | %.1f | %.1f | %.1f | %s |" % (r["name"], r["t0"], r["t1"], r["rms"], r["lufs"], r["max_m"], r["peak"], r["label"]))
    w("\n![level per section](qa_sections.png)\n")
    w("Silences before the cut points (digital black apart from 24-bit dither):\n")
    w("| Window | Length | RMS | Peak |")
    w("|---|---|---|---|")
    for s in res["silences"]:
        w("| %.2f-%.2f s | %.0f ms | %.1f dBFS | %.1f dBFS |" % (s["a"], s["b"], s["ms"], s["rms"], s["peak"]))
    w("")

    w("## 4. Spectrogram\n")
    w("![spectrogram](qa_spectrogram.png)\n")
    w("Zooms at higher time resolution (2048-point window, 2.7 ms hop): individual piano notes and their echoes, the 8th/16th "
      "ostinato, drum hits as vertical strokes, chord changes every 2 bars (1 bar in the lift), the riser sweeping up into the "
      "silences, braams opening on the 2-bar downbeats.\n")
    w("![spectrogram zooms](qa_spectrogram_zoom.png)\n")

    w("## 5. Spectral balance\n")
    sp = res["spec"]
    w("Reference: a typical hybrid-orchestral trailer tilt, written as octave-band levels relative to the 125 Hz octave "
      "(about -3 dB/octave from 125 Hz to 4 kHz, a strong 63 Hz octave, an air roll-off above 8 kHz), with a +/- %g dB corridor. "
      "This curve is my stated assumption, not a measurement of a commercial master.\n" % TOL)
    w("| Octave | " + " | ".join("%g" % c for c in sp["cs"]) + " |")
    w("|---|" + "---|" * len(sp["cs"]))
    w("| reference (dB) | " + " | ".join("%+.1f" % TARGET[c] for c in sp["cs"]) + " |")
    for name, d in sp["bands"].items():
        w("| %s | " % name + " | ".join("%+.1f" % v for v in d["rel"]) + " |")
    w("| 35-75 s minus reference | " + " | ".join("%+.1f" % v for v in sp["dev"]) + " |")
    w("\nFitted tilt 250 Hz-8 kHz (third-octaves, 35-75 s): %.1f dB/octave. Largest third-octave excess over that line in "
      "2-5 kHz: %+.1f dB.\n" % (sp["slope"], sp["presence_max"]))
    w("![spectral balance](qa_spectrum.png)\n")
    w("Hits (first 400 ms for loudness and peak; first 1 s for the sub share):\n")
    w("| Cue | Kind | Momentary (LUFS) | RMS (dBFS) | Peak (dBFS) | 30-80 Hz RMS (dBFS) | Sub minus full-band (dB) |")
    w("|---|---|---|---|---|---|---|")
    for h in res["hits"]:
        w("| %.1f s | %s | %.1f | %.1f | %.1f | %.1f | %+.1f |" % (h["t"], h["kind"], h["m"], h["rms"], h["peak"], h["sub"], h["sub"] - h["full"]))
    w("")

    w("## 6. Stereo\n")
    s = res["stereo"]
    w("| Measure | Value |")
    w("|---|---|")
    w("| L/R correlation, mean / 5th percentile / minimum | %+.2f / %+.2f / %+.2f (at %.1f s) |" % (s["mean"], s["p5"], s["min"], s["tmin"]))
    w("| 400 ms windows with positive correlation | %.1f %% |" % s["pos"])
    w("| Loudness lost when folded to mono | %.2f LU |" % s["mono_loss"])
    w("| Side level below 100 Hz, relative to mid | %.1f dB (bass is mono) |" % s["side_lo"])
    w("\n![stereo correlation](qa_correlation.png)\n")

    w("## 7. Cue timing\n")
    o = res["onsets"]
    w("Two blind onset detectors run over the whole file with no knowledge of the grid (%d and %d onsets found). For each "
      "cue the nearest detected onset is reported. Detector A fires on the first sample where the energy of a causal band "
      "(> 2 kHz, 0.3-2 kHz or full band) exceeds the preceding 20 ms by 8 dB; detector B is log-magnitude spectral flux on a "
      "5.3 ms STFT (0.5 ms hop).\n" % (o["n_a"], o["n_b"]))
    w("| Cue (s) | Sample | Kind | A: error (ms) | A: strength (dB) | A: band | A: rank of %d | B: error (ms) | B: z-score |" % o["n_a"])
    w("|---|---|---|---|---|---|---|---|---|")
    for r in orows:
        w("| %.1f | %d | %s | %+.2f | %.0f | %s | %d | %+.2f | %.0f |" % (r["cue"], round(r["cue"] * SR), r["kind"], r["err_a"], r["str_a"], r["band"], r["rank"], r["err_b"], r["z_b"]))
    w("\n![cue onsets](qa_onsets.png)\n")

    w("## 8. Harmony and rhythm, measured from the audio\n")
    hm = res["harmony"]
    w("Key estimate (Krumhansl-Kessler profiles against the 7.5-80 s chroma): " + ", ".join("%s r = %.2f" % (n, r) for r, n in hm["keys"]) + ".\n")
    w("Per bar: the three strongest pitch classes between 110 Hz and 2.2 kHz and the share of that energy on the tones of the "
      "intended chord (bars 14 and 30 are Asus4 -> A, so D counts; the previous chord's reverb tail and passing melody notes account for the rest).\n")
    w("| Bar | Time (s) | Intended | Strongest pitch classes | Chord-tone share |")
    w("|---|---|---|---|---|")
    for r in hm["rows"]:
        w("| %d | %.1f | %s | %s | %.0f %% |" % (r["bar"], r["t"], r["chord"], ", ".join("%s %.0f %%" % (n, 100 * v) for n, v in r["top"]), 100 * r["share"]))
    w("\n![harmony](qa_harmony.png)\n")
    w("Envelope modulation (prominence over the 1-12 Hz floor) at the beat / 8th / 16th rates (1.6 / 3.2 / 6.4 Hz):\n")
    w("| Section | Beat | 8ths | 16ths |")
    w("|---|---|---|---|")
    for k_, v in hm["mod"].items():
        w("| %s | %+.1f dB | %+.1f dB | %+.1f dB |" % (k_, v[0], v[1], v[2]))
    w("\nThe lift is written in 3+3+2 accents over running 16ths, which is why its beat-rate line is weak and its 16th-rate line strong.\n")

    w("## 9. Stems\n")
    w("Stems are printed through the master bus (same EQ and the same gain curve as the mix), so they sum to the master: "
      "residual peak **%.1f dBFS**. `braams.wav` carries all brass (braams, horns, low brass); `keys.wav` is the motif piano.\n" % res["stems"]["null"])
    w("| Stem | Duration | Peak (dBFS) | Loudness (LUFS, ungated) | Size |")
    w("|---|---|---|---|---|")
    for r in res["stems"]["rows"]:
        w("| `%s.wav` | %.3f s | %.1f | %.1f | %s |" % (r["name"], r["n"] / SR, r["peak"], r["lufs"], fmt_size(r["size"])))
    w("\n![stems](qa_stems.png)\n")

    w("## 10. The 30 s bed\n")
    b = res["bed"]
    w("| Measure | Value |")
    w("|---|---|")
    w("| Length | %.3f s (%d samples) |" % (b["n"] / SR, b["n"]))
    w("| Integrated loudness, own / ffmpeg | %.2f / %.1f LUFS (deliberately 2 LU under the theme) |" % (b["lufs"], b["ff"]["I"]))
    w("| True peak / sample peak | %.2f dBTP / %.2f dBFS |" % (b["tp"], b["sample_peak_db"]))
    w("| Loudness range | %.1f LU |" % b["lra"])
    w("| Short-term loudness: start -> peak | %.1f -> %.1f LUFS (peak at %.1f s) |" % (b["st_start"], b["st_peak"], b["t_peak"]))
    w("| Max momentary | %.1f LUFS |" % b["max_m"])
    w("| End hit onset vs. 27.500 s | %+.2f ms (strength %.0f dB) |" % (b["hit_err"], b["hit_str"]))
    w("| Last 200 ms | %.1f dBFS RMS |" % b["end_rms"])
    w("| Full-scale samples / DC | %d / %.0f dBFS |" % (b["clipped_samples"], b["dc_db"]))
    w("\n![bed](qa_bed.png)\n")

    if res["lossy"]:
        w("## 11. Lossy encodes\n")
        w("Decoded back to PCM with ffmpeg and compared with the WAV.\n")
        w("| File | Codec | Bit rate | Decoded length | Offset vs. WAV | Loudness | True peak | Size |")
        w("|---|---|---|---|---|---|---|---|")
        for r in res["lossy"]:
            w("| `%s` | %s | %d kb/s | %.3f s | %+d samples | %.2f LUFS | %.2f dBTP | %s |" % (r["name"], r["codec"], r["kbps"], r["n"] / SR, r["offset"], r["lufs"], r["tp"], fmt_size(r["size"])))
        w("")
    (out / "QA.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    ok = main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
    sys.exit(0 if ok else 1)
