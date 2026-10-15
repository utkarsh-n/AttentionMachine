#!/usr/bin/env python3
"""Verify the production Swift tone encoder without requiring audio hardware.

The helper compiles SensoryTone directly from App/SensoryEngine.swift, exports
every note, and checks its actual WAV data with independent PCM measurements.
It does not establish speaker quality, latency, haptic feel, or route behavior.
"""
from pathlib import Path
import array
import json
import math
import subprocess
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "App" / "SensoryEngine.swift"
tone_source = SOURCE.read_text().split("enum SensoryTone {", 1)[1]

with tempfile.TemporaryDirectory(prefix="attention-sensory-") as temporary:
    temporary = Path(temporary)
    swift = temporary / "CheckTones.swift"
    swift.write_text("import Foundation\nenum SensoryTone {" + tone_source + r'''
let directory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
for note in 0..<24 {
    try SensoryTone.wav(pitch: note).write(to: directory.appendingPathComponent("note-\(note).wav"))
}
precondition(SensoryTone.wav(pitch: -99) == SensoryTone.wav(pitch: 0))
precondition(SensoryTone.wav(pitch: 999) == SensoryTone.wav(pitch: 23))
precondition(SensoryTone.clampedUnit(-1) == 0)
precondition(SensoryTone.clampedUnit(2) == 1)
precondition(SensoryTone.clampedUnit(.nan) == 0)
precondition(SensoryTone.clampedUnit(.infinity) == 0)
let weights = [0.0, 0.05, 0.5, 0.85, 1.0]
let haptics = weights.map { SensoryTone.hapticIntensity(weight: $0) }
let volumes = weights.map { SensoryTone.playbackVolume(weight: $0, gain: 1) }
precondition(haptics.first == 0 && haptics.last == 1)
precondition(volumes.first == 0 && volumes.last! < 1)
precondition(zip(haptics, haptics.dropFirst()).allSatisfy { $0 < $1 })
precondition(zip(volumes, volumes.dropFirst()).allSatisfy { $0 < $1 })
precondition(SensoryTone.playbackVolume(weight: 1, gain: 0) == 0)
precondition(SensoryTone.playbackVolume(weight: .nan, gain: 1) == 0)
precondition(SensoryTone.playbackVolume(weight: 1, gain: .nan) == 0)
precondition(SensoryTone.hapticIntensity(weight: 999) == 1)
precondition(SensoryTone.hapticIntensity(weight: -1) == 0)
precondition(SensoryTone.playbackVolume(weight: 1, gain: 999) <= 0.8)
''')
    executable = temporary / "CheckTones"
    subprocess.run(["xcrun", "swiftc", "-module-cache-path", str(temporary / "module-cache"),
                    str(swift), "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(temporary)], check=True)
    worst_frequency_error = 0.0
    peak = 0
    maximum_boundary_delta = 0
    maximum_mean = 0.0
    for note in range(24):
        path = temporary / f"note-{note}.wav"
        with wave.open(str(path), "rb") as wav:
            assert wav.getnchannels() == 1, f"Note {note}: must be mono"
            assert wav.getsampwidth() == 2, f"Note {note}: must be 16-bit PCM"
            assert wav.getframerate() == 22050, f"Note {note}: unexpected sample rate"
            samples = array.array("h", wav.readframes(wav.getnframes()))
            assert 0.109 <= len(samples) / wav.getframerate() <= 0.111
        assert samples[0] == samples[-1] == 0, f"Note {note}: nonzero waveform boundary"
        boundary_delta = max(abs(samples[1] - samples[0]), abs(samples[-1] - samples[-2]))
        assert boundary_delta <= 2, f"Note {note}: abrupt waveform boundary"
        maximum_boundary_delta = max(maximum_boundary_delta, boundary_delta)
        note_peak = max(map(abs, samples))
        assert 1000 < note_peak <= 9000, f"Note {note}: unexpected PCM peak"
        peak = max(peak, note_peak)
        mean = abs(sum(samples) / len(samples)) / 32768
        assert mean < 0.001, f"Note {note}: unexpected DC offset"
        maximum_mean = max(maximum_mean, mean)
        crossings = []
        for index in range(int(0.01 * 22050), int(0.09 * 22050)):
            a, b = samples[index], samples[index + 1]
            if a <= 0 < b:
                crossings.append(index + (-a / (b - a)))
        assert len(crossings) >= 5
        measured_frequency = 22050 * (len(crossings) - 1) / (crossings[-1] - crossings[0])
        expected_frequency = 220 * 2 ** (note / 12)
        relative_error = abs(measured_frequency - expected_frequency) / expected_frequency
        assert relative_error < 0.001, f"Note {note}: frequency error {relative_error:.2%}"
        worst_frequency_error = max(worst_frequency_error, relative_error)
    print(json.dumps({
        "notes_checked": 24,
        "format": "mono 16-bit PCM, 22050 Hz",
        "duration_seconds": len(samples) / 22050,
        "maximum_pcm_peak": peak,
        "maximum_boundary_sample_jump": maximum_boundary_delta,
        "maximum_normalized_dc_offset": maximum_mean,
        "maximum_relative_frequency_error": worst_frequency_error,
        "pitch_and_strength_bounds": "passed",
        "weak_to_strong_mapping_and_zero_gain": "passed",
        "physical_audio_and_haptics": "not assessed",
    }, indent=2))
