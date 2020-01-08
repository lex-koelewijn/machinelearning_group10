# ### Imports

import pandas as pd
import numpy as np
from midiutil import MIDIFile

# ### Data

df_input = pd.read_csv('data/F.txt', sep='\t', header=None)
df_input.head()


# ### Helper functions

# This function turns ```0 0 0 0 1 1``` into ```(0, 4), (1, 2)```. This is required for proper playback.

def convert(channel):
    converted = []
    duration = 0
    last_pitch = 0
    for pitch in channel:
        if pitch is last_pitch:
            duration += 1
        else:
            converted.append((last_pitch, duration))
            duration = 1
        last_pitch = pitch
    return converted


# # MIDI MAGIC

# +
# MIDIUtil parameters
track = 0
time = 0
tempo = 105   #average BPM of bach
volume = 100  #0-127
# Custom parameters
rythm = 4
max_duration = 4

MyMIDI = MIDIFile(1)
MyMIDI.addTempo(track, time, tempo)

for channel in df_input:
    time_passed = 0
    for pitch, duration in convert(df_input[channel]):
        if pitch is not 0:
            MyMIDI.addNote(track, channel, pitch, (time + time_passed)/rythm, min(duration, max_duration), volume)
        time_passed += duration
# -

# ## Convert to .wav

# You need fluidsynth installed for the conversion: ```sudo apt install fluidsynth```

# +
with open("output.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

from midi2audio import FluidSynth
fs = FluidSynth('data/soundfont.sf2', sample_rate=100)
fs.midi_to_audio('output.mid', 'output.wav')
