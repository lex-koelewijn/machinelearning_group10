# ### Imports

import pandas as pd
import numpy as np
from midiutil import MIDIFile

# ### Data

F = pd.read_csv('data/F.txt', sep='\t', header=None)
F.head()
F.describe()


# ### Helper functions

def convert_channel(channel):
    converted_channel = []
    duration = 1
    last_pitch = 0
    for pitch in channel:
        if pitch is last_pitch:
            duration += 1
        else:
            converted_channel.append((last_pitch, duration))
            duration = 1
        last_pitch = pitch
    return converted_channel


# # MIDI MAGIC

# +
track = 0
time = 0
tempo = 105   #average BPM of bach
volume = 100  #0-127
rythm = 4

MyMIDI = MIDIFile(1)
MyMIDI.addTempo(track, time, tempo)

for channel in F:
    time_passed = 0
    for pitch, duration in convert_channel(F[channel]):
        if pitch is not 0:
            MyMIDI.addNote(track, channel, pitch, (time + time_passed)/rythm, duration, volume)
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
