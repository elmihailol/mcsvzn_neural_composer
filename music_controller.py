import operator
from heapq import nlargest
import numpy
from keras import Sequential
from keras.layers import LSTM, Dense
from music21 import converter, instrument, note, chord, stream
from music21.ext import joblib
import music21


def create_midi(prediction_output):
	"""
	Генерируем МИДИ-файл из сообщений
	prediction_output: список МИДИ-сообщений
	"""
    offset_plus = 0
    output_notes = []
    offset = 0
    zero_counter = 0
    for pattern in prediction_output:
        s = pattern.split("|")
        pattern = s[0]
        octave = s[2]
        try:
            offset += float(s[1])
            if float(s[1]) == 0:
                zero_counter += 1
            else:
                zero_counter = 0
        except:
            print("error", s[1])
            continue
            
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.PipeOrgan()
                new_note.volume.velocity = 60
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.volume.velocity = 60
            new_note.octave = octave
            new_note.storedInstrument = instrument.PipeOrgan()
            output_notes.append(new_note)


    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')


def get_msg(file):
	"""
	Импортируем МИДИ-сообщения из файла
	file: путь к файлу
	return: список сообщений
	"""
    notes = []
    n = 0
    try:
        midi = converter.parse(file)
        print(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:  
            notes_to_parse = midi.flat.notes
        prev_offset = 0
        for element in notes_to_parse:
            new_offset = 0.5
            if prev_offset == element.offset:
                new_offset = 0
            else:
                new_offset = element.offset - prev_offset
                if 0 < new_offset <= 0.5:
                    new_offset = 0.5
                if 0.5 < new_offset <= 1:
                    new_offset = 1

                if 1 < new_offset <= 1.5:
                    new_offset = 1.5
                if 1.5 < new_offset:
                    new_offset = 2
            if isinstance(element, note.Note):
                notes.append(str(element.pitch) + "|" + str(new_offset) + "|" + str(element.octave))
                n += 1
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder) + "|" + str(new_offset)
                             + "|")
                n += 1

            prev_offset = element.offset
    except Exception as e:
        print("Что - то не так: ", e)
    return notes
