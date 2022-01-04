import pretty_midi


def demo_format_convert(data, f, *inputs):
    return [[f(x, *inputs) for x in track] for track in data]


def bpm_to_alpha(bpm):
    return 60 / bpm


def add_notes(track, shift_second):
    notes = []
    ss = 0
    for i, seg in enumerate(track):
        notes += [pretty_midi.Note(n.velocity, n.pitch,
                                   n.start + ss, n.end + ss)
                  for n in seg]
        ss += shift_second
    return notes


def demo_to_midi(data, names, bpm=90., shift_second=None, shift_beat=None):
    alpha = bpm_to_alpha(bpm)
    if shift_second is None:
        shift_second = alpha * shift_beat
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for track, name in zip(data, names):
        ins = pretty_midi.Instrument(0, name=name)
        ins.notes = add_notes(track, shift_second)
        midi.instruments.append(ins)
    return midi


def write_demo(fn, data, names, bpm=90., shift_second=None, shift_beat=None):
    midi = demo_to_midi(data, names, bpm, shift_second, shift_beat)
    midi.write(fn)


