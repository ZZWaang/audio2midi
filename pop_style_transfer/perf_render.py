import numpy as np
import pretty_midi as pm


def intensity_feat(audio, sr):
    dur = int(len(audio) / sr) + 1

    intensity = np.log10(1 + audio ** 2)
    intensity /= intensity.max()

    avg_int = np.zeros(dur * 2 + 1)
    for t in range(len(avg_int)):
        frame = int(t * sr / 2)
        avg_int[t] = intensity[max(frame - 3 * sr, 0):
                               min(frame + 3 * sr, len(intensity))].sum()

    avg_int /= avg_int.max()

    return avg_int


def chord_change_times(analysis):
    times = [0.]
    prev_chord = analysis[0, 3:]
    for i in range(1, len(analysis)):
        chord = analysis[i, 3:]
        if (chord != prev_chord).any():
            times.append(analysis[i, 0])
        prev_chord = chord
    return times


def int_to_dvel(x, min=-6, max=3):
    return min + (max - min) * x


def write_notes(notes, change_times, intensity, output_fn):
    ins = pm.Instrument(0)
    rendered_notes = []
    for n in notes:
        t = int(max(min(np.round(n.start * 2), len(intensity) - 1), 0))
        d_vel = int_to_dvel(intensity[t])
        rendered_notes.append(pm.Note(int(n.velocity + d_vel),
                                      int(n.pitch), n.start, n.end))

    ins.notes = rendered_notes
    midi = pm.PrettyMIDI(initial_tempo=60)
    # midi.instruments.append(ins)

    control_changes = []
    for ct in change_times:
        control_changes.append(pm.ControlChange(64, 0, ct))
        control_changes.append(pm.ControlChange(64, 80, ct + 0.01))
    ins.control_changes = control_changes
    midi.instruments.append(ins)
    midi.write(output_fn)


def _update_truncate_and_jump(analysis, db_ids, i, db_id, jump):
    """
    Decide 1) Whether to jump/skip the next downbeat, and 2) truncate
    the prediction if the measure is incomplete.
    """

    # Case: last measure.E.g., db_id -> [1, 2, 3]
    if i == len(db_ids) - 1:
        truncate = analysis[db_id, 2]  # #beats in the measure (<= 4)

    # Case: current bar is incomplete E.g., db_id -> [1, 2, 3, 1, 2, 3, 4]
    elif analysis[db_id, 2] < 4:
        truncate = analysis[db_id, 2]
        jump = False

    # Case: regular E.g., db_id -> [1, 2, 3, 4, 1, 2, 3]
    else:
        truncate = analysis[db_id + 4, 2] + 4
        jump = True
    return truncate, jump


def _velocity_assignment(onset, pitch):
    """A naive algorithm to render performance"""

    # low and high notes are louder
    vel = 80 if 48 <= pitch <= 69 else 85

    # Notes having more metrical importance are louder
    is_on2 = onset % 2 == 0  # on half note
    is_on1 = onset % 1 == 0  # on quarter note

    if is_on2:
        vel += 3
    elif not is_on1:
        vel -= 3
    return vel



def pred_to_notes(model, analysis, predictions):
    notes = [model.pianotree_dic.grid_to_pr_and_notes(pred, bpm=60)[1]
             for pred in predictions]


def prediction_to_notes(to_notes_func, analysis, predictions, autoregressive):
    """
    Convert model predictions to a continuous piece:
    - Translate the onset to the corresponding downbeat
    - To the original tempo
    """

    # find all downbeat row indices
    db_ids = np.where(analysis[:, 1])[0]

    # a list to store all the output pretty_midi notes.
    notes = []

    jump = False  # indicator to determine whether to skip the current downbeat

    # safety
    for _ in range(10):
        analysis = np.concatenate(
            [analysis, np.zeros((1, analysis.shape[-1]))], 0)
        analysis[-1, 0] = analysis[-2, 0] + analysis[-2, 0] - analysis[-3, 0]

    for i, db_id in enumerate(db_ids):

        # if we skip a downbeat, the next downbeat must not be skipped.
        if jump:
            jump = False
            continue

        # update jump for the next iteration and compute truncate length.
        if autoregressive:
            truncate = analysis[db_id, 2]  # jump is always False
        else:
            truncate, jump = \
                _update_truncate_and_jump(analysis, db_ids, i, db_id, jump)

        # Convert prediction to notes
        pred_notes = to_notes_func(predictions[i])


        start_beat = db_id

        for n in pred_notes:

            # truncation
            if n.start >= truncate:
                continue

            # compute the exact start and end time in the original tempo
            s_id = int(start_beat + n.start)
            s_beat_sec = analysis[s_id, 0]
            s_beat_dur = analysis[s_id + 1, 0] - analysis[s_id, 0]
            s_sub_div_ratio = n.start - int(n.start)
            s_sec = s_beat_sec + s_sub_div_ratio * s_beat_dur

            e_id = int(start_beat + n.end)
            e_beat_sec = analysis[e_id, 0]
            e_beat_dur = analysis[e_id + 1, 0] - analysis[e_id, 0]
            e_sub_div_ratio = n.end - int(n.end)
            e_sec = e_beat_sec + e_sub_div_ratio * e_beat_dur

            # velocity rendering
            vel = _velocity_assignment(n.start, n.pitch)

            notes.append(pm.Note(vel, n.pitch, s_sec, e_sec))
    return notes


def write_prediction(output_fn, to_notes_func, analysis,
                     predictions, audio, sr, autoregressive):
    # convert predicted pianotree to pretty_midi notes. render note velocities.
    notes = prediction_to_notes(to_notes_func, analysis,
                                predictions, autoregressive)

    # retrieve two features for simple rule-based performance rendering
    change_times = chord_change_times(analysis)
    intensity = intensity_feat(audio, sr)

    # output
    write_notes(notes, change_times, intensity, output_fn)
