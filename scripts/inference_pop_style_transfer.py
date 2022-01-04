import os
import sys
sys.path.insert(0, os.path.realpath(os.path.join(sys.path[0], '..')))
from pop_style_transfer import load_model, acc_audio_to_midi
import argparse
import warnings

warnings.simplefilter('ignore', UserWarning)

# define command line arguments
parser = argparse.ArgumentParser(description='Audio2Midi arrangement')

parser.add_argument('input_acc_audio_path', metavar='input_acc_audio_path',
                    type=str)

parser.add_argument('output_midi_path', metavar='output_midi_folder_path',
                    type=str)

parser.add_argument('-o', '--original', dest='input_audio_path',
                    help='(original) input audio path',
                    default=None)
parser.add_argument('-m', '--model', dest='model_id',
                    help="mode_id in 'a2s', 'a2s-nochd', 'supervised', "
                         "'supervised+kl', 'all'",
                    default='a2s')
parser.add_argument('-a', '--auto', dest='autoregressive',
                    action='store_const',
                    const=True, default=False,
                    help='Turn on autoregressive mode (default off)')

parser.add_argument('--alt', dest='alt', action='store_const', const=True,
                    default=False, help='To use stage-3a alternative pt file.')

args = parser.parse_args()


if __name__ == '__main__':
    if args.model_id not in ['a2s', 'a2s-nochd', 'supervised',
                             'supervised+kl', 'all']:
        raise ValueError("Invalid model_id.")

    # in the command line mode, output_midi_path should be a folder
    output_dir = os.path.splitext(args.output_midi_path)[0]
    os.makedirs(output_dir, exist_ok=True)

    if args.model_id != 'all':  # do audio2midi arrangement once
        model, model0 = load_model(args.model_id, args.autoregressive,
                                   args.alt)
        output_midi_path = \
            os.path.join(output_dir,
                         f'{args.model_id}-{args.autoregressive}.mid')
        acc_audio_to_midi(args.input_acc_audio_path,
                          output_midi_path,
                          model, model0,
                          args.input_audio_path,
                          input_analysis_npy_path=None,
                          save_analysis_npy_path=None)
    else:  # enumerate all possible inputs.
        input_analysis_npy_path = None
        save_analysis_npy_path = 'tmp_analysis.npy'

        for model_id in ['a2s', 'a2s-nochd', 'supervised', 'supervised+kl']:
            for autoregressive in [True, False]:
                if model_id in ['supervised', 'supervised+kl'] \
                        and autoregressive:
                    continue

                for alt in [True, False]:
                    if model_id != 'a2s' and alt:
                        continue

                    print(f"- Case (model_id={model_id}, "
                          f"autoregressive={autoregressive}, alt={alt})")
                    model, model0 = load_model(model_id, autoregressive)
                    output_midi_path = \
                        os.path.join(output_dir,
                                     f'model-{model_id}-auto-{autoregressive}-'
                                     f'alt-{alt}.mid')

                    acc_audio_to_midi(args.input_acc_audio_path,
                                      output_midi_path,
                                      model, model0,
                                      args.input_audio_path,
                                      input_analysis_npy_path,
                                      save_analysis_npy_path)
                    if input_analysis_npy_path is None:
                        input_analysis_npy_path = save_analysis_npy_path
                        save_analysis_npy_path = None
        os.remove(input_analysis_npy_path)
