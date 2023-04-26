import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.spatial.distance import cdist
from pyswarm import pso
from pydub import AudioSegment

SAMPLE_RATE = 44100
FREQUENCY = 440
DURATION = 5  # in seconds
AMPLITUDE = 2 ** 15 - 1
AUDIO_TYPE = "S16LE"  # 16-bit signed little-endian PCM

class Operator:
    def __init__(self, freq_ratio, level, attack, decay, sustain, release):
        self.freq_ratio = freq_ratio
        self.level = level
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

    def envelope(self, duration, sample_rate=SAMPLE_RATE):
        t = np.linspace(0, duration, duration * sample_rate, False)
        env = np.piecewise(t, [t < self.attack, (t >= self.attack) & (t < self.attack + self.decay), t >= self.attack + self.decay],
            [lambda t: (t / self.attack) * self.level, lambda t: self.level - (t - self.attack) * (self.level - self.sustain) / self.decay, self.sustain])
        return env

    def waveform(self, base_freq, duration, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE):
        freq = base_freq * self.freq_ratio
        t = np.linspace(0, duration, duration * sample_rate, False)
        sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
        return sine_wave.astype(np.int16)

class Algorithm:
    def __init__(self, operator_list, algorithm_number):
        self.operator_list = operator_list
        self.algorithm_number = algorithm_number

    def dx7_algorithm(self, base_freq, duration):
        op_waveforms = [op.waveform(base_freq, duration) * op.envelope(duration) for op in self.operator_list]
        modulated_wave = np.zeros_like(op_waveforms[0])

        def chain(ops):
            chained = op_waveforms[ops[0]]
            for op in ops[1:]:
                chained *= op_waveforms[op]
            return chained

        def add_to_wave(ops):
            nonlocal modulated_wave
            if type(ops[0]) == list:
                for op in ops:
                    modulated_wave += chain(op)
            else:
                modulated_wave += chain(ops)

        algorithms = [
            [[5], [0, 4], [1, 4], [2, 4], [3, 4]],
            [[5], [4], [0, 1, 2, 3]],
            [[5], [4], [0, 1], [2, 3]],
            [[5], [4], [0], [1], [2], [3]],
            [[5], [0, 1, 2, 3, 4]],
            [[5], [0, 1, 2, 4], [3, 4]],
            [[5], [0, 1, 4], [2, 3]],
            [[5], [0, 1], [2, 4], [3, 4]],
            [[5], [0], [1, 4], [2, 4], [3, 4]],
            [[5], [0], [1, 2, 3, 4]],
            [[5], [0], [1, 2], [3, 4]],
            [[5], [0], [1], [2, 3], [4]],
            [[5], [0], [1], [2], [3], [4]],
            [[4, 5], [0, 1, 2, 3]],
            [[4, 5], [0, 1], [2, 3]],
            [[4, 5], [0], [1], [2], [3]],
            [[3, 4, 5], [0, 1, 2]],
            [[3, 4, 5], [0], [1, 2]],
            [[3, 4, 5], [0], [1], [2]],
            [[2, 3, 4, 5], [0, 1]],
            [[2, 3, 4, 5], [0], [1]],
            [[1, 2, 3, 4, 5], [0]],
            [[5], [0, 1], [2, 3]],
            [[5], [0], [1], [2, 3]],
            [[5], [0], [1], [2], [3]],
            [[4, 5], [0, 1, 2]],
            [[4, 5], [0, 1], [2]],
            [[4, 5], [0], [1], [2]],
            [[3, 4, 5], [0, 1]],
            [[3, 4, 5], [0], [1]],
            [[2, 3, 4, 5], [0]],
            [[1, 2, 3, 4, 5]],
        ]

        add_to_wave(algorithms[self.algorithm_number - 1])

        return modulated_wave.astype(np.int16)
    
def main():
    # Load the given .wav file
    given_wav_file = "./528491.wav"
    sample_rate, given_audio = wavfile.read(given_wav_file)

    # Convert the given stereo audio to mono
    if len(given_audio.shape) > 1 and given_audio.shape[1] > 1:
        given_audio = given_audio.mean(axis=1)

    def error_function(params):
        print("1")
        # Define Operators with the given parameters
        operators = [Operator(freq_ratio=params[i], level=params[i + 1], attack=params[i + 2], decay=params[i + 3], sustain=params[i + 4], release=params[i + 5]) for i in range(0, 36, 6)]
        algorithm_number = int(params[0])
        # Define Algorithm with a list of Operators and a fixed algorithm number
        algorithm = Algorithm(operators, algorithm_number=algorithm_number)  # Change the algorithm_number to a valid value

        # Synthesize the sound using the defined Algorithm
        modulated_wave = algorithm.dx7_algorithm(FREQUENCY, DURATION)

        target_length = len(given_audio)
        if len(modulated_wave) > target_length:
            modulated_wave = modulated_wave[:target_length]
        elif len(modulated_wave) < target_length:
            padding = np.zeros(target_length - len(modulated_wave))
            modulated_wave = np.concatenate((modulated_wave, padding))
        # Calculate the mean squared error between the synthesized and given audio
        mse = ((given_audio.astype(np.float) - modulated_wave.astype(np.float)) ** 2).mean()

        return mse

    # Define parameter bounds for PSO
    # Define parameter bounds for PSO
    # Define parameter bounds for PSO
    lb = [0] + [0.1, 0.1, 0, 0, 0, 0] * 6  # Lower bounds (0 for algorithm number)
    ub = [31] + [10, 1, 1, 1, 1, 1] * 6  # Upper bounds (31 for algorithm number)

    # Apply Particle Swarm Optimization
    optimized_params, _ = pso(error_function, lb, ub, maxiter=50000, swarmsize=300, debug=True)

   # Define Operators with optimized parameters
    operators = [Operator(freq_ratio=optimized_params[i], level=optimized_params[i + 1], attack=optimized_params[i + 2], decay=optimized_params[i + 3], sustain=optimized_params[i + 4], release=optimized_params[i + 5]) for i in range(1, 37, 6)]

    # Extract the optimized algorithm number and ensure it's an integer
    algorithm_number = int(optimized_params[0])

    # Define Algorithm with a list of Operators and the optimized algorithm number
    algorithm = Algorithm(operators, algorithm_number=algorithm_number)

    # Synthesize the sound using the optimized Algorithm
    modulated_wave = algorithm.dx7_algorithm(FREQUENCY, DURATION)

    # Export the optimized synthesized sound to a WAV file
    audio_segment = AudioSegment(modulated_wave.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
    audio_segment.export("optimized_output.wav", format="wav")

if __name__ == "__main__":
    main()