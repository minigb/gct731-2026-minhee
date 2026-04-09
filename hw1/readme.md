# Homework #1: Piano Music Transcription 

Automatic music transcription (AMT) refers to an automated process that converts musical signals into music scores. Piano Music Transcription  is a specific AMT task for piano music. In this homework, you will implement a small version of "Onsets and Frames", a state-of-the-art piano transcription model with the following goals

* Experiencing the pipeline of training and evaluate a deep learning model for piano music transcription (audio-to-MIDI)
* Building a convolutional recurrent neural network to predict a sequential output (piano-roll style) 
* Learning to handle MIDI files
* Improving the existing model with new ideas 

Before starting this homework, please read the [paper](https://arxiv.org/abs/1710.11153) first!

## Dataset
We use [a subset of the MAESTRO dataset](https://drive.google.com/file/d/1EQ6fFJRhAEugkkCwG2YvmXJyL7Q3Xhes/view?usp=sharing) that contains 170 performance pieces played by junior pianists. The audio files and their corresponding midi files are paired for each piece. We will convert the midi files into piano rolls and train our network to predict them from the audio in a supervised way. We randomly selected 100 / 20 / 50 (train / valid / test) performances from the original dataset for this homework.

## Baseline: a Simplified Onsets and Frames Model 
We provide a Python notebook file [GCT731-HW1.ipynb](https://colab.research.google.com/drive/1ljIU5vk8ZaFzUNmD5Y7W4Yt2jM0qqpCf?usp=sharing
) which includes all components for data preparation, building, training, and evaluting a baseline model. The baseline model is a simplifed onsets and frames model where two independent CNN stacks are used for onset and frame predictions, respectively. You can train and evalute the model by simply executing the cells one by one in the Python notebook file. 

## Script-Based Workflow (Hydra + multirun)
Training now uses [Hydra](https://hydra.cc/) for config composition and sweeps.

Install dependencies first:

```
cd hw1
pip install hydra-core
```

Run one experiment with default config:

```
python3 train.py
```

Override config values from CLI:

```
python3 train.py \
  model=onsets_and_frames \
  data.path=gct731-maestro \
  optimization.learning_rate=5e-4 \
  model.cnn_unit=32 \
  model.fc_unit=128 \
  model.rnn_unit=128 \
  optimization.epochs=10 \
  experiment.name=exp_lr5e4_c32_f128_r128
```

Use the new offset-aware model variant:

```
python3 train.py \
  model=offset_conditioned \
  optimization.offset_loss_weight=1.0 \
  experiment.name=exp_offset_conditioned
```

Run multiple experiments consecutively (Hydra multirun):

```
python3 train.py -m \
  model=basic,onsets_and_frames \
  optimization.learning_rate=1e-3,5e-4 \
  model.cnn_unit=24,32
```

Outputs are saved under `hw1/experiments/`:
- single run: `experiments/<experiment_name>/`
- multirun sweep: `experiments/multirun/<timestamp>/<job_id>_<experiment_name>/`

Saved artifacts per run:
- `config.json`: compact resolved training config
- `config_resolved.json`: full resolved Hydra config
- `history.json`: per-epoch metrics
- `best_model.pt`: best checkpoint on validation loss
- `.hydra/`: Hydra metadata (`config.yaml`, `hydra.yaml`, `overrides.yaml`)

### Optional: Weights & Biases (W&B)
Enable W&B by switching the config group:

```
python3 train.py wandb=enabled wandb.project=gct731-hw1 wandb.run_name=baseline_run
```

W&B with multirun:

```
python3 train.py -m \
  wandb=enabled \
  optimization.learning_rate=1e-3,5e-4 \
  model=basic,onsets_and_frames
```

### Evaluate a Trained Checkpoint
You can load and evaluate a saved checkpoint:

```
python3 evaluate_checkpoint.py \
  --checkpoint experiments/exp_lr5e4_c32_f128_r128/best_model.pt \
  --split test
```

This saves metrics to `eval_<split>.json` next to the checkpoint by default.


## Question 1: Review the Baseline Model (5pts)
Your first task is to understand the baseline model. Unlike music classification tasks, the output of music transcription models is a temporal sequence such as frame-level MIDI notes (i.e., piano rolls). This requires a careful model design in building the neural network. When a mini-batch of mel spectrograms are used as input, the model takes the input data as B (batch) X C (channel) X T (time) x F (frequency) where C is 1 for the mel spectrogram, T corresponds to time frames, and F corresponds to frequency bins in mel. The dimensions of B, C, T, F change as the mel spectrograms are processed with convolution and pooling layers. In addition, the 4-dimensional tensors are partially transposed or flattened in order to be processed with fully connected layers. This is found at this part of the code. 
```
class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size, 1 channel, frames, input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5))

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
```
Write a detailed description of the baseline model including your answers to the following questions (but not limited to them):
- Why the max-pooling is set to (1,2)? 
- What are the dimensions of the cnn output (x = self.cnn(x)) 
- Why x = mel.unsqueeze(1) ?
- Why  x = x.transpose(1, 2).flatten(-2) ? 


## Question 2: Run the Baseline Model (5pts)
You can train and evaluate the model without any modification. When you run the code, you will see this log which displays frame and onset F1 accuracy on the validation set: 
```
Loading 1 group(s) of MAESTRO_small at gct634-maestro
Loading group train: 100%|██████████| 100/100 [00:58<00:00,  1.70it/s]
Loading 1 group(s) of MAESTRO_small at gct634-maestro
Loading group validation: 100%|██████████| 20/20 [00:11<00:00,  1.69it/s]
[Epoch 1/10]
100%|██████████| 1000/1000 [03:33<00:00,  4.68it/s, loss: 1.569e-01]
metric/frame/frame_f1       : 0.6368
metric/frame/onset_f1       : 0.6281
metric/note/f1              : 0.7481
metric/note-with-offsets/f1 : 0.2794...
```

In order to avoid spending too long time for training the model, we will fix the number of epoch to 10 (If you have a powerful GPU machine, you can increase the epoch. But, this is optional). Given this constraint, try to find the best validation accuracy by changing the learning rate, the size of cnn unit and fc unit, or other hyperparameters. If the note F1 is between 0.8 and 0.85, you are on the right track. 

## Question 3: Implement the Onsets and Frames Model (10pts)
Now, you will implement the [original onsets and frames model](https://arxiv.org/abs/1710.11153), where the onset stack has an LSTM module and the frame stack is combined with the output of the onset stack and then processed with another LSTM for the frame prediction.

Below is a diagram that illustrates the onsets and frames model.
```
+-------------------+
| Frame Predictions |
+-------------------+
          ▲
+---------+---------+        
|      Sigmoid      |        
+-------------------+        
          ▲                           
+---------+---------+       +---------+---------+
|        FC         |       | Onset Predictions |
+-------------------+       +-------------------+
          ▲             +------------ ▲
+---------+---------+   |   +---------+---------+
|      BiLSTM       |   |   |      Sigmoid      |
+-------------------+   |   +-------------------+
          ▲  ▲----------+             ▲
+---------+---------+       +---------+---------+
|      Sigmoid      |       |        FC         |
+-------------------+       +-------------------+
          ▲                           ▲
+---------+---------+       +---------+---------+
|        FC         |       |      BiLSTM       |
+-------------------+       +-------------------+
          ▲                           ▲
+---------+---------+       +---------+---------+
|    Conv Stack     |       |    Conv Stack     |
+-------------------+       +-------------------+
               ▲                ▲
           +---+----------------+----+
           |   Log Mel-Spectrogram   |
           +-------------------------+
                        ▲
           +------------+------------+
           |          Audio          |
           +-------------------------+
```

You can implement the model by filling in the following function. 
```
class OnsetsAndFrames(nn.Module):
    def __init__(self, cnn_unit, fc_unit, rnn_unit):
        super().__init__()
        # Your part 

    def forward(self, audio):
        # Your part 

        return frame_pred, onset_pred
```

Note that the bi-directional LSTM (BiLSTM) has a doubled output because it has two LSTMs. Compare this the accuracy on the validation set 


## Question 4: Improve the Piano Transcription Model (5pts)
Your last mission is to improve the onsets and frames model with your own ideas. Some possible ideas are as below but not limited to them. 

- Data Augmentation: key transpose (pitch-shifting in semitone units), random masking (blank masking or Gaussian masking)
- Input representation: Constant-Q transform ([nnAudio](https://github.com/KinWaiCheuk/nnAudio) is a recommneded library for CQT) or other time-frequency representations 
- Neural Network: [HPPNet](https://github.com/WX-Wei/HPPNet) (harmonic dilated conv layers and frequency grouped RNN layer) 
- Loss: regression loss in the [high-resolution piano transcription model](https://arxiv.org/abs/2010.01815), focal loss or other losses for class imbalance in the onset network  
- Teacher-Student model using the noisy student approach 


## Deliverables
You should submit your Python code (`.ipynb` or `.py` files) and homework report (`.pdf` file) to KLMS. The report should include:

* Model review
* Experiments and Results
* Discussion


## Evaluation Criteria
* Did you answer the question 1 precisely?
* Did you construct the onsets and frames model correctly and compare it to the baseline?
* Did you improve the onsets and frames models with your ideas?
* Did you write findings and discussions?
* English does not need to be flawless but the text should be understandable and the code should be re-implementable.
