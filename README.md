# Performance Evaluation of Imitation Learning Algorithms with Human Experts
>This project was done as my Bachelor of Science thesis at KTH with another student.

The project aimed to evaluate the difference in the performance of three different imitation learning algorithms. The environment in which the evaluation took place was TORCS, a car racing simulator. The expert driver was one of us driving with a steering wheel and pedals interfacing with TORCS.

1. Regular imitation learning

   This is the most simple form of imitation learning where a machine learning model trains on existing data. It is very easy to implement but suffers from compounding errors.
2. DAGGER (Dataset Aggregation)

   DAGGER is a bit more complex in the way that it constantly switches the controls from the training model to the human driver. This way, the car is safe from a bit of the compounding errors experienced in the first method.
3. HG-DAGGER (Human-Gated Dataset Aggregation)
   
   The third and last algorithm that we evaluated was HG-DAGGER which uses an ensemble of neural networks to decide if the action choice is unanimous or if the human expert should take the controls. This lets the car run on either the human or neural network for a longer time.

We found that HG-DAGGER was the best option by far when using a human expert. More about the conclusion and further discussions can be found in the [thesis](Performance_Evaluation_of_Imitation_Learning_Algorithms_with_Human_Experts.pdf).

<p align="center">
  <img src="./images/1.gif" />
</p>

## Usage example
First install the requirements.

```sh
pip install -r requirements.txt
```

The neural network architecture can be configured in the agent.py file. To run the project, launch torcs and select a network driver.

```sh
python main.py
```

## Development setup

Python 3 and TORCS are needed to run the project.

We used a steering wheel and pedals to interface with pygame which in turn sent the actions to TORCS. This was needed since the keyboard arrows were not enough to reliably steer a car in TORCS.


## Meta

Erik Båvenstrand – [Portfolio](https://bavenstrand.se) – erik@bavenstrand.se

Distributed under the MIT license. See ``LICENSE`` for more information.

[github.com/ErikBavenstrand](https://github.com/ErikBavenstrand)

Special thanks to Chris X Edwards who developed snakeoil which lets us interface with TORCS through Python.
