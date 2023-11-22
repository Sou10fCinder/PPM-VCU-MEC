# Power and Performance Models for Virtual Computing Units in Mobile Edge Computing (PPM-VCU-MEC)

This repository implements the models introduced in the paper titled "Power and Performance Models for Virtual Computing Units in Mobile Edge Computing." The paper discusses three different types of tasks, including CPU-intensive tasks, GPU-intensive tasks, and IO-intensive tasks. each with power and performance models, resulting in a total of six models. We obtained optimal parameters for each model through a differential evolution algorithm, and these parameters were then used for training the six models.

## Models Overview

The six models aim to comprehensively capture the intricate relationship between power consumption and performance metrics for the specified tasks. The code for the differential evolution optimization process, which determines optimal parameters for each model, can be found in the "DifferentialEvolutionTuning" directory. The optimized parameters are subsequently applied to the training process, implemented in the "ModelTraining" directory.

## Repository Structure

- **DifferentialEvolutionTuning:** Contains the code for the differential evolution optimization process, yielding optimal parameters for each model.

- **ModelTraining:** Includes the code responsible for training the models using the obtained optimal parameters. This directory encapsulates the complete training pipeline.

## Usage

To reproduce results and use the models in specific scenarios, follow the annotations provided in each .py source code file step by step.

Feel free to explore and adapt the models to fit your research or application needs. If you encounter any issues or have questions, refer to the provided documentation or reach out to the contributors listed in the paper.

**Note:** Ensure compliance with the licensing terms outlined in the MIT License for this repository.

## Citation

If you find this repository or the associated models useful in your work, kindly cite our paper:

