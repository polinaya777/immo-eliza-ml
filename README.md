# Real Estate Machine Learning Model

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Support](#support)
5. [Contributing](#Contributing)
6. [License](#license)

## Overview

This Python script is designed to predict real estate price in Belgium. After the scraping, cleaning and analyzing, we have a dataset with about 76 000 properties, roughly equally spread across houses and apartments. We are going to preprocess the data and finally build a performant machine learning model!


## Installation

1. Clone the repository to your local machine: 
    ```
    git clone https://github.com/polinaya777/immo-eliza-ml/
    ```
2. Ensure you have Python installed (version 3.10 or higher).

3. Install the required libraries by running:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Open a terminal or command prompt.

2. Navigate to the directory where the script is located.

3. Run the script to train the model:
    ```
    python train.py
    ```
4. Place your data to predict to the data directory:
    ```
    "data/FILENAME"
    ```
5. Run the script to generate predictions:
    ```
    python predict.py -i "data/FILENAME" -o "output/predictions.csv"
    ```
6. The script will generate predictions and save it into a CSV file named `predictions.csv` in the output directory.


## Support
If you encounter any issues or have any questions, please feel free to open an issue in this repository.


## Contributing

Contributions are welcome!


## License

The Real Estate Data Scraper project is licensed under the [MIT License](./LICENSE).
