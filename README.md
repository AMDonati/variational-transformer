# variational-transformer

## Description of the research project
The research project is described in this [document](https://drive.google.com/file/d/1Yu4hx16hbtbiSu0j6bC3lSueLCKgq6Lj/view?usp=share_link)


### requirements 
The libraries needed to run the code are provided in the file requirements.txt.
* To run all the scripts from the origin repo, run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`


### Preprocess the ROC Dataset: 

* The csv file for the dataset is available [here](https://drive.google.com/file/d/1CM6HS2dOrOd2XUtt8c3fThycU70FMv4z/view?usp=sharing)
* To process the dataset, first create the data folder:  

`mkdir data`\\ 
`cd data`\\
`mkdir ROC`
* Then run: `python src/data_provider/preprocess_ROC.py`
> This create three pkl files for each dataset split, and a "vocab.json" file. 

### Run the models (Baseline Transformer or VAE Transformer)
* Baseline Transformer: `python src/scripts/run_transformer.py -model "transformer" -num_layers $NUM_LAYERS -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH`
* VAE Transformer: `python -u src/scripts/run_transformer.py -model "VAE" -latent "attention" -num_layers $NUM_LAYERS -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH`
> Examples of script are in "src/scripts/sh".

### Postprocess the results:
* To merge the results of several experiments, run: `python src/scripts/postprocess.py -path $PATH`
> "$PATH" is the path of a folder gathering multiple experiments. 
* For the VAE Transformer, you can log on tensorboard to check the training results: `tensorboard --logdir=$PATH/logs`

