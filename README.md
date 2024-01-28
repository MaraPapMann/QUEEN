# QUEEN (QUEry UnlEarNing)

You can run the following commands to create a new environments for running the codes with Anaconda:
```shell
conda env create -f environment.yml
conda activate queen
```

For Caltech256, CUB200, TinyImageNedt200, Indoor67 and ImageNet1k, you can also use the scripts ```dataset.sh``` to download and unzip in shell:
```shell
sh dataset.sh
```

Please use the following command to run the experiment.
```shell
python scripts/run.py
```