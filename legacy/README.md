# How to reproduce Aschwanden & Brinkerhoff (2022)

How can I check the published emulators and the error statistics given?

## Evaluate Emulators from arcticdata.io

First, download the emulators and training data from arcticdata.io:

    $ ./download_data.sh
   
Next, evaluate the emulators using the Legacy emalator and dataset classes.
If you diff the NNEmualtor and LegacyNNEmulator you will see that LegacyNNEmulator has

    self.norm_4 = nn.LayerNorm(n_hidden_3)

which looks like a bug. Fortunately, the training used *n_hidden_{1,2,3,4}=128* which means this typo has/had no consequences.
The NNEmulator has this typo fixed.

To get the error statistics, run

    $ python evaluate_emulator_legacy.py --num_models 50 --emulator_dir .  --data_dir speeds_v2/ --target_file observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc --samples_file ../data/samples/velocity_calibration_samples_100.csv
    
Yes, this script is incredibly slow. New version have been improved for speed considerably. 

    $ python evaluate_emulator.py --num_models 50 --emulator_dir .  --data_dir speeds_v2/ --target_file observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc --samples_file ../data/samples/velocity_calibration_samples_100.csv

You should get:
```
Final Score:yr, MBE=-28.61 m/yr, RMSE=355 m/yr, Pearson r=0.9986, r²=0.9944     
=======================================================
MAE=39.19m/yr, MBE=-10.92 m/yr, RMSE=4070 m/yr, Pearson r=1.00, r2=0.99
```

For unknown reasons, the evaluation done for the manuscript did not include training dataset 17. If you delete this file, you'll get the values from the manuscript:

```
Final Score:
=======================================================
MAE=42.06m/yr, MBE=-12.12 m/yr, RMSE=4224 m/yr, Pearson r=1.00, r2=0.99
```

