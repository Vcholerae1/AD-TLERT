# Flat time-varying resistivity model

`0_flat_model.ipynb` constructs a flat 2.5D synthetic model without running
forward modelling or inversion. The background is 100 ohm-m and a fixed
rectangular body changes from 100 to 400 ohm-m along one continuous temporal
sequence: a slow change first, followed by a nearby double event after the
model returns to the 100 ohm-m background.

Generated arrays, metadata, and validation figures are written to
`data/test_model/0_flat_model/`.
