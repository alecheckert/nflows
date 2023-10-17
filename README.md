# nflows
Toy implementation of normalizing/generative flows in `numpy`.
A playground for some ideas in different normalizing flow layers.

## TODO
 - Guaranteed-invertible version of PlanarFlow (see end of Rezende & Mohamed 2015)
 - Finish RadialFlow class
 - Test for correct serialization of all available Flows in a single model
 - AffineFlow: make so it only operates on the last dimension
 - In general reconsider the `Flow.shape` property
 - keras-like syntax for defining `Model`s?
 - time reversal for PastConv1D
