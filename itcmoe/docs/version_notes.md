# Version notes

Documentation, comments, filenames, and runtime messages were normalized
for this anonymous release.
The method name remains itcmoe; existing checkpoint fields remain compatible.

Normalization is checked against the Python AST: executable structure, numeric
constants, function names, tensor operations and English protocol keys must be
unchanged. Nonfunctional string constants and comments may differ. Real-model
training and evaluation provide a separate numerical before/after check.

Historical source notices and licenses remain intact. Private run logs, source
snapshots and identity-bearing result archives are not part of the release.
