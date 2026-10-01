# Validation status

The English-only release is under real-model before/after validation. It must
not be described as fully reproduced until the corresponding completion report
records successful training, export and paired evaluation.

The reference comparison freezes the pre-edit source and uses the same original
checkpoint, ordered basis, covariance, routed rank inputs, prompts, seeds,
device and optimization settings for both versions. It retrains all 48 layers
of the 10% profile and includes Hot compensation. This does not constitute a
fresh reproduction of every budget or every paper table.

Historical scores in results/summary.json are not results from this validation
run. CPU tests and language/AST checks alone are not sufficient evidence of
end-to-end reproducibility. The original model and complete datasets are not
included in this source package.
