Innovation: PGELU (parametric GELU) is a parametric variant of GELU using beta and alpha parameters. The equation is as follows: PGELU(x)=β(x-α)Φ(β(x-α)). This adds scaling and shifting factors to GELU in a way that appears to be novel relative to other variants.

CIFAR-10 Analytics (41 seeds):
Final loss standard deviation - 27.6% reduction (future trials have shown around 10% reduction)
Loss range - 34.9% reduction
Performance - 0.12% increase (future trials have shown figures like 0.4% and 0.75% increases)

NLP Analytics:
Loss - Notably more stable (standard seemed to increase at the third epoch for some reason while PGELU consistently decreased)
Performance - 2.71% relativistic greater end performance (1.83% absolute), consistently beat across 80% of epochs
