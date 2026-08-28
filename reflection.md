# Reflection: Building an MLOps Pipeline for PyTorch

## What was the most challenging part?

Not the model. Training a ResNet-18 on CIFAR-10 and wrapping it in a FastAPI
service was the most familiar work in the assignment. The hard part was
discovering that a pipeline can look finished and be comprehensively broken —
and that the components meant to warn me were themselves the ones failing
silently.

My CI workflow had failed on all six runs since I added it, and I had not
noticed, because the failure was quiet in a specific way. Both Docker build jobs
declared `needs: lint-and-test`. With lint red, they never reported `failure`;
they reported `skipped`. I had been reading the absence of a red build as
evidence of a green one. Neither image had ever actually been built by CI.

Lint was red for a reason I would not have guessed. I installed `ruff` unpinned,
so CI resolved a newer release whose default rule set was wider than my local
version's. The same command passed on my machine and failed on the runner with
ten errors. Pinning it exposed a second divergence immediately: `torch==2.2.2` is
compiled against the NumPy 1.x C ABI, and the unpinned `numpy` resolved to 2.x,
so `torchvision`'s `ToTensor` died with `RuntimeError: Numpy is not available`.
That one could never have reproduced locally — my machine had torch 2.11 and
NumPy 2.3, and did not match my own pinned requirements at all.

Underneath both sat a plainer mistake. `configs/training_config.yaml` was
referenced by `train.py`, by `Dockerfile.train`, and by my README, and existed in
none of them. Git does not track empty directories, so `configs/` never left my
laptop. `docker build` had been failing at `COPY configs/` — which the skipped CI
job had hidden.

The Kubernetes lesson was sharper. I had written the GPU bonus directly into the
default training Job: `nvidia.com/gpu: 1`, a node selector, a toleration. No node
on Minikube satisfies any of that, so the pod sat `Pending` indefinitely, no
checkpoint was written, and the serving Deployment had nothing to load. An
optional bonus had quietly disabled the mandatory end-to-end validation. Splitting
it into a separate manifest cost nothing and made both paths work.

What I take away is that reproducibility is not a tidiness concern, it is the
whole discipline. Every failure above was an environment that differed from the
one I believed I had: an unpinned linter, an unpinned array library, a directory
Git declined to carry, a node label that existed only in my assumptions. Training
the network was a small part of this project. Making it run identically somewhere
other than my own machine was the project.

## Use of AI assistance

I used Claude (Anthropic) as a pair programmer for this assignment: to audit the
repository against the specification, to diagnose the CI failures described
above, and to draft the manifests, the lint configuration and the fixes. Its use
is cited in the commit message and pull request description of every change it
contributed to. I have reviewed each change and can explain it.
