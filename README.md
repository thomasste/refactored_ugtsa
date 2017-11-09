# Setup the environment
```
mkvirtualenv tensorflow -p python3
git clone git@github.com:tensorflow/tensorflow.git
git clone git@github.com:thomasste/refactored_ugtsa.git
cd tensorflow
git checkout r1.3
ln -s ../refactored_ugtsa refactored_ugtsa
```

Append those lines to WORKSPACE:
```
http_archive(
    name = "com_github_nelhage_boost",
    strip_prefix = "rules_boost-master",
    type = "tar.gz",
    urls = [
        "https://github.com/nelhage/rules_boost/archive/master.tar.gz"
    ],
)
load("@com_github_nelhage_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
```

Follow the steps from https://www.tensorflow.org/install/install_sources in order to build and install tensorflow pip package.

# Build ucb_trainer
```
cd tensorflow
bazel build --config=opt //refactored_ugtsa:ucb_trainer
```

# Build model
Create directories refactored_ugtsa/build/models and refactored_ugtsa/build/graphs and then run:
```
cd refactored_ugtsa
workon tensorflow
python build_model.py omringa 10112017_basic 1
python build_model.py omringa 10112017_vertical 1
```

# Improve model with ucb_trainer
```
cd refactored_ugtsa/build
./ucb_trainer omringa__10112017_basic__1 3000 10 0
./ucb_trainer omringa__10112017_vertical__1 3000 10 0
```