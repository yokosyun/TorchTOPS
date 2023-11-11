# Development

## install
```
mkdir ~/venv
python3.8 -m venv ~/venv/torchtops
source ~/venv/torchtops/bin/activate
pip3 install pip --upgrade
pip3 install -r requirements.txt
pre-commit install
```

## set environment path
set PYTHONPATH as root of this repository
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## test
```
python3.8 tools/example.py
```

## Reference
[torchprof](https://github.com/awwong1/torchprof)
[fvcore](https://github.com/facebookresearch/fvcore)
