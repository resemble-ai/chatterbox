## This is a light version of chatterbox that removes:
* ```librosa``` and all it's required dependencies and instead uses ```torchaudio``` and ```scipy```
* ```perth``` because who the heck needs watermarking anyways
* ```omegaconf``` and instead use standard python
* others I forget, but everything works

However, I did add ```soundfile``` because I like it.

# Installation
>Go through these steps in order.
```
python -m venv .
```

```
.\Scripts\activate
```

```
python.exe -m pip install --upgrade pip
```

```
pip install uv
```

Next, make sure appropriate versons of torch, torchaudio, and CUDA are installed.

```
uv pip install -r requirements.txt
```

```
pip install git+https://github.com/BBC-Esq/chatterbox-light.git --no-deps
```

Enjoy!
