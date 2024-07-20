# `attain`

A little Markov Chain library, for funsies.


## Usage

```python
>>> import attain
>>> mc = attain.MarkovChain()

# In the Real World™, you'd probably want to create the corpus
# sequence-of-tokens a better way. But for example purposes...
>>> sentences = """
...     Hello, world!
...     Hello Kitty is my favorite.
...     My life for Aiur!
... """
>>> corpus = [
...     word.lower().rstrip(",!.").strip()
...     for word
...     in sentences.split()
... ]

# At current, training can only be done once.
>>> mc.train(corpus)

>>> mc.generate()
['is', 'my', 'life', 'for', 'aiur']

>>> mc.generate_sentence()
"For aiur kitty is my favorite!"
```


## Tests

```shell
$ git clone https://github.com/toastdriven/attain.git
$ cd attain

$ pipenv install --dev

$ pipenv shell

$ pytest tests
```


## License

New BSD
