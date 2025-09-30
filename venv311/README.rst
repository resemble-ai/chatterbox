jaconv
==========
|coveralls| |pyversion| |version| |license| |download|

jaconv (Japanese Converter) is interconverter for Hiragana, Katakana, Hankaku (half-width character) and Zenkaku (full-width character)

`Japanese README <https://github.com/ikegami-yukino/jaconv/blob/master/README_JP.rst>`_ is available.

INSTALLATION
==============

::

 $ pip install jaconv


USAGE
============

See also `document <http://ikegami-yukino.github.io/jaconv/jaconv.html>`_

.. code:: python

  import jaconv

  # Hiragana to Katakana
  jaconv.hira2kata('ともえまみ')
  # => 'トモエマミ'

  # Hiragana to half-width Katakana
  jaconv.hira2hkata('ともえまみ')
  # => 'ﾄﾓｴﾏﾐ'

  # Katakana to Hiragana
  jaconv.kata2hira('巴マミ')
  # => '巴まみ'

  # half-width character to full-width character
  # default parameters are followings: kana=True, ascii=False, digit=False
  jaconv.h2z('ﾃｨﾛ･ﾌｨﾅｰﾚ')
  # => 'ティロ・フィナーレ'

  # half-width character to full-width character
  # but only ascii characters
  jaconv.h2z('abc', kana=False, ascii=True, digit=False)
  # => 'ａｂｃ'

  # half-width character to full-width character
  # but only digit characters
  jaconv.h2z('123', kana=False, ascii=False, digit=True)
  # => '１２３'

  # half-width character to full-width character
  # except half-width Katakana
  jaconv.h2z('ｱabc123', kana=False, digit=True, ascii=True)
  # => 'ｱａｂｃ１２３'

  # an alias of h2z
  jaconv.hankaku2zenkaku('ﾃｨﾛ･ﾌｨﾅｰﾚabc123')
  # => 'ティロ・フィナーレabc123'

  # full-width character to half-width character
  # default parameters are followings: kana=True, ascii=False, digit=False
  jaconv.z2h('ティロ・フィナーレ')
  # => 'ﾃｨﾛ・ﾌｨﾅｰﾚ'

  # full-width character to half-width character
  # but only ascii characters
  jaconv.z2h('ａｂｃ', kana=False, ascii=True, digit=False)
  # => 'abc'

  # full-width character to half-width character
  # but only digit characters
  jaconv.z2h('１２３', kana=False, ascii=False, digit=True)
  # => '123'

  # full-width character to half-width character
  # except full-width Katakana
  jaconv.z2h('アａｂｃ１２３', kana=False, digit=True, ascii=True)
  # => 'アabc123'

  # an alias of z2h
  jaconv.zenkaku2hankaku('ティロ・フィナーレａｂｃ１２３')
  # => 'ﾃｨﾛ･ﾌｨﾅｰﾚａｂｃ１２３'

  # normalize
  jaconv.normalize('ティロ･フィナ〜レ', 'NFKC')
  # => 'ティロ・フィナーレ'

  # Hiragana to alphabet
  jaconv.kana2alphabet('じゃぱん')
  # => 'japan'

  # Alphabet to Hiragana
  jaconv.alphabet2kana('japan')
  # => 'じゃぱん'

  # Katakana to Alphabet
  jaconv.kata2alphabet('ケツイ')
  # => 'ketsui'

  # Alphabet to Katakana
  jaconv.alphabet2kata('namba')
  # => 'ナンバ'

  # Hiragana to Julius's phoneme format
  jaconv.hiragana2julius('てんきすごくいいいいいい')
  # => 't e N k i s u g o k u i:'


NOTE
============

jaconv.normalize method expand unicodedata.normalize for Japanese language processing.

.. code::

    '〜' => 'ー'
    '～' => 'ー'
    "’" => "'"
    '”'=> '"'
    '“' => '``'
    '―' => '-'
    '‐' => '-'
    '˗' => '-'
    '֊' => '-'
    '‐' => '-'
    '‑' => '-'
    '‒' => '-'
    '–' => '-'
    '⁃' => '-'
    '⁻' => '-'
    '₋' => '-'
    '−' => '-'
    '﹣' => 'ー'
    '－' => 'ー'
    '—' => 'ー'
    '―' => 'ー'
    '━' => 'ー'
    '─' => 'ー'




.. |coveralls| image:: https://coveralls.io/repos/ikegami-yukino/jaconv/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/ikegami-yukino/jaconv?branch=master
    :alt: coveralls.io

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/jaconv.svg

.. |version| image:: https://img.shields.io/pypi/v/jaconv.svg
    :target: http://pypi.python.org/pypi/jaconv/
    :alt: latest version

.. |license| image:: https://img.shields.io/pypi/l/jaconv.svg
    :target: http://pypi.python.org/pypi/jaconv/
    :alt: license

.. |download| image:: https://static.pepy.tech/personalized-badge/neologdn?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads
    :target: https://pepy.tech/project/neologdn
    :alt: download
