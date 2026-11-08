# Third-party notices

Chatterbox was created by Resemble AI. Its MIT license and copyright notice
remain in [LICENSE](LICENSE). This fork retains the copyright and license
headers in the source files.

The S3 generation implementation includes code with Apache License 2.0
headers from Alibaba, Matcha-TTS, WeNet, ESPnet, Mobvoi, Johns Hopkins
University, Shigeki Karita, Xingchen Song, and other authors identified in
those files. The license text is included in
[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). See the headers under
`src/chatterbox/models/s3gen/` for each file's attribution.

The voice encoder and other components also retain their original MIT
headers. Installed dependencies carry their own licenses. Preserve their
notices when redistributing a dependency bundle or wheelhouse.

Model checkpoints are distributed separately from this Python package.
`models.lock.json` records the upstream model repositories and their model
card license declarations when inspected. Model backups preserve the complete
upstream snapshot, including model cards and any license files. A license
declaration for an upstream checkpoint does not describe independently
published conversions or fine-tunes.
