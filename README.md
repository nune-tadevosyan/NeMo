[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<h2>🔁 TopIPL: Iterative Pseudo-Labeling for ASR</h2>

<p>This repository includes the <strong>TopIPL</strong> implementation for <em>Automatic Speech Recognition (ASR)</em>.<br>
It introduces a flexible training mechanism where you can enable Iterative Pseudo-Labeling simply by modifying the config file – no code changes required.</p>

<h3>🛠️ How to Enable TopIPL in Config</h3>

<p>To get started, add the following block under <code>ipl:</code> in your experiment config. 
You can refer to an example configuration here: 
<a href="https://github.com/nune-tadevosyan/NeMo/blob/TopIPL/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml" target="_blank">
fastconformer_hybrid_transducer_ctc_bpe.yaml
</a>.
</p>
<pre><code>ipl:
  n_epochs: &lt;int&gt;            # Number of epochs before first pseudo-label generation
  restore_pc: &lt;bool&gt;         # Whether to restore pseudo-labels if they already exist (default: False)
  manifest_filepath: &lt;str&gt;   # Path to the original dataset manifest
  tarred_audio_filepaths: &lt;str&gt; # Path to tarred audio files (if applicable)
  is_tarred: &lt;bool&gt;          # Whether the dataset is tarred
  dataset_weights: &lt;float|list&gt; # Fraction or weights of dataset to use (non-tar only, default: 1)
  limit_train_batches: &lt;int&gt; # Used only for Lhotse-style manifests during training with PLs
  cache_manifest: &lt;str&gt;      # Path to the cached manifest file
  m_epochs: 0                # Deprecated: use `max_steps` to control training instead
  p_cache: &lt;float&gt;           # Probability to update pseudo-label cache
  cache_prefix: &lt;str&gt;        # Prefix for saved cache files
  batch_size: &lt;int&gt;          # Batch size for generating pseudo-labels
  do_average: &lt;bool&gt;         # If True, average multiple checkpoints for PL generation
  path_to_model: &lt;str&gt;       # Checkpoint paths to average (required if `do_average` is True)
</code></pre>

<p>
  Then you can run the training using the following script:
  <code>examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py</code>, 
  passing your config file as a parameter.
</p>

<h3>📌 Notes</h3>
<ul>
  <li>✅ Current implementation supports <strong>Hybrid</strong> and <strong>CTC</strong> models.</li>
  <li>⚙️ You can extend this to any other ASR model by inheriting from <code>IPLMixin</code>.</li>
  <li>🧾 For training with <strong>tarred datasets using Lhotse</strong>, make sure to:
    <ul>
      <li>Set <code>skip_manifest_entries: True</code> in the config.</li>
    </ul>
  </li>
  <li>⏱️ The <code>max_steps</code> parameter should be set explicitly to ensure the learning rate scheduler behaves correctly. This applies to all training modes.</li>
</ul>


<h3>💡 Additional Considerations</h3>
<ul>
  <li>⚠️ In <strong>extensive data settings</strong>, this in-place implementation may cause memory issues (though this was not observed during our internal experiments).</li>
  <li>🧪 To avoid such issues, we recommend using an alternative implementation available in the <strong>Speech Data Processing</strong> examples. 
    This version:
    <ul>
      <li>Stops training at the end of each epoch,</li>
      <li>Generates pseudo-labels offline,</li>
      <li>And restarts training with the updated labels.</li>
    </ul>
  </li>
  <li>⚙️ This approach is compatible with <code>nemo-run</code>, and the generated commands can be submitted as separate jobs.</li>
  <li>🔗 See: <a href="https://github.com/NVIDIA/NeMo-speech-data-processor/pull/121" target="_blank">Speech Data Processing Examples</a></li>
  <li>📦 When using the Speech Data Processing approach to run training, you must include the <code>IPLCallback</code> from NeMo by passing it to <code>exp_manager</code> as a callback.
    <br>
    🔗 See: <a href="https://github.com/NVIDIA/NeMo/pull/13671" target="_blank">IPL Callback in NeMo</a>
  </li>
</ul>
