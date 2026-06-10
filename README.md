# Magika

An Elixir binding of [Google's Magika](https://securityresearch.google/magika) —
deep-learning file content type detection.

Magika identifies the content type of a file (e.g. `html`, `python`, `pdf`,
`zip`, `png`) from its bytes using a small, fast ONNX model. This library is a
faithful port of the reference Python implementation's `standard_v3_3` model:
it runs the **same ONNX model** (vendored under `priv/`) via
[OnnxRuntime](https://hex.pm/packages/onnxruntime) and reproduces the same
feature extraction, confidence thresholds, and label-resolution logic. Its
output matches the Python tool exactly on the upstream test corpus (77/77
files).

## How it works

1. **Corner cases first.** Empty inputs → `empty`; very small or
   whitespace-only inputs are classified as `txt`/`unknown` by a UTF-8 check,
   without invoking the model. Directories and symlinks get dedicated labels.
2. **Feature extraction.** For larger inputs, Magika reads up to `block_size`
   (4096) bytes from the start and end, strips ASCII whitespace, and takes
   `beg_size` (1024) bytes from the front and `end_size` (1024) from the back,
   padding with a dedicated padding token (256). This yields a 2048-int vector.
3. **Inference.** The vector is fed to the ONNX model (`int32[batch, 2048] →
   float32[batch, 214]`, a softmax over 214 content types). The argmax is the
   raw "deep-learning" label and its probability is the score.
4. **Label resolution.** An overwrite map and per-content-type confidence
   thresholds turn the raw label into the final output. Low-confidence
   predictions are generalized to `txt` (text) or `unknown` (binary).

## Installation

Add `magika` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:magika, "~> 0.1.0"}
  ]
end
```

The ONNX runtime is provided by the
[`onnxruntime`](https://hex.pm/packages/onnxruntime) package (Elixir bindings
for Microsoft ONNX Runtime), which fetches precompiled native binaries for
common platforms, so no manual toolchain setup is required for installation.

## Usage

The model is loaded once and hosted by a supervised `Magika.Server` that starts
automatically with the `:magika` application — so you call the API directly,
without managing or passing around an instance:

```elixir
# Identify raw bytes:
{:ok, result} = Magika.identify("<!DOCTYPE html>\n<html>...</html>")
result.prediction.output.label      #=> "html"
result.prediction.output.mime_type  #=> "text/html"
result.prediction.output.group      #=> "code"
result.prediction.score             #=> 0.86...

# Identify a file on disk:
{:ok, result} = Magika.identify_path("/path/to/document.pdf")
result.prediction.output.label      #=> "pdf"

# Missing/unreadable files return an {:error, result} with a status:
{:error, result} = Magika.identify_path("/nope")
result.status                       #=> :file_not_found

# Identify from an open binary device:
{:ok, device} = File.open("photo.png", [:read, :binary])
{:ok, result} = Magika.identify_stream(device)
File.close(device)
result.prediction.output.label      #=> "png"
```

Inference runs in the **calling** process: the server owns the instance's
lifecycle and publishes it via `:persistent_term`, so concurrent calls don't
serialize through a single mailbox and the configuration isn't copied per call.

### Prediction mode

The prediction mode controls how strict Magika is before trusting the model's
guess. The hosted server uses `:high_confidence` by default; change it in your
application config:

```elixir
# config/config.exs
config :magika, prediction_mode: :best_guess
```

* `:high_confidence` (default) — keep the model prediction only when its score
  clears the per-content-type threshold (falling back to the medium threshold
  otherwise).
* `:medium_confidence` — keep it when the score clears the generic medium
  threshold.
* `:best_guess` — always return the raw model prediction.

When the score is too low for the chosen mode, the output is generalized to
`txt` (for textual content types) or `unknown` (for binary ones), and
`result.prediction.overwrite_reason` is set to `:low_confidence`.

### Standalone instances (advanced)

You normally don't need this. For one-off scripts or tests you can build an
instance directly with `Magika.new/1` and pass it as the first argument,
bypassing the supervised server:

```elixir
magika = Magika.new(prediction_mode: :best_guess)
{:ok, result} = Magika.identify(magika, "<!DOCTYPE html>...")
```

## Result shape

`Magika.identify*/2` returns `{:ok, %Magika.Result{}}` or, for filesystem
errors, `{:error, %Magika.Result{}}`:

```
%Magika.Result{
  status: :ok,                       # or :file_not_found | :permission_error
  path: "/path/to/file" | nil,
  prediction: %Magika.Prediction{
    output: %Magika.ContentTypeInfo{ # what Magika reports to you
      label: "python",
      mime_type: "text/x-python",
      group: "code",
      description: "Python source",
      extensions: ["py", ...],
      is_text: true
    },
    dl: %Magika.ContentTypeInfo{...}, # raw model prediction (or `undefined`)
    score: 0.9998,                    # model confidence for `dl`
    overwrite_reason: :none           # :none | :low_confidence | :overwrite_map
  }
}
```

## Model

The vendored model is Magika's `standard_v3_3` (Apache-2.0), copied verbatim
from the [upstream repository](https://github.com/google/magika) along with its
`config.min.json` and `content_types_kb.min.json`.

## License

Apache-2.0, matching upstream Magika. The bundled model and configuration files
are © Google LLC.
