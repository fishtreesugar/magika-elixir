defmodule Magika do
  @moduledoc """
  Elixir binding of [Magika](https://github.com/google/magika), Google's
  deep-learning file content type detector.

  Magika identifies the content type of a file (e.g. `html`, `python`, `pdf`,
  `zip`) from its bytes, using a small ONNX model run via
  [OnnxRuntime](https://hex.pm/packages/onnxruntime). It is a faithful port of
  the reference Python implementation's `standard_v3_3` model and inference
  logic.

  ## Usage

  The model is loaded once and hosted by a supervised `Magika.Server` that
  starts automatically with the `:magika` application. Call the API without
  threading an instance around:

      {:ok, result} = Magika.identify("<!DOCTYPE html>\\n<html>...</html>")
      result.prediction.output.label       #=> "html"
      result.prediction.output.mime_type   #=> "text/html"
      result.prediction.score              #=> 0.99...

      {:ok, result} = Magika.identify_path("/path/to/file.pdf")
      result.prediction.output.label       #=> "pdf"

  ## Prediction mode

  The prediction mode controls how strict Magika is before trusting the model's
  guess. The hosted server uses `:high_confidence` by default; change it in your
  application config:

      config :magika, prediction_mode: :best_guess

  The modes:

    * `:high_confidence` (default) — keep the model prediction only when its
      score clears the per-content-type threshold (falling back to the
      medium-confidence threshold otherwise).
    * `:medium_confidence` — keep the model prediction when its score clears the
      generic medium-confidence threshold.
    * `:best_guess` — always return the model prediction regardless of score.

  When the score is too low for the chosen mode, the output is generalized to
  `txt` (for text content types) or `unknown` (for binary content types).

  ## Standalone instances (advanced)

  You normally don't need this. For one-off scripts or tests you can build an
  instance with `new/1` and pass it as the first argument, bypassing the
  supervised server. A `Magika.t()` is immutable and safe to reuse:

      magika = Magika.new(prediction_mode: :best_guess)
      {:ok, result} = Magika.identify(magika, content)

  A specific named server can also be targeted with the `:server` option:

      {:ok, result} = Magika.identify(content, server: MyApp.Magika)
  """

  alias Magika.{Config, Features, Inference, Prediction, Result}

  @enforce_keys [:model, :config, :prediction_mode]
  defstruct [:model, :config, :prediction_mode]

  @type prediction_mode :: :high_confidence | :medium_confidence | :best_guess

  @type t :: %__MODULE__{
          model: OnnxRuntime.Model.t(),
          config: Config.t(),
          prediction_mode: prediction_mode()
        }

  @doc """
  Creates a new Magika instance, loading the model and configuration.

  ## Options

    * `:prediction_mode` — one of `:high_confidence` (default),
      `:medium_confidence`, `:best_guess`.
    * `:model_path` — path to a custom `model.onnx`. Defaults to the vendored
      `standard_v3_3` model.
    * `:model_config_path` — path to a custom `config.min.json`.
    * `:content_types_kb_path` — path to a custom `content_types_kb.min.json`.
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    prediction_mode = Keyword.get(opts, :prediction_mode, :high_confidence)

    unless prediction_mode in [:high_confidence, :medium_confidence, :best_guess] do
      raise ArgumentError, "invalid :prediction_mode #{inspect(prediction_mode)}"
    end

    model_path = Keyword.get(opts, :model_path, Config.default_model_path())
    config_path = Keyword.get(opts, :model_config_path, Config.default_model_config_path())
    kb_path = Keyword.get(opts, :content_types_kb_path, Config.default_kb_path())

    %__MODULE__{
      model: OnnxRuntime.load(model_path),
      config: Config.load(config_path, kb_path),
      prediction_mode: prediction_mode
    }
  end

  @doc """
  Identifies the content type of the given raw `content` (a binary).

  Resolves the hosted instance from a `Magika.Server`. Pass `server:` to target
  a specific named server (defaults to `Magika.Server`). Alternatively, pass a
  `Magika` instance as the first argument to bypass the server entirely.

  Always returns `{:ok, result}` — identification of in-memory bytes cannot
  fail the way a filesystem read can.
  """
  @spec identify(binary(), keyword()) :: {:ok, Result.t()}
  @spec identify(t(), binary()) :: {:ok, Result.t()}
  def identify(content, opts \\ [])

  def identify(content, opts) when is_binary(content) and is_list(opts) do
    identify(server_instance(opts), content)
  end

  def identify(%__MODULE__{} = magika, content) when is_binary(content) do
    prediction = predict_from_content(magika, content)
    {:ok, %Result{status: :ok, prediction: prediction}}
  end

  @doc """
  Identifies the content type of the file at `path`.

  Resolves the hosted instance from a `Magika.Server`. Pass `server:` to target
  a specific named server (defaults to `Magika.Server`). Alternatively, pass a
  `Magika` instance as the first argument to bypass the server entirely.

  Returns `{:ok, result}` on success, or `{:error, result}` when the path does
  not exist or cannot be read. Directories and other special files are reported
  via dedicated content types (`directory`, `symlink`, `unknown`).

  Files are sampled using bounded reads from the beginning and end. If the
  beginning contains too few meaningful bytes for inference, the UTF-8 fallback
  scans the file incrementally, preserving its classification without loading
  the whole file into memory.
  """
  @spec identify_path(Path.t(), keyword()) :: {:ok, Result.t()} | {:error, Result.t()}
  @spec identify_path(t(), Path.t()) :: {:ok, Result.t()} | {:error, Result.t()}
  def identify_path(path, opts \\ [])

  def identify_path(path, opts) when is_binary(path) and is_list(opts) do
    identify_path(server_instance(opts), path)
  end

  def identify_path(%__MODULE__{} = magika, path) when is_binary(path) do
    case classify_path(magika, path) do
      {:ok, prediction} ->
        {:ok, %Result{status: :ok, prediction: prediction, path: path}}

      {:error, status} ->
        {:error, %Result{status: status, prediction: nil, path: path}}
    end
  end

  @doc """
  Identifies the content type read from an open binary `IO.device`/file.

  Resolves the hosted instance from a `Magika.Server`. Pass `server:` to target
  a specific named server (defaults to `Magika.Server`). Alternatively, pass a
  `Magika` instance as the first argument to bypass the server entirely.

  The whole stream is read into memory (Magika only needs a bounded prefix and
  suffix, but reading fully keeps the implementation simple and correct). The
  caller is responsible for opening and closing the device.
  """
  @spec identify_stream(IO.device(), keyword()) :: {:ok, Result.t()}
  @spec identify_stream(t(), IO.device()) :: {:ok, Result.t()}
  def identify_stream(device, opts \\ [])

  def identify_stream(device, opts) when is_list(opts) and not is_struct(device, __MODULE__) do
    identify_stream(server_instance(opts), device)
  end

  def identify_stream(%__MODULE__{} = magika, device) do
    content =
      case IO.binread(device, :eof) do
        :eof -> <<>>
        data when is_binary(data) -> data
      end

    identify(magika, content)
  end

  defp server_instance(opts) do
    opts |> Keyword.get(:server, Magika.Server.default_name()) |> Magika.Server.instance()
  end

  @doc "Returns the loaded model's name (the model directory basename)."
  @spec model_name(t()) :: String.t()
  def model_name(%__MODULE__{}), do: "standard_v3_3"

  # ── Path corner cases ──────────────────────────────────────────────────────

  defp classify_path(magika, path) do
    case File.lstat(path) do
      {:error, :enoent} ->
        {:error, :file_not_found}

      {:error, _reason} ->
        {:error, :permission_error}

      {:ok, %File.Stat{type: :symlink}} ->
        # Default behaviour follows the symlink; resolve via File.stat.
        classify_resolved(magika, path)

      {:ok, %File.Stat{type: :directory}} ->
        {:ok, special_prediction(magika, "directory")}

      {:ok, %File.Stat{type: :regular}} ->
        read_and_classify(magika, path)

      {:ok, _other} ->
        {:ok, special_prediction(magika, "unknown")}
    end
  end

  defp classify_resolved(magika, path) do
    case File.stat(path) do
      {:error, :enoent} -> {:error, :file_not_found}
      {:error, _} -> {:error, :permission_error}
      {:ok, %File.Stat{type: :directory}} -> {:ok, special_prediction(magika, "directory")}
      {:ok, %File.Stat{type: :regular}} -> read_and_classify(magika, path)
      {:ok, _} -> {:ok, special_prediction(magika, "unknown")}
    end
  end

  defp read_and_classify(magika, path) do
    case File.open(path, [:read, :binary, :raw], &predict_from_file(magika, &1)) do
      {:ok, {:ok, prediction}} -> {:ok, prediction}
      _ -> {:error, :permission_error}
    end
  end

  defp predict_from_file(magika, file) do
    case :file.position(file, :eof) do
      {:ok, size} when size > 0 ->
        predict_from_seekable_file(magika, file, size)

      # Some regular files, such as Linux procfs entries, report zero size or
      # reject seeking to EOF. Read them sequentially with bounded buffers.
      {:ok, 0} ->
        predict_from_sequential_file(magika, file)

      {:error, reason} when reason in [:einval, :espipe] ->
        predict_from_sequential_file(magika, file)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp predict_from_seekable_file(%__MODULE__{config: config} = magika, file, size) do
    with {:ok, prefix} <- read_file_block(file, 0, min(size, config.block_size)) do
      cond do
        size <= config.block_size ->
          {:ok, predict_from_content(magika, prefix)}

        insufficient_bytes?(prefix, config) ->
          # The current in-memory fallback validates the entire content, not
          # just the sampled ends. Keep that behavior with bounded buffers.
          with {:ok, 0} <- :file.position(file, :bof) do
            predict_from_sequential_file(magika, file)
          end

        true ->
          with {:ok, suffix} <-
                 read_file_block(file, size - config.block_size, config.block_size) do
            # Feature extraction only examines the first and last block, so
            # joining these windows preserves the features even when they overlap.
            {:ok, dl_prediction(magika, prefix <> suffix)}
          end
      end
    end
  end

  defp read_file_block(file, offset, size), do: :file.pread(file, offset, size)

  defp predict_from_sequential_file(%__MODULE__{config: config} = magika, file) do
    with {:ok, size, prefix, suffix, valid?} <-
           read_file_sample(file, config.block_size, {0, <<>>, <<>>, <<>>}) do
      cond do
        size <= config.block_size ->
          {:ok, predict_from_content(magika, prefix)}

        insufficient_bytes?(prefix, config) ->
          {:ok, special_prediction(magika, if(valid?, do: "txt", else: "unknown"))}

        true ->
          {:ok, dl_prediction(magika, prefix <> suffix)}
      end
    end
  end

  defp read_file_sample(file, block_size, {size, prefix, suffix, pending}) do
    case :file.read(file, block_size) do
      :eof ->
        {:ok, size, prefix, suffix, pending == <<>>}

      {:ok, chunk} ->
        prefix = binary_slice(prefix <> chunk, 0, block_size)
        tail = suffix <> chunk
        keep = min(byte_size(tail), block_size)
        suffix = binary_part(tail, byte_size(tail) - keep, keep)
        pending = utf8_pending(pending, chunk)
        read_file_sample(file, block_size, {size + byte_size(chunk), prefix, suffix, pending})

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp utf8_pending(false, _chunk), do: false

  defp utf8_pending(pending, chunk) do
    case :unicode.characters_to_binary(pending <> chunk, :utf8, :utf8) do
      valid when is_binary(valid) -> <<>>
      # A UTF-8 character can straddle reads; retain at most three bytes.
      {:incomplete, _valid, rest} -> rest
      {:error, _valid, _rest} -> false
    end
  end

  # ── Content classification ──────────────────────────────────────────────────

  defp predict_from_content(%__MODULE__{config: config} = magika, content) do
    size = byte_size(content)

    cond do
      size == 0 ->
        special_prediction(magika, "empty")

      size < config.min_file_size_for_dl ->
        few_bytes_prediction(magika, content)

      true ->
        # If the n-th token (n = min_file_size_for_dl) is padding, then after
        # stripping whitespace we do not have enough meaningful bytes for a
        # reliable DL prediction; fall back to the few-bytes heuristic.
        if insufficient_bytes?(content, config) do
          few_bytes_prediction(magika, content)
        else
          dl_prediction(magika, content)
        end
    end
  end

  defp insufficient_bytes?(content, config) do
    beg = Features.extract_beg(content, config)
    Enum.at(beg, config.min_file_size_for_dl - 1) == config.padding_token
  end

  defp dl_prediction(%__MODULE__{config: config} = magika, content) do
    features = Features.extract(content, config)
    {dl_label, score} = Inference.predict(magika.model, features, config.target_labels_space)
    {output_label, reason} = resolve_output(magika, dl_label, score)

    %Prediction{
      dl: Config.content_type_info(config, dl_label),
      output: Config.content_type_info(config, output_label),
      score: score,
      overwrite_reason: reason
    }
  end

  # Apply the overwrite map and confidence thresholds to turn a raw DL label
  # and score into the final output label. Mirrors
  # `_get_output_label_from_dl_label_and_score` in the reference.
  defp resolve_output(%__MODULE__{config: config} = magika, dl_label, score) do
    mapped = Map.get(config.overwrite_map, dl_label, dl_label)
    base_reason = if mapped != dl_label, do: :overwrite_map, else: :none

    keep? =
      case magika.prediction_mode do
        :best_guess ->
          true

        :high_confidence ->
          threshold = Map.get(config.thresholds, dl_label, config.medium_confidence_threshold)
          score >= threshold

        :medium_confidence ->
          score >= config.medium_confidence_threshold
      end

    if keep? do
      {mapped, base_reason}
    else
      # Not confident enough: generalize to txt or unknown based on whether the
      # (mapped) content type is textual.
      generalized =
        if Config.content_type_info(config, mapped).is_text, do: "txt", else: "unknown"

      reason = if dl_label == generalized, do: :none, else: :low_confidence
      {generalized, reason}
    end
  end

  # For very small files (or files that are mostly whitespace), decide between
  # txt and unknown by attempting to interpret the bytes as UTF-8.
  defp few_bytes_prediction(%__MODULE__{config: config}, content) do
    label = if String.valid?(content), do: "txt", else: "unknown"
    undefined_special(config, label)
  end

  defp special_prediction(%__MODULE__{config: config}, output_label) do
    undefined_special(config, output_label)
  end

  # Build a prediction for inputs handled without the model: the DL label is the
  # special `undefined` content type, the score is 1.0, and there is no
  # overwrite.
  defp undefined_special(config, output_label) do
    %Prediction{
      dl: Config.content_type_info(config, "undefined"),
      output: Config.content_type_info(config, output_label),
      score: 1.0,
      overwrite_reason: :none
    }
  end
end
