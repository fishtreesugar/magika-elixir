defmodule Magika.Config do
  @moduledoc """
  Loads and holds Magika's model configuration and the content-types knowledge
  base.

  This mirrors `ModelConfig` and the `content_types_kb` loading logic from the
  reference Python implementation. The data is read from the JSON files vendored
  under `priv/` and is intended to be loaded once and reused.
  """

  alias Magika.ContentTypeInfo

  @txt_mime_type "text/plain"
  @unknown_mime_type "application/octet-stream"
  @unknown_group "unknown"

  @enforce_keys [
    :beg_size,
    :mid_size,
    :end_size,
    :use_inputs_at_offsets,
    :medium_confidence_threshold,
    :min_file_size_for_dl,
    :padding_token,
    :block_size,
    :target_labels_space,
    :thresholds,
    :overwrite_map,
    :content_types
  ]
  defstruct @enforce_keys

  @type t :: %__MODULE__{
          beg_size: non_neg_integer(),
          mid_size: non_neg_integer(),
          end_size: non_neg_integer(),
          use_inputs_at_offsets: boolean(),
          medium_confidence_threshold: float(),
          min_file_size_for_dl: non_neg_integer(),
          padding_token: non_neg_integer(),
          block_size: non_neg_integer(),
          # ordered list of labels; index matches the model output dimension
          target_labels_space: [String.t()],
          thresholds: %{String.t() => float()},
          overwrite_map: %{String.t() => String.t()},
          content_types: %{String.t() => ContentTypeInfo.t()}
        }

  @doc """
  Loads the configuration for the given model directory and content-types KB.

  Both arguments default to the files vendored in this package's `priv/`.
  """
  @spec load(Path.t(), Path.t()) :: t()
  def load(model_config_path \\ default_model_config_path(), kb_path \\ default_kb_path()) do
    config = model_config_path |> File.read!() |> Jason.decode!()
    content_types = load_content_types_kb(kb_path)

    %__MODULE__{
      beg_size: config["beg_size"],
      mid_size: config["mid_size"],
      end_size: config["end_size"],
      use_inputs_at_offsets: config["use_inputs_at_offsets"],
      medium_confidence_threshold: config["medium_confidence_threshold"],
      min_file_size_for_dl: config["min_file_size_for_dl"],
      padding_token: config["padding_token"],
      block_size: config["block_size"],
      target_labels_space: config["target_labels_space"],
      thresholds: config["thresholds"],
      overwrite_map: config["overwrite_map"],
      content_types: content_types
    }
  end

  @doc "Path to the default vendored model directory."
  @spec default_model_dir() :: Path.t()
  def default_model_dir, do: Path.join([priv_dir(), "models", "standard_v3_3"])

  @doc "Path to the default vendored ONNX model."
  @spec default_model_path() :: Path.t()
  def default_model_path, do: Path.join(default_model_dir(), "model.onnx")

  @doc "Path to the default vendored model config JSON."
  @spec default_model_config_path() :: Path.t()
  def default_model_config_path, do: Path.join(default_model_dir(), "config.min.json")

  @doc "Path to the default vendored content-types knowledge base JSON."
  @spec default_kb_path() :: Path.t()
  def default_kb_path, do: Path.join([priv_dir(), "config", "content_types_kb.min.json"])

  @doc """
  Returns the `ContentTypeInfo` for a label, raising if it is unknown.
  """
  @spec content_type_info(t(), String.t()) :: ContentTypeInfo.t()
  def content_type_info(%__MODULE__{content_types: cts}, label) do
    case Map.fetch(cts, label) do
      {:ok, info} -> info
      :error -> raise ArgumentError, "unknown content type label: #{inspect(label)}"
    end
  end

  defp load_content_types_kb(kb_path) do
    kb_path
    |> File.read!()
    |> Jason.decode!()
    |> Map.new(fn {name, info} ->
      is_text = info["is_text"]
      default_mime = if is_text, do: @txt_mime_type, else: @unknown_mime_type

      ct_info = %ContentTypeInfo{
        label: name,
        mime_type: info["mime_type"] || default_mime,
        group: info["group"] || @unknown_group,
        description: info["description"] || name,
        extensions: info["extensions"] || [],
        is_text: is_text
      }

      {name, ct_info}
    end)
  end

  defp priv_dir, do: :code.priv_dir(:magika) |> to_string()
end
