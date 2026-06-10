defmodule Magika.Features do
  @moduledoc """
  Feature extraction (v2) from a file's content.

  This is a faithful port of Magika's `_extract_features_from_seekable` logic:

    * Read at most `block_size` bytes from the beginning and from the end.
    * Strip ASCII whitespace (`lstrip` for the beginning, `rstrip` for the end).
    * Take `beg_size` bytes from the start and `end_size` bytes from the end,
      padding with `padding_token` when there are not enough bytes.

  The resulting feature vector is `beg ++ end`, a list of `beg_size + end_size`
  integers, suitable for feeding to the model.

  `mid` features and `use_inputs_at_offsets` are not implemented, matching the
  current reference implementation (which asserts `mid_size == 0` and
  `use_inputs_at_offsets == false`).
  """

  alias Magika.Config

  # The bytes Python's bytes.strip()/lstrip()/rstrip() treat as whitespace:
  # \t \n \v \f \r and space.
  defguardp is_whitespace(byte) when byte in [?\t, ?\n, ?\v, ?\f, ?\r, ?\s]

  @doc """
  Extracts the model feature vector from `content`.

  Returns a list of `beg_size + end_size` integers.
  """
  @spec extract(binary(), Config.t()) :: [non_neg_integer()]
  def extract(content, %Config{} = config) do
    beg_feature(content, config) ++ end_feature(content, config)
  end

  @doc """
  Returns the `beg` portion of the feature vector only.

  Used to replicate the reference check on whether, post-stripping, there are
  enough meaningful bytes (i.e. `beg[min_file_size_for_dl - 1]` is not padding).
  """
  @spec extract_beg(binary(), Config.t()) :: [non_neg_integer()]
  def extract_beg(content, %Config{} = config), do: beg_feature(content, config)

  # The beginning feature: read the first block, lstrip, then take `beg_size`
  # bytes from the front, right-padding when there aren't enough.
  defp beg_feature(_content, %Config{beg_size: 0}), do: []

  defp beg_feature(content, %Config{beg_size: beg_size, padding_token: pad} = config) do
    content
    |> first_bytes(config.block_size)
    |> lstrip()
    |> first_bytes(beg_size)
    |> :binary.bin_to_list()
    |> pad_trailing(beg_size, pad)
  end

  # The end feature: read the last block, rstrip, then take `end_size` bytes
  # from the back, left-padding when there aren't enough.
  defp end_feature(_content, %Config{end_size: 0}), do: []

  defp end_feature(content, %Config{end_size: end_size, padding_token: pad} = config) do
    content
    |> last_bytes(config.block_size)
    |> rstrip()
    |> last_bytes(end_size)
    |> :binary.bin_to_list()
    |> pad_leading(end_size, pad)
  end

  # `binary_slice/3` clamps to the available bytes, so these never raise when
  # the binary is shorter than `count`.
  defp first_bytes(bin, count), do: binary_slice(bin, 0, count)

  defp last_bytes(bin, count) when byte_size(bin) <= count, do: bin
  defp last_bytes(bin, count), do: binary_part(bin, byte_size(bin) - count, count)

  defp pad_trailing(ints, size, pad), do: ints ++ List.duplicate(pad, size - length(ints))
  defp pad_leading(ints, size, pad), do: List.duplicate(pad, size - length(ints)) ++ ints

  defp lstrip(<<byte, rest::binary>>) when is_whitespace(byte), do: lstrip(rest)
  defp lstrip(bin), do: bin

  defp rstrip(bin), do: binary_slice(bin, 0, rstrip_length(bin, byte_size(bin)))

  defp rstrip_length(_bin, 0), do: 0

  defp rstrip_length(bin, length) do
    case :binary.at(bin, length - 1) do
      byte when is_whitespace(byte) -> rstrip_length(bin, length - 1)
      _ -> length
    end
  end
end
