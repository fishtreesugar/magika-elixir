defmodule Magika.Prediction do
  @moduledoc """
  The outcome of identifying a single input.

  Fields:

    * `:output` — the `Magika.ContentTypeInfo` Magika returns to the user. This
      accounts for confidence thresholds and the overwrite map.
    * `:dl` — the `Magika.ContentTypeInfo` of the raw deep-learning model
      prediction. For inputs handled without the model (empty files, very small
      files, directories, symlinks) this is the special `undefined` type.
    * `:score` — the model's confidence for `:dl` in `[0.0, 1.0]`. For inputs
      handled without the model this is `1.0`.
    * `:overwrite_reason` — why `:output` differs from `:dl`, one of
      `:none`, `:low_confidence`, `:overwrite_map`.
  """

  alias Magika.ContentTypeInfo

  @enforce_keys [:output, :dl, :score, :overwrite_reason]
  defstruct [:output, :dl, :score, :overwrite_reason]

  @type overwrite_reason :: :none | :low_confidence | :overwrite_map

  @type t :: %__MODULE__{
          output: ContentTypeInfo.t(),
          dl: ContentTypeInfo.t(),
          score: float(),
          overwrite_reason: overwrite_reason()
        }
end
