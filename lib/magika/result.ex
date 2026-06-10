defmodule Magika.Result do
  @moduledoc """
  The top-level result of an identification call.

  `:status` is `:ok` when identification succeeded, in which case
  `:prediction` holds a `Magika.Prediction`. On failure (e.g. a missing or
  unreadable file) `:status` carries the error reason and `:prediction` is
  `nil`.
  """

  alias Magika.Prediction

  @enforce_keys [:status]
  defstruct status: :ok, prediction: nil, path: nil

  @type status :: :ok | :file_not_found | :permission_error

  @type t :: %__MODULE__{
          status: status(),
          prediction: Prediction.t() | nil,
          path: Path.t() | nil
        }

  @doc "Returns true when the result holds a successful prediction."
  @spec ok?(t()) :: boolean()
  def ok?(%__MODULE__{status: :ok}), do: true
  def ok?(%__MODULE__{}), do: false
end
