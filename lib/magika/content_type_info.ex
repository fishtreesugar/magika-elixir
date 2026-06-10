defmodule Magika.ContentTypeInfo do
  @moduledoc """
  Metadata about a single content type, loaded from Magika's content types
  knowledge base (`content_types_kb.min.json`).
  """

  @enforce_keys [:label, :mime_type, :group, :description, :extensions, :is_text]
  defstruct [:label, :mime_type, :group, :description, :extensions, :is_text]

  @type t :: %__MODULE__{
          label: String.t(),
          mime_type: String.t(),
          group: String.t(),
          description: String.t(),
          extensions: [String.t()],
          is_text: boolean()
        }
end
