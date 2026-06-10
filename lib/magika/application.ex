defmodule Magika.Application do
  @moduledoc """
  Starts the default `Magika.Server` under a supervisor when the `:magika`
  application boots.

  The model is loaded once and hosted for the lifetime of the application; the
  API in `Magika` resolves this server automatically.

  The only configurable knob is the prediction mode (see `Magika` for what the
  modes mean), set in your application config:

      config :magika, prediction_mode: :best_guess

  It defaults to `:high_confidence`.
  """

  use Application

  @impl true
  def start(_type, _args) do
    prediction_mode = Application.get_env(:magika, :prediction_mode, :high_confidence)
    children = [{Magika.Server, prediction_mode: prediction_mode}]
    Supervisor.start_link(children, strategy: :one_for_one, name: Magika.Supervisor)
  end
end
