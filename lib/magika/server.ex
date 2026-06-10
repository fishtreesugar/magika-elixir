defmodule Magika.Server do
  @moduledoc """
  A supervised process that owns a loaded `Magika` instance.

  The server loads the ONNX model and configuration once in `init/1` and
  publishes the resulting `Magika.t()` into `:persistent_term`. The model and
  configuration are immutable and ONNX Runtime inference is thread-safe, so
  callers fetch the instance with `instance/1` and run inference directly in
  their own process — there is no serialization through the server's mailbox
  and no per-call copying of the (large) configuration.

  The server therefore owns the *lifecycle* of the instance: it loads it on
  start and erases the published term on shutdown. It is normally started for
  you by `Magika.Application`, but you can also place it in your own
  supervision tree:

      children = [
        {Magika.Server, prediction_mode: :high_confidence}
      ]

  Multiple independently-named servers can coexist:

      {Magika.Server, name: MyApp.Magika, model_path: "..."}

  and are then addressed via `Magika.identify(content, server: MyApp.Magika)`.
  """

  use GenServer

  @default_name __MODULE__

  @doc "The default registered name used when none is given."
  @spec default_name() :: atom()
  def default_name, do: @default_name

  @doc """
  Starts the server.

  Accepts all options of `Magika.new/1` plus `:name` (the registered process
  name, defaulting to `Magika.Server`).
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, @default_name)
    GenServer.start_link(__MODULE__, Keyword.put(opts, :name, name), name: name)
  end

  @doc """
  Returns the hosted `Magika.t()` for the given server name.

  Raises if the named server is not running.
  """
  @spec instance(atom()) :: Magika.t()
  def instance(name \\ @default_name) do
    case :persistent_term.get(key(name), nil) do
      nil ->
        raise "Magika server #{inspect(name)} is not running. Ensure the :magika " <>
                "application is started, or add `Magika.Server` to your supervision tree."

      instance ->
        instance
    end
  end

  @doc "Returns true if a server with the given name has published its instance."
  @spec running?(atom()) :: boolean()
  def running?(name \\ @default_name) do
    :persistent_term.get(key(name), nil) != nil
  end

  @impl true
  def init(opts) do
    {name, new_opts} = Keyword.pop!(opts, :name)
    # Make a best-effort attempt to clean up the published term on shutdown.
    Process.flag(:trap_exit, true)
    instance = Magika.new(new_opts)
    :persistent_term.put(key(name), instance)
    {:ok, %{name: name, instance: instance}}
  end

  @impl true
  def handle_call(:instance, _from, state) do
    {:reply, state.instance, state}
  end

  @impl true
  def terminate(_reason, %{name: name}) do
    :persistent_term.erase(key(name))
    :ok
  end

  defp key(name), do: {__MODULE__, name}
end
