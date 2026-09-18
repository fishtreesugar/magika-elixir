defmodule Magika.MixProject do
  use Mix.Project

  @version "0.1.0-rc.0"
  @source_url "https://github.com/fishtreesugar/magika-elixir"
  @upstream_url "https://github.com/google/magika"

  def project do
    [
      app: :magika,
      version: @version,
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      description:
        "Elixir binding of Google's Magika: deep-learning file content type detection.",
      package: package(),
      deps: deps(),
      docs: docs(),
      name: "Magika",
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {Magika.Application, []}
    ]
  end

  defp package do
    [
      files: ~w(lib priv mix.exs README.md LICENSE),
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "Upstream Magika" => @upstream_url,
        "Project page" => "https://securityresearch.google/magika"
      }
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md"],
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end

  defp deps do
    [
      # Elixir bindings for Microsoft ONNX Runtime.
      {:onnxruntime, "~> 0.1.0-rc.1"},
      {:nx, "~> 0.9 or ~> 1.0"},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.37", only: :dev, runtime: false}
    ]
  end
end
