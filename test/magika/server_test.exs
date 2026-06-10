defmodule Magika.ServerTest do
  use ExUnit.Case, async: true

  describe "default server (started by the :magika application)" do
    test "is running and hosts an instance" do
      assert Magika.Server.running?()
      assert %Magika{} = Magika.Server.instance()
    end

    test "server-backed identify/1 resolves the hosted instance" do
      {:ok, r} = Magika.identify("import sys\n\ndef main():\n    print(sys.argv)\n\nmain()\n")
      assert r.prediction.output.label == "python"
    end

    test "server-backed identify_path/1 resolves the hosted instance" do
      {:ok, r} = Magika.identify_path("mix.exs")
      assert r.prediction.output.label == "elixir"
    end

    test "server-backed identify_stream/1 resolves the hosted instance" do
      path =
        Path.join(System.tmp_dir!(), "magika_srv_stream_#{System.unique_integer([:positive])}")

      File.write!(path, "<!DOCTYPE html>\n<html><body><p>hello there friend</p></body></html>")
      on_exit(fn -> File.rm(path) end)

      {:ok, device} = File.open(path, [:read, :binary])
      {:ok, r} = Magika.identify_stream(device)
      File.close(device)

      assert r.prediction.output.label == "html"
    end
  end

  describe "custom named server" do
    test "can be started independently and addressed via :server option" do
      name = :"magika_test_#{System.unique_integer([:positive])}"
      start_supervised!({Magika.Server, name: name, prediction_mode: :best_guess})

      assert Magika.Server.running?(name)
      {:ok, r} = Magika.identify("a,b,c\n1,2,3\n", server: name)
      # best_guess never generalizes to txt/unknown due to low confidence.
      assert r.prediction.output.label == r.prediction.dl.label
    end

    test "erases its published instance on shutdown" do
      name = :"magika_test_#{System.unique_integer([:positive])}"
      pid = start_supervised!({Magika.Server, name: name})
      assert Magika.Server.running?(name)

      :ok = stop_supervised(Magika.Server)
      refute Process.alive?(pid)
      refute Magika.Server.running?(name)
    end
  end

  test "instance/1 raises for an unknown server" do
    assert_raise RuntimeError, ~r/is not running/, fn ->
      Magika.Server.instance(:no_such_magika_server)
    end
  end
end
