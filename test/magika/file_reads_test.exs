defmodule Magika.FileReadsTest do
  # The read-volume test temporarily installs Erlang call tracing patterns.
  use ExUnit.Case, async: false

  @moduletag :tmp_dir

  setup_all do
    {:ok, magika: Magika.new()}
  end

  test "path predictions match in-memory predictions around sampling boundaries", context do
    block_size = context.magika.config.block_size
    pattern = "import sys\n\ndef main():\n    print(sys.argv)\n\nmain()\n"

    for size <- [
          0,
          1,
          7,
          8,
          1023,
          1024,
          block_size - 1,
          block_size,
          block_size + 1,
          2 * block_size - 1,
          2 * block_size,
          2 * block_size + 1
        ] do
      content = binary_part(String.duplicate(pattern, div(size, byte_size(pattern)) + 1), 0, size)
      assert_same_prediction(context, content)
    end
  end

  test "large files preserve predictions in every prediction mode", context do
    content =
      "  \n\t<!DOCTYPE html>\n<html><body>" <>
        String.duplicate("<p>A document with a large middle section.</p>\n", 100_000) <>
        "</body></html>\n\t  "

    for mode <- [:high_confidence, :medium_confidence, :best_guess] do
      magika = %{context.magika | prediction_mode: mode}
      assert_same_prediction(%{context | magika: magika}, content)
    end
  end

  test "whitespace fallback still validates bytes outside the sampled ends", context do
    block_size = context.magika.config.block_size

    for middle <- ["", "text", <<0>>, <<255>>, <<0xC0, 0xAF>>, <<0xED, 0xA0, 0x80>>] do
      content =
        String.duplicate(" ", block_size) <> middle <> String.duplicate(" ", 2 * block_size)

      assert_same_prediction(context, content)
    end

    # Seven meaningful bytes in the first block also trigger the fallback.
    assert_same_prediction(context, "abcdefg" <> String.duplicate(" ", 3 * block_size) <> <<255>>)
  end

  test "whitespace fallback handles UTF-8 sequences across read boundaries", context do
    block_size = context.magika.config.block_size

    for character <- ["¢", "€", "😀"], split <- 1..(byte_size(character) - 1) do
      content =
        String.duplicate(" ", 2 * block_size - split) <>
          character <> String.duplicate(" ", block_size)

      assert_same_prediction(context, content)
    end

    for ending <- [<<0xC2>>, <<0xE2, 0x82>>, <<0xF0, 0x9F, 0x98>>] do
      assert_same_prediction(context, String.duplicate(" ", 2 * block_size - 1) <> ending)
    end
  end

  test "normal large files read at most two blocks of file data", context do
    path = Path.join(context.tmp_dir, "large.py")
    size = 64 * 1024 * 1024

    File.open!(path, [:write, :binary, :raw], fn file ->
      :ok = :file.pwrite(file, 0, "import sys\nprint(sys.argv)\n")
      :ok = :file.pwrite(file, size - 1, "\n")
    end)

    {{:ok, result}, read_sizes} =
      trace_file_reads(fn -> Magika.identify_path(context.magika, path) end)

    assert result.status == :ok
    assert read_sizes != []
    assert Enum.sum(read_sizes) <= 2 * context.magika.config.block_size
  end

  test "whitespace fallback scans the full file with bounded buffers", context do
    path = Path.join(context.tmp_dir, "whitespace.txt")
    content = String.duplicate(" ", 128 * context.magika.config.block_size)
    File.write!(path, content)

    {{:ok, result}, read_sizes} =
      trace_file_reads(fn -> Magika.identify_path(context.magika, path) end)

    assert result.prediction.output.label == "txt"
    assert result.prediction.dl.label == "undefined"
    assert Enum.sum(read_sizes) >= byte_size(content)
    assert Enum.max(read_sizes) <= context.magika.config.block_size
  end

  @tag skip: not File.exists?("/proc/version")
  test "Linux procfs files preserve predictions despite reporting zero size", context do
    for path <- ["/proc/version", "/proc/self/cmdline"] do
      assert File.stat!(path).size == 0
      content = File.read!(path)
      assert byte_size(content) > 0
      {:ok, expected} = Magika.identify(context.magika, content)
      {:ok, actual} = Magika.identify_path(context.magika, path)
      assert actual.prediction == expected.prediction
    end
  end

  defp assert_same_prediction(context, content) do
    path = Path.join(context.tmp_dir, "sample")
    File.write!(path, content)

    {:ok, expected} = Magika.identify(context.magika, content)
    {:ok, actual} = Magika.identify_path(context.magika, path)

    assert actual.path == path
    assert actual.status == expected.status
    assert actual.prediction == expected.prediction
  end

  # Measure bytes returned by the file APIs, including the original whole-file
  # read. Tracing only the worker avoids counting fixture setup and model loading.
  defp trace_file_reads(fun) do
    parent = self()

    worker =
      spawn_link(fn ->
        receive do
          :run -> send(parent, {self(), :result, fun.()})
        end

        receive do
          :stop -> :ok
        end
      end)

    functions = [{:file, :read_file, 1}, {:file, :read, 2}, {:file, :pread, 3}]

    try do
      for function <- functions do
        :erlang.trace_pattern(function, [{:_, [], [{:return_trace}]}], [:local])
      end

      :erlang.trace(worker, true, [:call, {:tracer, self()}])
      send(worker, :run)
      assert_receive {^worker, :result, result}, 10_000

      ref = :erlang.trace_delivered(worker)
      assert_receive {:trace_delivered, ^worker, ^ref}, 10_000

      {result, collect_read_sizes(worker, [])}
    after
      :erlang.trace(worker, false, [:call])
      for function <- functions, do: :erlang.trace_pattern(function, false, [:local])
      send(worker, :stop)
    end
  end

  defp collect_read_sizes(worker, sizes) do
    receive do
      {:trace, ^worker, :return_from, {:file, _, _}, {:ok, data}} when is_binary(data) ->
        collect_read_sizes(worker, [byte_size(data) | sizes])

      {:trace, ^worker, :return_from, _, _} ->
        collect_read_sizes(worker, sizes)

      {:trace, ^worker, :call, _} ->
        collect_read_sizes(worker, sizes)
    after
      0 -> sizes
    end
  end
end
