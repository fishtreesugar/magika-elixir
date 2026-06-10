defmodule MagikaTest do
  use ExUnit.Case, async: true

  # One shared Magika instance for the whole module: loading the model is the
  # expensive part and it is immutable, so it is safe to reuse.
  setup_all do
    {:ok, magika: Magika.new()}
  end

  describe "identify/2 with raw content" do
    test "classifies an empty input as empty", %{magika: m} do
      {:ok, r} = Magika.identify(m, "")
      assert r.prediction.output.label == "empty"
      assert r.prediction.dl.label == "undefined"
      assert r.prediction.score == 1.0
      assert r.prediction.output.mime_type == "inode/x-empty"
    end

    test "classifies tiny textual input as txt without the model", %{magika: m} do
      {:ok, r} = Magika.identify(m, "hi")
      assert r.prediction.output.label == "txt"
      assert r.prediction.dl.label == "undefined"
      assert r.prediction.output.is_text
    end

    test "classifies tiny binary input as unknown without the model", %{magika: m} do
      {:ok, r} = Magika.identify(m, <<0xFF, 0xFE, 0x00>>)
      assert r.prediction.output.label == "unknown"
      assert r.prediction.dl.label == "undefined"
    end

    test "classifies whitespace-only input as txt (insufficient meaningful bytes)", %{magika: m} do
      {:ok, r} = Magika.identify(m, String.duplicate(" ", 4096))
      assert r.prediction.output.label == "txt"
      assert r.prediction.dl.label == "undefined"
    end

    test "classifies a python snippet with high confidence", %{magika: m} do
      content =
        "import os\nimport sys\n\ndef main():\n    print(\"hello world\")\n\nif __name__ == \"__main__\":\n    main()\n"

      {:ok, r} = Magika.identify(m, content)
      assert r.prediction.output.label == "python"
      assert r.prediction.score > 0.99
      assert r.prediction.output.mime_type == "text/x-python"
    end
  end

  describe "identify_path/2 corner cases" do
    test "returns an error result for a missing path", %{magika: m} do
      {:error, r} = Magika.identify_path(m, "/no/such/file/anywhere.xyz")
      assert r.status == :file_not_found
      assert r.prediction == nil
    end

    test "classifies a directory as directory", %{magika: m} do
      {:ok, r} = Magika.identify_path(m, File.cwd!())
      assert r.prediction.output.label == "directory"
      assert r.prediction.dl.label == "undefined"
    end

    test "reads and classifies a regular file", %{magika: m} do
      path = Path.join(System.tmp_dir!(), "magika_test_#{System.unique_integer([:positive])}.py")
      File.write!(path, "def f(x):\n    return x + 1\n\nprint(f(41))\n")
      on_exit(fn -> File.rm(path) end)

      {:ok, r} = Magika.identify_path(m, path)
      assert r.path == path
      assert r.prediction.output.is_text
    end
  end

  describe "identify_stream/2" do
    test "reads content from an open device", %{magika: m} do
      path = Path.join(System.tmp_dir!(), "magika_stream_#{System.unique_integer([:positive])}")

      File.write!(
        path,
        "<!DOCTYPE html>\n<html><body><h1>Title</h1><p>Body text here.</p></body></html>"
      )

      on_exit(fn -> File.rm(path) end)

      {:ok, device} = File.open(path, [:read, :binary])
      {:ok, r} = Magika.identify_stream(m, device)
      File.close(device)

      assert r.prediction.output.label == "html"
    end
  end

  describe "prediction modes" do
    test "best_guess never generalizes due to low confidence" do
      best = Magika.new(prediction_mode: :best_guess)
      content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
      {:ok, rb} = Magika.identify(best, content)
      refute rb.prediction.overwrite_reason == :low_confidence
      assert rb.prediction.output.label == rb.prediction.dl.label
    end

    test "rejects an invalid prediction mode" do
      assert_raise ArgumentError, fn -> Magika.new(prediction_mode: :nope) end
    end
  end

  describe "parity with the reference Python implementation" do
    @fixtures_dir Path.join([__DIR__, "fixtures"])
    @external_resource Path.join(@fixtures_dir, "expected.json")
    @expected @external_resource |> File.read!() |> Jason.decode!()

    test "matches reference output_label, dl_label and score for every fixture", %{magika: m} do
      mismatches =
        for sample <- @expected, reduce: [] do
          acc ->
            path = Path.join([@fixtures_dir, "basic", sample["file"]])
            {:ok, r} = Magika.identify_path(m, path)
            p = r.prediction

            cond do
              p.output.label != sample["output_label"] ->
                [{sample["dir"], :output_label, p.output.label, sample["output_label"]} | acc]

              p.dl.label != sample["dl_label"] ->
                [{sample["dir"], :dl_label, p.dl.label, sample["dl_label"]} | acc]

              abs(p.score - sample["score"]) > 1.0e-4 ->
                [{sample["dir"], :score, p.score, sample["score"]} | acc]

              p.output.mime_type != sample["mime_type"] ->
                [{sample["dir"], :mime, p.output.mime_type, sample["mime_type"]} | acc]

              true ->
                acc
            end
        end

      assert mismatches == [], "parity mismatches:\n" <> inspect(mismatches, pretty: true)
    end
  end
end
