defmodule Magika.Inference do
  @moduledoc """
  Runs the ONNX model on an extracted feature vector and decodes the result.

  The model takes a `{batch, beg_size + end_size}` int32 tensor named `bytes`
  and returns a `{batch, n_labels}` float32 tensor of per-label probabilities
  (a softmax). We take the argmax to get the predicted label index and its
  score.
  """

  @doc """
  Predicts the content type label and score for a single feature vector.

  `features` is the list of integers produced by `Magika.Features.extract/2`,
  and `labels` is the ordered `target_labels_space` (index → label name).

  Returns `{label, score}`.
  """
  @spec predict(OnnxRuntime.Model.t(), [non_neg_integer()], [String.t()]) ::
          {String.t(), float()}
  def predict(model, features, labels) do
    [{label, score}] = predict_batch(model, [features], labels)
    {label, score}
  end

  @doc """
  Predicts for a batch of feature vectors at once.

  Returns a list of `{label, score}` tuples, one per input row, in order.
  """
  @spec predict_batch(OnnxRuntime.Model.t(), [[non_neg_integer()]], [String.t()]) ::
          [{String.t(), float()}]
  def predict_batch(_model, [], _labels), do: []

  def predict_batch(model, batch, labels) do
    rows = length(batch)
    cols = length(hd(batch))

    input =
      batch
      |> List.flatten()
      |> Nx.tensor(type: :s32)
      |> Nx.reshape({rows, cols})

    {output} = OnnxRuntime.run(model, input)

    # The output is backed by OnnxRuntime.Backend; move it to the default
    # backend before computing argmax / gathering scores.
    output = Nx.backend_transfer(output)

    indices = output |> Nx.argmax(axis: 1) |> Nx.to_flat_list()
    labels_by_index = List.to_tuple(labels)

    indices
    |> Enum.with_index()
    |> Enum.map(fn {label_idx, row} ->
      score = output[[row, label_idx]] |> Nx.to_number()
      {elem(labels_by_index, label_idx), score}
    end)
  end
end
