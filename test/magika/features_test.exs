defmodule Magika.FeaturesTest do
  use ExUnit.Case, async: true

  alias Magika.{Config, Features}

  setup_all do
    {:ok, config: Config.load()}
  end

  test "produces a vector of beg_size + end_size integers", %{config: c} do
    feats = Features.extract(String.duplicate("a", 5000), c)
    assert length(feats) == c.beg_size + c.end_size
    assert Enum.all?(feats, &(&1 in 0..256))
  end

  test "lstrips leading whitespace from the beginning bytes", %{config: c} do
    feats = Features.extract("    \n\t" <> "ABCDEFGH" <> String.duplicate("z", 5000), c)
    # First real byte after stripping should be ?A.
    assert hd(feats) == ?A
  end

  test "rstrips trailing whitespace from the end bytes", %{config: c} do
    content = String.duplicate("z", 5000) <> "WXYZ" <> "   \n\t"
    feats = Features.extract(content, c)
    # Last byte of the vector should be ?Z (whitespace stripped from the end).
    assert List.last(feats) == ?Z
  end

  test "pads short beginning content at the end with the padding token", %{config: c} do
    # 10 meaningful bytes; well below beg_size, so the beg block is padded.
    feats = Features.extract("ABCDEFGHIJ", c)
    beg = Enum.take(feats, c.beg_size)
    assert Enum.take(beg, 10) == ~c"ABCDEFGHIJ"
    assert Enum.at(beg, 10) == c.padding_token
    assert Enum.at(beg, c.beg_size - 1) == c.padding_token
  end

  test "pads short end content at the beginning with the padding token", %{config: c} do
    feats = Features.extract("ABCDEFGHIJ", c)
    last_end = Enum.drop(feats, c.beg_size)
    # End block is left-padded, so the real bytes land at the tail.
    assert List.last(last_end) == ?J
    assert hd(last_end) == c.padding_token
  end
end
