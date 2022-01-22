Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "sm-attention"}
])

defmodule Transformer do
  def model({inp, tar}, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size) do
    enc =
      inp
      |> encoder(num_layers, d_model, num_heads, dff, input_vocab_size)

    dec =
      tar
      |> decoder(enc, num_layers, d_model, num_heads, dff, target_vocab_size)

    dec
    |> Axon.dense(target_vocab_size)
    |> Axon.softmax()
  end

  def encoder(input, n, d_model, num_heads, dff, input_vocab_size) do
    embed = Axon.embedding(input, input_vocab_size, d_model)

    # TODO: Positional Encoding

    for _ <- 1..n, reduce: embed do
      x ->
        encoder_layer(x, d_model, num_heads, dff)
    end
  end

  def decoder(input, enc_input, n, d_model, num_heads, dff, target_vocab_size) do
    embed = Axon.embedding(input, target_vocab_size, d_model)

    # TODO: Positional Encoding

    for _ <- 1..n, reduce: embed do
      x ->
        decoder_layer(x, enc_input, d_model, num_heads, dff)
    end
  end

  def encoder_layer(input, d_model, num_heads, dff) do
    out1 =
      input
      |> Axon.self_attention(d_model, num_heads)
      |> Axon.add(input)
      |> Axon.layer_norm()

    ffn_out = point_wise_dense(out1, d_model, dff) |> Axon.dropout(rate: 0.3)

    out1
    |> Axon.add(ffn_out)
    |> Axon.layer_norm()
  end

  def decoder_layer(input, enc_input, d_model, num_heads, dff) do
    out1 =
      input
      |> Axon.self_attention(d_model, num_heads)
      |> Axon.add(input)
      |> Axon.layer_norm()

    out2 =
      enc_input
      |> Axon.self_attention(d_model, num_heads)
      |> Axon.add(out1)
      |> Axon.layer_norm()

    ffn_out = point_wise_dense(out2, d_model, dff) |> Axon.dropout(rate: 0.3)

    ffn_out
    |> Axon.add(out2)
    |> Axon.layer_norm()
  end

  defp point_wise_dense(input, d_model, dff) do
    input
    |> Axon.dense(dff)
    |> Axon.relu()
    |> Axon.dense(d_model)
  end
end
