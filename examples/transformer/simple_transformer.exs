Mix.install([
  {:axon, "~> 0.1.0-dev", path: "."}
])

defmodule Transformer do
  def model({inp, tar}, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_source, pe_target) do
    enc =
      inp
      |> encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_source)

    dec =
      tar
      |> decoder(enc, num_layers, d_model, num_heads, dff, target_vocab_size, pe_target)

    dec
    |> Axon.dense(target_vocab_size)
    |> Axon.softmax()
  end

  def encoder(input, n, d_model, num_heads, dff, input_vocab_size, max_positional_encoding) do
    embed = Axon.embedding(input, input_vocab_size, d_model)
    embed = Axon.nx(embed, fn x -> Nx.multiply(Nx.sqrt(x), d_model) end)
    pos_enc = positional_encoding(max_positional_encoding, d_model)
    pos_enc = Axon.reshape(pos_enc, {1, :auto, 1})
    embed = Axon.add(embed, pos_enc)

    # TODO: Padding Mask

    for _ <- 1..n, reduce: embed do
      x ->
        encoder_layer(x, d_model, num_heads, dff)
    end
  end

  def decoder(input, enc_input, n, d_model, num_heads, dff, target_vocab_size, max_positional_encoding) do
    embed = Axon.embedding(input, target_vocab_size, d_model)
    embed = Axon.nx(embed, fn x -> Nx.multiply(Nx.sqrt(x), d_model) end)
    pos_enc = positional_encoding(max_positional_encoding, d_model)
    # TODO: Add Axon.slice and slice to sequence length
    pos_enc = Axon.reshape(pos_enc, {1, :auto, d_model})
    embed = Axon.add(embed, pos_enc)

    # TODO: Look-ahead mask

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
    |> IO.inspect
    |> Axon.relu()
    |> Axon.dense(d_model)
  end

  defp positional_encoding(max_n, d_model) do
    fun = fn ->
      pos = Nx.iota({max_n}) |> Nx.new_axis(-1)
      i = Nx.iota({d_model}) |> Nx.new_axis(0)
      angle_rates = Nx.divide(1, Nx.power(10_000, Nx.divide(Nx.multiply(2, Nx.floor(Nx.divide(i, 2))), d_model)))
      enc = Nx.multiply(pos, angle_rates)

      counter = Nx.iota(enc, axis: 0)
      Nx.select(Nx.equal(Nx.remainder(counter, 2), 0), Nx.sin(angle_rates), Nx.cos(angle_rates))
    end

    val = Nx.Defn.jit(fun, [])
    Axon.constant(val)
  end
end
