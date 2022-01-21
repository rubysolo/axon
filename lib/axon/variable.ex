defmodule Axon.Variable do
  @moduledoc false
  # Represents a variable which is aggregated internally
  # to the layer but is not considered "trainable" like
  # a parameter. Variables are updated given the aggregator.
  # For now, we only support EMA.
  defstruct [:id, :name, :shape, :initializer, :aggregator]
end
