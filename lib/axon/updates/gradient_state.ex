defmodule Axon.Updates.GradientState do
  @moduledoc false

  # Update gradient state which is essentially a linked
  # list consisting of :state and :next_state. This simplifies
  # the recursion of the stateful/stateless combinators.
  @derive {
    Nx.Container,
    containers: [:state, :next_state]
  }
  defstruct state: %{}, next_state: {}
end