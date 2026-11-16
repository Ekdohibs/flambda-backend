open BinInt
open BinNums
open Datatypes
open FlambdaBindingTime
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaProvers
open FlambdaTypeAccessSpec
open FlambdaTypeGrammar
open FlambdaTypingEnv
open List
open PeanoNat

module To_lift : sig
  type t =
    | Immutable_block of coq_Z * bool * scannable_block_shape * simple list
    | Boxed_number of boxable_number * coq_Z
end

module Reification_result : sig
  type t =
    | Lift of To_lift.t
    | Simple of simple
    | Cannot_reify
    | Invalid
end

module Reify : sig
  val reify_field :
    Typing_env.t ->
    (Alloc_mode.t -> variable -> bool option) ->
    Alloc_mode.t ->
    flambda_type * flambda_kind ->
    simple option option

  val reify_fields :
    Typing_env.t ->
    (Alloc_mode.t -> variable -> bool option) ->
    Alloc_mode.t ->
    (flambda_type * flambda_kind) list ->
    simple list option

  val try_to_reify_fields :
    Typing_env.t ->
    (Alloc_mode.t -> variable -> bool option) ->
    Alloc_mode.t ->
    (flambda_type * flambda_kind) list ->
    simple list option option

  val var_allowed :
    Typing_env.t ->
    (variable -> bool) ->
    (variable -> bool) ->
    Typing_env.t ->
    Name_mode.t ->
    Alloc_mode.t ->
    variable ->
    bool option

  val unboxed_number_kind : boxable_number -> naked_number_kind

  val reify_expanded :
    (Alloc_mode.t -> variable -> bool option) ->
    Typing_env.t ->
    Reification_result.t ->
    flambda_type ->
    Reification_result.t option

  val reify :
    Typing_env.t ->
    (variable -> bool) ->
    (variable -> bool) ->
    Typing_env.t ->
    flambda_type ->
    Reification_result.t option
end
