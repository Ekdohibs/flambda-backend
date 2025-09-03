module Var : sig
  type t

  val create : unit -> t
end

type 'a pattern

module Pattern : sig
  type 'a t = 'a pattern

  val any : 'a t

  val var : Var.t -> 'a -> 'a t

  val untag : 'a t -> 'a t

  type 'a block_field

  val block_field :
    Target_ocaml_int.t -> Flambda_kind.t -> 'a t -> 'a block_field

  val block : ?tag:Tag.t -> 'a block_field list -> 'a t

  type 'a array_field

  val array_field :
    Target_ocaml_int.t -> Flambda_kind.t -> 'a t -> 'a array_field

  val array : 'a array_field list -> 'a t

  type 'a closure_field

  val value_slot : Value_slot.t -> 'a t -> 'a closure_field

  val function_slot : Function_slot.t -> 'a t -> 'a closure_field

  val closure : 'a closure_field list -> 'a t
end

type 'a expr

module Expr : sig
  type 'a t = 'a expr

  module Function_type : sig
    type 'a t

    val create : Code_id.t -> rec_info:'a -> 'a t
  end

  val unknown : Flambda_kind.t -> 'a t

  val unknown_with_subkind : Flambda_kind.With_subkind.t -> 'a t

  val tag_immediate : 'a -> 'a t

  val immutable_block :
    is_unique:bool ->
    Tag.t ->
    shape:Flambda_kind.Block_shape.t ->
    Alloc_mode.For_types.t ->
    fields:'a list ->
    'a t

  val exactly_this_closure :
    Function_slot.t ->
    all_function_slots_in_set:
      'a Function_type.t Or_unknown.t Function_slot.Map.t ->
    all_closure_types_in_set:'a Function_slot.Map.t ->
    all_value_slots_in_set:'a Value_slot.Map.t ->
    Alloc_mode.For_types.t ->
    'a t
end

type 'a rewrite

module Rule : sig
  type 'a t = 'a rewrite

  val identity : 'a -> 'a t

  val rewrite : 'a Pattern.t -> Var.t expr -> 'a t
end

module Make (X : sig
  type t

  module Map : Container_types.Map with type key = t

  val rewrite : t -> Typing_env.t -> Type_grammar.t -> t rewrite

  val block_slot :
    ?tag:Tag.t -> t -> Target_ocaml_int.t -> Typing_env.t -> Type_grammar.t -> t

  val array_slot :
    t -> Target_ocaml_int.t -> Typing_env.t -> Type_grammar.t -> t

  type set_of_closures

  val set_of_closures :
    t ->
    Function_slot.t ->
    Typing_env.t ->
    Type_grammar.closures_entry ->
    set_of_closures

  val rec_info :
    Typing_env.t ->
    set_of_closures ->
    Function_slot.t ->
    Code_id.t ->
    Type_grammar.t ->
    t

  val value_slot :
    set_of_closures -> Value_slot.t -> Typing_env.t -> Type_grammar.t -> t

  val function_slot :
    set_of_closures -> Function_slot.t -> Typing_env.t -> Type_grammar.t -> t
end) : sig
  (** Rewrite the provided typing env.

        The provided name map contains all the names that must be preserved,
        along with their abstraction. It must only contain symbols and variables
        with [Name_mode.normal] that exist in the provided typing env. *)
  val rewrite :
    Typing_env.t -> (X.t * Flambda_kind.t) Name.Map.t -> Typing_env.t
end
