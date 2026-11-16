(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* What a Flambda type means: which values of the machine have it.

   This file is to be reviewed: what the meaning reads of a value (the
   first section), [type_satisfied] and what it is made of, and the
   statements. The grammar is FlambdaTypeGrammar.v. The compiler has no
   such definition to compare with: what a type means there is what
   [Meet], [Join] and the provers ([Provers], [Expand_head]) take it to
   mean, and those are the files to read this against, with the creation
   functions of [Type_grammar] and [More_type_creators]. Every later
   statement about types (the environment, meet, join, the provers) is
   made with [type_satisfied], so a mistake here is a mistake in all of
   them.

   The compiler has two implementations of meet and join: [Meet_and_join]
   (meet_and_join.ml, with join_levels_old.ml), which is the one it runs
   unless told otherwise, at every optimisation level, and
   [Meet_and_n_way_join], which it runs for the n-way join algorithm
   and for match-in-match ([Meet.meet] chooses, by
   [Flambda_features.use_n_way_join]). A function named below is in
   both under that name unless one is said; where the two differ it is
   said.

   The environment. A type mentions names: the [Simple] of an alias, the
   variables that hold the results of [Is_int], [Get_tag] and [Is_null],
   the names an environment extension has equations on. So a value has a
   type relative to an environment, of which the meaning needs two
   things and no more:
   - the values of variables, [vars], an environment of the machine. It
     is used through [eval_simple] and [lookup] only;
   - the static environment, [statics]: what symbols are bound to. A
     symbol is a value by itself, a pointer, and a type says what it
     points to. It is used only through the observations of the first
     section.
   Nothing of the typing environment is needed (levels, binding times,
   canonical aliases): that is what is known, and this is what is true.
   What satisfies a typing environment is for the file that defines one
   to say, with [type_satisfied] for each name it has a type for. A
   variable that exists only in types has to have a value in [vars] all
   the same; to say that there is one is the typing environment's
   business.

   What the meaning is, in short ([type_satisfied]):
   - [Unknown] at a kind: every value of that kind. [Bottom]: none.
   - An alias [Equals s] at a kind: the value of [s], if of that kind.
     So a value has a type only if it has the kind of the type
     ([type_satisfied_kind_statement]).
   - A head of kind value: [Null], if [is_null] allows it; or a value
     that is not [Null] and has the [non_null] part.
   - [Variant]: a tagged immediate whose number has the type
     [immediates], or a block that the row-like structure [blocks]
     admits: the case of its tag, or failing one the case [other_tags],
     fixes its size, its shape and the types of its fields.
   - [Boxed_number]: a box of that number whose contents have the
     type given. [Closures]: a closure at a function slot whose case
     admits its set of closures. [String]: one of the strings.
     [Mutable_block]: nothing, see below.
   - A set of numbers: a number of the kind that is in the set. For a
     naked immediate, also the result of each relation listed, on the
     value of the name it is listed with ([relation_holds]).
   - An environment extension, where it is attached (a case of a
     row-like structure, the two extensions of a variant): each name has
     the type of its equation.

   Choices to check, each taken from what [Meet] does:
   - The heads of kind value are disjoint: a block, a boxed number, a
     closure and a string are four different things
     ([meet_head_of_kind_value_non_null]: "This assumes that all the
     different constructors are incompatible"). The observations of the
     machine are disjoint to match ([observations_disjoint_statement],
     in FlambdaValueSpec.v).
   - The variables [is_int], [get_tag] and [is_null] of a type and the
     relations of a naked immediate say the same thing from two sides,
     and the compiler records the two together
     ([Typing_env.add_is_null_relation] and the two beside it, in
     flambda2_types.ml). So they agree here: the variable of a type
     holds a number that the relation relates to each value of the type
     ([is_null_coherent_statement], [is_int_coherent_statement],
     [get_tag_coherent_statement]). The variables [is_int] and [is_null]
     say so of every value of the type, [get_tag] of its blocks only
     ([meet_variant] says why).
   - A product (the fields of a block, the closures and the value slots
     of a set of closures) says something of the components that exist
     and nothing of the others: an absent component is unknown, and what
     exists is for the index of the case to say ([Type_grammar]:
     "{ 1 => Unknown; 2 => V } is equal to { 2 => V }"). The function
     types of a set of closures are read in the same way: the code of a
     function is said if the set has the function. Both joins keep a
     function type only if every type joined has one
     ([join_closures_entry], and [n_way_join_closures_entry] since
     f489f9445e). Before that the n-way join kept a function type
     that only some of the types joined had, under an index that no
     longer said that the set had the function (discrepancy T1-4);
     the closure of a set without the function has the type it made
     then ([closure_absent_function_has] in the Proofs). The
     code of a function is known where the index says that the set has
     it ([closures_case_function_code_statement]), and always for the
     function at the slot of the closure itself
     ([closures_case_own_code_statement]): those are the two cases in
     which a prover reads a function type
     ([Row_like_for_closures.get_single_tag], [Provers.prove_code_ids]).
     Even read so, the n-way join was no upper bound then: where one
     type said nothing of the code of the function at the slot of the
     closure and another named it, the joined type named it
     ([nway_join_has_not] in the Proofs).
   - A relation is not the execution of its primitive
     ([relation_holds]). It relates a value and a number wherever the
     compiler records it of them, which is not where the primitive has
     a result:
     - [Is_null] relates every value of kind value: to 1 if it is
       [Null], to 0 if not, poison and a symbol that is not bound
       included. That is the compiler's reading of a constant
       ([Relation.of_const], by [Reg_width_const.is_null];
       [is_null_of_const_statement]), and the type of a scrutinee with
       a variable [is_null] admits all these. The primitive has no
       value on a symbol that is not bound, which no well-formed term
       uses, nor on poison.
     - [Is_int] relates a tagged immediate to 1 and a block to 0, and
       no other value: none that is poison (see below, on poison). The
       compiler records it for
       [Is_int { variant_only = true }]
       ([Simplify_unary_primitive.simplify_is_int]) and for a parameter
       that is unboxed as a variant, of a scrutinee that it takes to be
       a variant and not null ([add_is_int_relation]). Its reading of a
       constant is wider, 0 for [Null] and 1 for every other constant
       ([Relation.of_const]); the two agree where this one says
       something of a constant ([relation_of_const_sound_statement]).
     - [Get_tag] relates a block to its tag, if that is a tag (a number
       from 0 to 255, [is_tag]), and a tagged immediate to any number;
       it relates no value that is poison. The machine does not test
       the tag of a block that it makes; a well-formed term makes none
       out of range (decision D23 of the project), and [Meet] takes a
       number that is no tag to be the tag of no block
       ([Tag.create_from_targetint], in [reduce_inverse_relations]).
       Unboxing records the relation of a
       parameter that may be an immediate
       (simplify/unboxing/build_unboxing_denv.ml), and passes the tag 0
       on the paths where it is one (unboxing_epa.ml,
       [compute_extra_args_for_variant]); [add_get_tag_relation] keeps
       the immediates for that reason, and on a constant
       [reduce_head_of_kind_naked_immediate] drops the relation. The
       primitive has no value on an immediate, and nothing here gives
       it one.
     The result of a primitive, where it has one, is related, that of
     [Get_tag] if it is a tag ([primitive_related_statement]), and
     that is all that ties the two. Where a primitive has a result is
     not said here.
   - One reduction of [Meet] is not sound for this meaning, nor for any
     under which the environments that unboxing makes are true: from a
     relation [Get_tag] whose number is among some tags,
     [reduce_inverse_relations] makes the name a block with one of
     those tags, where it may be an immediate.
     [relation_get_tag_statement] has what does follow. Its other
     reductions are sound ([relation_is_null_statement],
     [relation_is_int_statement]).
   - Poison. A fact of the types is about a value that is not poison
     (decision D26 of the project). The compiler gives a poison
     constant the unknown type of its kind
     ([More_type_creators.type_for_const]), but unboxing says more of
     names that hold poison. For a field of a variant that is unboxed
     in turn, it passes poison on the paths where the argument has no
     such field, with 0 for the tag and for [Is_int] of it
     (unboxing_epa.ml); and the environment of the handler has the
     shape and the relations of that field on every path
     (build_unboxing_denv.ml, [denv_of_decision]). Such an environment
     is not true on those paths: not of the name that holds poison,
     nor of the names that hold the 0s. Here:
     - [Is_int] and [Get_tag] hold of no value that is poison
       ([relation_holds]): what [Meet] concludes from them, an
       immediate or a block, poison is not. So a variant, which is the
       type that has variables for the two, is no type of poison;
     - [Is_null] relates a poison of kind value to 0, as it does every
       value of that kind that is not [Null]: what [Meet] concludes
       from it, a value that is not [Null], poison is. So a type whose
       part that is not null admits poison (the unknown one) admits it
       still when it may be null, with a variable [is_null] that holds
       0 or with none; one whose part that is not null says what the
       value is admits no poison either way. The join relies on it: of
       a type of values that are not [Null] and one with such a
       variable it makes a type with a variable, which holds 0 where
       the first held;
     - what satisfies a typing environment is to set aside the names
       that hold poison and those whose type is a relation [Is_int] or
       [Get_tag] on one of them. That is for the file that defines one
       to say;
     - the types of a poison value are otherwise as they were. It has
       the types that say no more of a value than its kind, that it is
       this value, or that it is not [Null] (the unknown type, an alias
       to it, [any_non_null_value]), and none that says what it is: it
       is no immediate, no block, nothing allocated. The equations of
       an environment extension inside a type are of every value,
       poison included.
     The compiler relies on something that nothing in it checks: where
     a parameter holds poison, no use that reads it and no meet of its
     type to bottom is reachable. A statement that uses a fact about a
     name is to treat apart the case in which the name holds poison.
   - The code of a function type is the code of the closure, the very
     code id. The compiler means "that code or a newer version of it"
     ([Code_age_relation]); that is for the workstream on calls (S4) to
     put here, with a relation between code ids. Until then this is a
     restriction and not only something left out: a closure made with
     a newer version of the code has not, here, the type that names
     the older one, and the compiler takes it to have it
     ([meet_code_id] and the join of function types go by the
     relation; [Simplify_apply_expr] compares the code of the type
     with that of the call by it, and may make the call [Invalid]). So
     what is stated of meet, join and reification with this meaning is
     to leave out the types that name code, or to bring the relation
     in; and the code id that a prover answers is not to be read as
     the code of the closure at run time.
   - Allocation modes and [is_unique] mean nothing here: the machine has
     no regions, and uniqueness is about what may be duplicated. They
     matter to the compiler: a prover answers with the allocation mode
     of a type, or that a value holds nothing allocated locally
     ([Provers.never_holds_locally_allocated_values]); [Reify] lifts
     on the strength of that; and uniqueness is kept by meet and join
     and goes from a type to the block that reification makes
     ([Immutable_unique]). So what is stated with this meaning is
     about which values a type has. It says nothing of locality, of
     lifetime, of what may be duplicated or of whether lifting is
     safe: those need an invariant or a richer machine. In particular
     two values with equal contents are not thereby one symbol.

   What the meaning does not need and the operations will. [Meet] makes
   the meet of two products empty as soon as one component has no value
   ([meet_mapping]), and takes two shapes that differ to have no block
   in common ([meet_row_like_for_blocks]). Neither is true of the
   meaning as it stands, for every type of the grammar: an empty block
   has the type of the blocks of at least no field whose first field
   has no value, and a block of one field that is a value has both the
   shape [Value_only] and the shape [Mixed_record 1 []]
   ([wide_product_has] and [mixed_record_of_values_has] in the Proofs).
   They are true of the types in which:
   - the product of a case of blocks is no wider than the size of its
     index ([block_case_empty_statement]), and as wide if the size is
     exact ([Reify] asserts that);
   - the function types, the closure types and the value slot types of
     a case of closures are of slots among those of its index
     ([closures_case_function_code_statement],
     [closures_case_value_slot_statement]), and an exact index has the
     slot of the case among its closures;
   - a shape is written in one way: a mixed record has a flat suffix
     ([block_shapes_disjoint_statement]);
   - the kinds are right where the creation functions of the compiler
     check them (FlambdaTypeGrammar.v lists them).
   The creation functions of the compiler give such types if their
   callers give them what they expect, which they do not all check (a
   closure at a slot that its set has, closure types of slots of the
   set). Its raw constructors ([create_raw], which meet and join use)
   check nothing, and the n-way join did not keep the second
   condition before f489f9445e (see above, on products). The
   statements named say what
   each condition gives, for one case. FlambdaTypeWellFormedSpec.v has
   the conditions of a whole type as one test, [type_well_formed],
   with where the compiler establishes each, and the proof that the
   creation functions here give well-formed types. The meaning does
   not need it. The kinds of the fields of a block against its shape
   are not among the conditions: the compiler checks them only with
   its invariant checks on ([Row_like_for_blocks.check_field_tys]),
   and code whose kinds do not agree, which real programs have, makes
   types that break them.

   What an answer of a prover is about. The provers of the compiler come
   in two forms ([Provers.gen_value_to_proof] and [gen_value_to_meet]).
   A proof is about every value of the type. A meet shortcut is not: it
   leaves out the alternatives on which the operation asked about has
   no value (of a type that may be [Null] it answers for the part that
   is not), so its answer is about the values of the type on which
   that operation is defined, and is to be stated so.

   Where the machine is behind the grammar. The meaning is that of the
   machine of today (FlambdaSemanticsSpec.v), read through observations
   so that it changes with the machine and not with this file:
   - Blocks, boxed numbers and strings are read through the machine's
     [block_of], [boxed_number_of] and [string_of], and the kind of a
     value is the machine's [kind_of_value] (FlambdaValueSpec.v). A
     block or a boxed number is a value by itself or what a symbol is
     bound to, and what is said here of the one is said of the other. The
     two are different values all the same, and an alias tells them
     apart: a type [Equals s] is of the value of [s] and no other, so
     not of a symbol bound to a block with the tag and the fields of
     the block that [s] is. What is stated of lifting has to allow for
     that.
   - Closures are read through the machine's [closure_in_set] and
     [project_function_slot] (FlambdaValueSpec.v): a closure is the
     closure at a function slot of a set of closures, of which the
     machine has the code of each function slot and the values of the
     value slots. The closure of a set of one function whose slot has
     no name is read as the closure at [unnamed_function_slot] of a set
     with that one function. Nothing may rely on which slot the unnamed
     one is.
   - There are no mutable blocks, so [Mutable_block] has no value. With
     a heap (C4) it is to mean a mutable block, and a mutable block is
     then to be a block of [Variant] too, as [Meet] has it.
   - There is no value of kind [Region] or [Rec_info] but [Poison].
   - [Poison] is a value of its kind like another: at kind value it is
     not [Null], and it is no immediate and nothing allocated (see the
     choices above for the types it has). The contents of a boxed
     number may be poison; such a box has a type [Boxed_number] only if
     poison has the type of the contents.
   - The primitives are data (FlambdaPrimitiveSpec.v), and the relations
     are not defined with them. The result of [Is_null], of [Is_int]
     of a variant and of [Get_tag], where the primitive has one, is
     related to the value ([primitive_related_statement]). A constant
     is a tagged immediate here whatever its number; the primitives
     have a value only on one in the range of the machine. The two
     part only on a constant out of range, which no well-formed term
     is to have (decision D32, with D13): the difference goes when
     that is a condition of FlambdaWellFormedSpec.v.

   The statements say what each form of type means for the types built
   by the creation functions of the compiler, as an equivalence. The
   proofs are in FlambdaTypeMeaningProofs.v, which needs no review. It
   also has, for each constructor but [Mutable_block], a value that has
   a type built with it and a value that has not but would have one
   close to it; no value has a [Mutable_block] type. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar.
Import ListNotations.

(* ======================================================================== *)
(* What the meaning reads of a value                                        *)
(* ======================================================================== *)

(* The meaning reads a value through the observations of the machine
   (FlambdaValueSpec.v): [kind_of_value], [block_of], [boxed_number_of],
   [string_of], [closure_in_set] and [project_function_slot]. What is
   defined here is the number of a constant. *)

(* The number that a constant is, if it is one. *)
Definition number_of_const (c : const) : option Z :=
  match c with
  | Naked_immediate i | Tagged_immediate i | Naked_float32 i | Naked_float i
  | Naked_int8 i | Naked_int16 i | Naked_int32 i | Naked_int64 i
  | Naked_nativeint i | Naked_vec128 i | Naked_vec256 i | Naked_vec512 i
  | Naked_mask i => Some i
  | Null | Poison _ => None
  end.

(* ======================================================================== *)
(* The relations that types speak of                                        *)
(* ======================================================================== *)

Definition is_immediate (v : value) : Prop :=
  ∃ i : Z, v = Value_const (Tagged_immediate i).

Definition is_block (statics : static_env) (v : value) : Prop :=
  ∃ (tag : Z) (fields : list value), block_of statics v = Some (tag, fields).

(* A value is poison, of whatever kind. *)
Definition is_poison (v : value) : Prop :=
  ∃ k : flambda_kind, v = Value_const (Poison k).

(* A number is a tag: [Tag.create_from_targetint] has a tag for it. *)
Definition is_tag (n : Z) : Prop := (0 <= n <= 255)%Z.

(* What a type means by "this number is [Is_null], [Is_int] or [Get_tag]
   of that value". It is a relation of the types, not the execution of
   the primitive (see the header):
   - [Is_null]: the value is of kind value, and the number is 1 if it is
     [Null] and 0 if it is not. So it is 0 of a poison of kind value,
     and of a symbol that is not bound;
   - [Is_int]: the value is a tagged immediate and the number is 1, or
     it is a block and the number is 0;
   - [Get_tag]: the value is a block and the number is its tag, which
     is a tag; or the value is a tagged immediate, and then nothing is
     said of the number.
   [Is_int] and [Get_tag] hold of no value that is poison. *)
(* CR mshinwell: that [Is_int] and [Get_tag] hold of no poison is a
   decision of the model. It agrees with the machine, in which the two
   primitives have no value on poison, which is itself open (decision
   D8). It makes false the environments that unboxing makes, on the
   paths where a parameter holds poison: unboxing records these
   relations, with the number 0, of such parameters, and what satisfies
   an environment is to set those names aside. That is right only if no
   use that reads such a parameter, and no meet of its type to bottom,
   is reachable where it holds poison, and nothing in the compiler
   checks that. Decision D26. *)
(* CR mshinwell: [Is_null] relates a poison of kind value to 0. The
   relation is about whether the value is [Null], and a poison is not:
   that is the compiler's reading of a poison constant
   ([Reg_width_const.is_null]), and what its join of types takes for
   granted. The primitive is a use that reads the value, and has none
   on poison in the machine; whether it is to have one is open. A
   prover that finds the type [Null] for the name of such a relation
   answers 1 ([Provers.meet_naked_immediates]), which is not what
   this relation gives if the name holds poison and the environment
   says nothing of it: the provers are to be stated for environments
   that hold of every name (T1-9, T6-1). Decisions D8 and D26. *)
(* CR mshinwell: [Get_tag] relates a block to its tag only if the tag
   is from 0 to 255. The machine makes a block of any tag and does not
   test it; the terms of the compiler have tags in range, which is to
   be a condition of well-formed terms. The other choice is to relate
   every block to its tag, and to make "every block has a tag in
   range" a premise of what is stated of meet, of join and of the
   provers. Decision D23. *)
Definition relation_holds (statics : static_env) (relation : Relation.t)
    (v : value) (n : Z) : Prop :=
  match relation with
  | Relation.Is_null =>
    (v = Value_const Null ∧ n = 1%Z)
    ∨ (kind_of_value v = Value ∧ v <> Value_const Null ∧ n = 0%Z)
  | Relation.Is_int =>
    (is_immediate v ∧ n = 1%Z) ∨ (is_block statics v ∧ n = 0%Z)
  | Relation.Get_tag =>
    is_immediate v
    ∨ ((∃ fields : list value, block_of statics v = Some (n, fields))
       ∧ is_tag n)
  end.

(* The primitive of a relation: the one whose result the compiler
   records with it ([Simplify_unary_primitive.simplify_is_null],
   [simplify_is_int] with [variant_only], [simplify_get_tag]). *)
Definition primitive_of_relation (relation : Relation.t) : prim :=
  match relation with
  | Relation.Is_null => Unary Is_null
  | Relation.Is_int => Unary (Is_int true)
  | Relation.Get_tag => Unary Get_tag
  end.

(* The result of the primitive, where it has one, is related to its
   argument: that of [Get_tag] if it is a tag, which the machine does
   not test and the tag of a block of a well-formed term is (decision
   D23). *)
Definition primitive_related_statement : Prop :=
  ∀ (statics : static_env) (relation : Relation.t) (v : value) (n : Z),
    eval_prim statics (primitive_of_relation relation) [v]
    = Some (Value_const (Naked_immediate n)) ->
    (relation = Relation.Get_tag -> is_tag n) ->
    relation_holds statics relation v n.

(* ======================================================================== *)
(* Maps of types                                                            *)
(* ======================================================================== *)

(* The meaning of a type is defined by recursion on the type, and a type
   holds maps of types. Rocq accepts a recursive call on what a map
   binds only if the definition goes through the list that the map is.
   So the meaning is written with [every], [Keyed.for_all_bindings] and
   [Keyed.at_binding] of FlambdaMapSpec.v, which go through every pair
   of the list and ask something only of the pairs that the lookup
   finds, the bindings that count; its statements say so
   ([every_in_statement], [keyed_for_all_bindings_lookup_statement],
   [keyed_at_binding_lookup_statement]). On the maps keyed by function
   slots and value slots they are written [for_all_bindings] and
   [at_binding]; on those keyed by tags they take [Z.eqb], and on those
   keyed by the names of the types [name_eqb], which is a test of
   equality: *)
Definition name_eqb_decides_equality_statement : Prop :=
  decides_equality name_eqb.

(* ======================================================================== *)
(* The meaning of a type                                                    *)
(* ======================================================================== *)

(* A block has a shape: its tag is one that the shape permits, and its
   fields have the kinds of the shape. *)
Definition block_has_shape (shape : block_shape) (tag : Z)
    (fields : list value) : Prop :=
  match shape with
  | Scannable _ => (tag < no_scan_tag)%Z
  | Float_record => tag = double_array_tag
  end
  ∧ map kind_of_value fields = field_kinds shape (length fields).

(* The index of a case of blocks admits a block: its size, exactly or
   from below, and its shape. *)
Definition block_index_satisfied (tag : Z) (fields : list value)
    (index : row_like_index nat block_shape) : Prop :=
  match index_domain index with
  | Row_like_index_domain.Known size => length fields = size
  | Row_like_index_domain.At_least size => size <= length fields
  end
  ∧ block_has_shape (index_shape index) tag fields.

(* The index of a case of closures admits a set of closures: its
   function slots and its value slots, exactly or at least. *)
Definition closures_index_satisfied
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (index : row_like_index set_of_closures_contents unit) : Prop :=
  match index_domain index with
  | Row_like_index_domain.Known contents =>
    same_names (contents_closures contents) (domain functions)
    ∧ same_names (contents_value_slots contents) (domain value_slots)
  | Row_like_index_domain.At_least contents =>
    all_among (contents_closures contents) (domain functions) = true
    ∧ all_among (contents_value_slots contents) (domain value_slots) = true
  end.

Section Type_satisfied.

Variable statics : static_env.
Variable vars : env.

(* [Simple.name], evaluated. *)
Definition value_of_name (n : name) : option value :=
  eval_simple vars (simple_of_name n).

(* The variable of a type that stands for the result of a primitive on
   the value of the type, if the type has one, holds the number [n]. *)
Definition variable_is (x : option variable) (n : Z) : Prop :=
  match x with
  | Some x => lookup x vars = Some (Value_const (Naked_immediate n))
  | None => True
  end.

(* A type of kind [k] that is unknown, bottom, an alias, or given by a
   head, of which [head_satisfied] is the meaning. *)
Definition descr_satisfied {head : Type} (k : flambda_kind)
    (head_satisfied : head -> Prop) (v : value) (ty : type_descr head)
    : Prop :=
  match ty with
  | Unknown => kind_of_value v = k
  | Bottom => False
  | Ok (No_alias head) => head_satisfied head
  | Ok (Equals s) => eval_simple vars s = Some v ∧ kind_of_value v = k
  end.

(* A naked immediate: one of the numbers, if they are known, which is the
   result of each relation on the value of its name. *)
Definition head_of_kind_naked_immediate_satisfied (v : value)
    (head : head_of_kind_naked_immediate) : Prop :=
  ∃ i : Z,
    v = Value_const (Naked_immediate i)
    ∧ match naked_immediates head with
      | Or_unknown.Unknown => True
      | Or_unknown.Known numbers => i ∈ numbers
      end
    ∧ ∀ (relation : Relation.t) (n : name),
        (relation, n) ∈ inverse_relations head ->
        ∃ w : value,
          value_of_name n = Some w ∧ relation_holds statics relation w i.

(* A naked number of kind [k]: one of the numbers. *)
Definition naked_numbers_satisfied (k : naked_number_kind) (v : value)
    (numbers : list Z) : Prop :=
  ∃ (c : const) (n : Z),
    v = Value_const c
    ∧ kind_of_const c = Naked_number k
    ∧ number_of_const c = Some n
    ∧ n ∈ numbers.

(* The fields of a block have the types given, position by position, as
   far as both go. *)
Definition fields_satisfied (type_satisfied : value -> flambda_type -> Prop)
    : list flambda_type -> list value -> Prop :=
  fix fields_satisfied (tys : list flambda_type) (fields : list value)
      : Prop :=
    match tys, fields with
    | ty :: tys, field :: fields =>
      type_satisfied field ty ∧ fields_satisfied tys fields
    | _, _ => True
    end.

(* A case of blocks admits a block. The two arguments stand for the
   functions being defined below. *)
Definition block_case_satisfied
    (type_satisfied : value -> flambda_type -> Prop)
    (env_extension_satisfied : env_extension -> Prop) (tag : Z)
    (fields : list value) (case : row_like_block_case) : Prop :=
  block_index_satisfied tag fields (case_index case)
  ∧ fields_satisfied type_satisfied (case_maps_to case) fields
  ∧ env_extension_satisfied (case_env_extension case).

(* A case of closures admits the closure [v], which is at [slot] of a
   set of closures with [functions] and [value_slots]. The two
   arguments stand for the functions being defined below. *)
Definition closures_case_satisfied
    (closures_entry_satisfied : closures_entry -> Prop)
    (env_extension_satisfied : env_extension -> Prop)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (case : row_like_closures_case)
    : Prop :=
  closures_index_satisfied functions value_slots (case_index case)
  ∧ closures_entry_satisfied (case_maps_to case)
  ∧ env_extension_satisfied (case_env_extension case).

(* A value has a type. *)
(* CR mshinwell: a poison value has no type here that says what it is:
   no type of a variant, of a boxed number, of a closure or of a
   string. Unboxing records such types of parameters that hold poison
   on some paths, and what satisfies an environment is to set those
   names aside. The other way, that poison has every type of its kind,
   would let an alias put another value where poison was. Decision
   D26. *)
Fixpoint type_satisfied (v : value) (t : flambda_type) {struct t} : Prop :=
  match t with
  | Type_value ty =>
    descr_satisfied Value (head_of_kind_value_satisfied v) v ty
  | Type_naked_immediate ty =>
    descr_satisfied (Naked_number Kind_naked_immediate)
      (head_of_kind_naked_immediate_satisfied v) v ty
  | Type_naked_number k ty =>
    descr_satisfied (Naked_number k) (naked_numbers_satisfied k v) v ty
  | Type_rec_info ty =>
    descr_satisfied Rec_info (fun _ => kind_of_value v = Rec_info) v ty
  | Type_region ty =>
    descr_satisfied Region (fun _ => kind_of_value v = Region) v ty
  end

(* [Null], if it may be; or else a value that is not [Null] and has the
   non-null part. *)
with head_of_kind_value_satisfied (v : value) (head : head_of_kind_value)
    {struct head} : Prop :=
  match head with
  | Head_of_kind_value non_null is_null =>
    (v = Value_const Null
     ∧ match is_null with
       | Not_null => False
       | Maybe_null is_null => variable_is is_null 1
       end)
    ∨ (v <> Value_const Null
       ∧ match non_null with
         | Unknown => kind_of_value v = Value
         | Bottom => False
         | Ok head => head_of_kind_value_non_null_satisfied v head
         end
       ∧ match is_null with
         | Not_null => True
         | Maybe_null is_null => variable_is is_null 0
         end)
  end

with head_of_kind_value_non_null_satisfied (v : value)
    (head : head_of_kind_value_non_null) {struct head} : Prop :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    (∃ i : Z,
      v = Value_const (Tagged_immediate i)
      ∧ variable_is is_int 1
      ∧ match immediates with
        | Or_unknown.Unknown => True
        | Or_unknown.Known ty =>
          type_satisfied (Value_const (Naked_immediate i)) ty
        end
      ∧ match extensions with
        | No_extensions => True
        | Ext when_immediate _ => env_extension_satisfied when_immediate
        end)
    ∨ (∃ (tag : Z) (fields : list value),
        block_of statics v = Some (tag, fields)
        ∧ variable_is is_int 0
        ∧ variable_is get_tag tag
        ∧ match blocks with
          | Or_unknown.Unknown => True
          | Or_unknown.Known blocks =>
            row_like_for_blocks_satisfied tag fields blocks
          end
        ∧ match extensions with
          | No_extensions => True
          | Ext _ when_block => env_extension_satisfied when_block
          end)
  | Mutable_block _ => False
  | Boxed_number k contents _ =>
    ∃ c : const,
      boxed_number_of statics v = Some c
      ∧ kind_of_const c = unboxed_kind k
      ∧ type_satisfied (Value_const c) contents
  | Closures by_function_slot _ =>
    ∃ (slot : function_slot) (functions : list (function_slot * code_id))
      (value_slots : list (value_slot * value)),
      closure_in_set statics v = Some (slot, functions, value_slots)
      ∧ row_like_for_closures_satisfied v slot functions value_slots
          by_function_slot
  | String strs =>
    ∃ bytes : list Z, string_of statics v = Some bytes ∧ bytes ∈ strs
  end

(* The case of the tag of the block admits it; a tag whose case is
   unknown admits every block; a tag with no case goes by [other_tags]. *)
with row_like_for_blocks_satisfied (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks) {struct blocks} : Prop :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    Keyed.at_binding Z.eqb
      (fun case =>
         match case with
         | Or_unknown.Unknown => True
         | Or_unknown.Known case =>
           block_case_satisfied type_satisfied env_extension_satisfied tag
             fields case
         end)
      tag known_tags
    ∧ (Keyed.lookup Z.eqb tag known_tags = None ->
       match other_tags with
       | Or_bottom.Bottom => False
       | Or_bottom.Ok case =>
         block_case_satisfied type_satisfied env_extension_satisfied tag
           fields case
       end)
  end

(* There is a case for the function slot of the closure, and it admits
   it. *)
with row_like_for_closures_satisfied (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (closures : row_like_for_closures) {struct closures} : Prop :=
  match closures with
  | Row_like_for_closures known_closures =>
    is_bound slot known_closures = true
    ∧ at_binding
        (closures_case_satisfied
           (closures_entry_satisfied v slot functions value_slots)
           env_extension_satisfied functions value_slots)
        slot known_closures
  end

(* The functions of the set whose code is known have that code, and the
   closures of the set and the value slots have the types given: each
   if the set has it. What the set has is for the index of the case to
   say, not for the entry. *)
(* CR mshinwell: the code of a function type is the very code of the
   closure. The compiler means that code or a newer version of it
   ([Code_age_relation]), so a closure made with a newer version has
   not, here, the type that names the older one. It is to be put right
   with the calls (workstream S4); until then what is stated of meet,
   join and the provers leaves out the types that name code. No
   decision has a number. *)
(* CR mshinwell: a closure type of an entry asks its type of the
   closure projected at its slot only where the projection has a
   value. From a symbol the projection reads the symbol that the
   static set names for the slot, so where what a symbol is bound to
   is not valid (FlambdaClosuresSpec.statics_valid) a set can have the
   function and name no symbol for it, and the closure type says
   nothing. The meet of two entries takes a closure type with no
   value to leave no closure; that is right only where the static
   environment is valid, and what is stated of the meet supposes it
   ([well_formed_closures_case_closure_type_statement], in
   FlambdaTypeWellFormedSpec.v, is the step). The other choice is to
   ask here that the projection have a value. An entry would then say
   that its set has the function, which is for the index to say, and
   what is stated of the creation functions, of the accessors and of
   the provers would change. Found in the review of the meet. No
   decision has a number. *)
with closures_entry_satisfied (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (entry : closures_entry)
    {struct entry} : Prop :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    (∀ (g : function_slot) (function_type : function_type) (code : code_id),
      lookup g function_types = Some (Or_unknown.Known function_type) ->
      lookup g functions = Some code ->
      code = function_type_code_id function_type)
    ∧ for_all_bindings
        (fun g ty =>
           ∀ closure : value,
             project_function_slot statics slot g v = Some closure ->
             type_satisfied closure ty)
        closure_types
    ∧ for_all_bindings
        (fun w ty =>
           ∀ contents : value,
             lookup w value_slots = Some contents ->
             type_satisfied contents ty)
        value_slot_types
  end

(* Every name with an equation has a value, of the type of the
   equation. *)
(* CR mshinwell: an equation asks that its name have a value. A typing
   environment asks the type of a name only if the name has a value,
   since a variable that a join made existential stands for nothing on
   some runs. So an extension made of a level of an environment holds
   only if the names of the level have values: the meet has to show
   that of the levels it cuts, and the n-way join of the compiler does
   not ensure it of a variable that it brings in from one of the
   environments joined (T5-5, T5-10). The other reading, that an
   equation asks nothing of a name with no value, is that of an
   environment; with it the meet would be wrong where it adds the
   extensions that two inputs have for one case and drops the case if
   they do not meet, since nothing would tie the two types of a name
   with no value. No decision has a number. *)
with env_extension_satisfied (extension : env_extension) {struct extension}
    : Prop :=
  match extension with
  | Env_extension equations =>
    Keyed.for_all_bindings name_eqb
      (fun n ty =>
         ∃ v : value, value_of_name n = Some v ∧ type_satisfied v ty)
      equations
  end.

End Type_satisfied.

(* A case of blocks admits a block, and a case of closures a closure. *)
Definition row_like_block_case_satisfied (statics : static_env) (vars : env)
    (tag : Z) (fields : list value) (case : row_like_block_case) : Prop :=
  block_case_satisfied (type_satisfied statics vars)
    (env_extension_satisfied statics vars) tag fields case.

Definition row_like_closures_case_satisfied (statics : static_env)
    (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (case : row_like_closures_case)
    : Prop :=
  closures_case_satisfied
    (closures_entry_satisfied statics vars v slot functions value_slots)
    (env_extension_satisfied statics vars) functions value_slots case.

(* ======================================================================== *)
(* What every kind has: unknown, bottom, aliases                            *)
(* ======================================================================== *)

(* A value has a type only if it has the kind of the type. *)
Definition type_satisfied_kind_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    type_satisfied statics vars v t -> kind_of_value v = kind t.

Definition unknown_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : flambda_kind),
    type_satisfied statics vars v (unknown k) <-> kind_of_value v = k.

Definition bottom_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : flambda_kind),
    ~ type_satisfied statics vars v (bottom k).

Definition alias_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : flambda_kind)
    (s : simple),
    type_satisfied statics vars v (alias_type_of k s)
    <-> eval_simple vars s = Some v ∧ kind_of_value v = k.

(* The same of any type that is an alias. *)
Definition get_alias_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type)
    (s : simple),
    get_alias_opt t = Some s ->
    (type_satisfied statics vars v t
     <-> eval_simple vars s = Some v ∧ kind_of_value v = kind t).

Definition this_const_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (c : const),
    type_satisfied statics vars v (this_const c) <-> v = Value_const c.

(* A constant has the type that the compiler gives it. *)
Definition type_for_const_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (c : const),
    type_satisfied statics vars (Value_const c) (type_for_const c).

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

Definition these_naked_immediates_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (imms : list Z),
    type_satisfied statics vars v (these_naked_immediates imms)
    <-> ∃ i : Z, v = Value_const (Naked_immediate i) ∧ i ∈ imms.

(* A set of naked numbers of another kind. *)
Definition type_naked_number_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : naked_number_kind)
    (numbers : list Z),
    type_satisfied statics vars v
      (Type_naked_number k (Ok (No_alias numbers)))
    <-> ∃ (c : const) (n : Z),
          v = Value_const c
          ∧ kind_of_const c = Naked_number k
          ∧ number_of_const c = Some n
          ∧ n ∈ numbers.

(* A number that a relation relates to the value of a name. The three
   statements spell [relation_holds] out. *)
Definition is_int_for_scrutinee_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (scrutinee : name),
    type_satisfied statics vars v (is_int_for_scrutinee scrutinee)
    <-> ∃ (i : Z) (w : value),
          v = Value_const (Naked_immediate i)
          ∧ value_of_name vars scrutinee = Some w
          ∧ ((is_immediate w ∧ i = 1%Z) ∨ (is_block statics w ∧ i = 0%Z)).

(* The scrutinee may be an immediate, as a parameter that is unboxed
   may be; the number is then any number. *)
Definition get_tag_for_block_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (block : name),
    type_satisfied statics vars v (get_tag_for_block block)
    <-> ∃ (i : Z) (w : value),
          v = Value_const (Naked_immediate i)
          ∧ value_of_name vars block = Some w
          ∧ (is_immediate w
             ∨ ((∃ fields : list value,
                  block_of statics w = Some (i, fields))
                ∧ is_tag i)).

Definition is_null_for_scrutinee_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (scrutinee : name),
    type_satisfied statics vars v (is_null_for_scrutinee scrutinee)
    <-> ∃ (i : Z) (w : value),
          v = Value_const (Naked_immediate i)
          ∧ value_of_name vars scrutinee = Some w
          ∧ ((w = Value_const Null ∧ i = 1%Z)
             ∨ (kind_of_value w = Value ∧ w <> Value_const Null
                ∧ i = 0%Z)).

(* The two sides of a relation agree: the variable of a type holds a
   number that the relation relates to each value of the type. For
   [Get_tag] the variable need hold nothing where the value is an
   immediate, and the number is to be a tag: a variant with a variable
   [get_tag] is still a type of a block whose tag is out of range. *)
Definition is_null_coherent_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : variable),
    head_of_kind_value_satisfied statics vars v
      (Head_of_kind_value non_null (Maybe_null (Some is_null))) ->
    ∃ n : Z,
      lookup is_null vars = Some (Value_const (Naked_immediate n))
      ∧ relation_holds statics Relation.Is_null v n.

Definition is_int_coherent_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (is_int : variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool),
    head_of_kind_value_non_null_satisfied statics vars v
      (Variant (Some is_int) immediates get_tag blocks extensions is_unique) ->
    ∃ n : Z,
      lookup is_int vars = Some (Value_const (Naked_immediate n))
      ∧ relation_holds statics Relation.Is_int v n.

Definition get_tag_coherent_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) (n : Z),
    head_of_kind_value_non_null_satisfied statics vars v
      (Variant is_int immediates (Some get_tag) blocks extensions is_unique) ->
    lookup get_tag vars = Some (Value_const (Naked_immediate n)) ->
    is_tag n ->
    relation_holds statics Relation.Get_tag v n.

(* The compiler's reading of a relation on a constant
   ([Relation.of_const], which [reduce_head_of_kind_naked_immediate]
   uses) is sound: if a relation relates a constant to a number, and the
   compiler gives a number for it, it is that one. For [Is_null] it is
   exact on the constants of kind value, poison included. For [Is_int]
   it is not: the compiler gives 0 for [Null] and 1 for every other
   constant, poison and the naked numbers included, and the relation
   holds of the tagged immediates only. *)
Definition relation_of_const_sound_statement : Prop :=
  ∀ (statics : static_env) (relation : Relation.t) (c : const) (i n : Z),
    relation_holds statics relation (Value_const c) i ->
    Relation.of_const relation c = Or_bottom.Ok n ->
    i = n.

Definition is_null_of_const_statement : Prop :=
  ∀ (statics : static_env) (c : const) (n : Z),
    kind_of_const c = Value ->
    (relation_holds statics Relation.Is_null (Value_const c) n
     <-> Relation.of_const Relation.Is_null c = Or_bottom.Ok n).

(* ======================================================================== *)
(* Null, immediates and blocks                                              *)
(* ======================================================================== *)

Definition null_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value),
    type_satisfied statics vars v null <-> v = Value_const Null.

Definition any_non_null_value_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value),
    type_satisfied statics vars v any_non_null_value
    <-> kind_of_value v = Value ∧ v <> Value_const Null.

(* A variant with no variables for [Is_int] and [Get_tag]: a tagged
   immediate or a block. *)
Definition create_variant_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (is_unique : bool)
    (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions),
    type_satisfied statics vars v
      (create_variant is_unique immediates blocks extensions)
    <-> (∃ i : Z,
          v = Value_const (Tagged_immediate i)
          ∧ match immediates with
            | Or_unknown.Unknown => True
            | Or_unknown.Known ty =>
              type_satisfied statics vars (Value_const (Naked_immediate i)) ty
            end
          ∧ match extensions with
            | No_extensions => True
            | Ext when_immediate _ =>
              env_extension_satisfied statics vars when_immediate
            end)
        ∨ (∃ (tag : Z) (fields : list value),
            block_of statics v = Some (tag, fields)
            ∧ match blocks with
              | Or_unknown.Unknown => True
              | Or_unknown.Known blocks =>
                row_like_for_blocks_satisfied statics vars tag fields blocks
              end
            ∧ match extensions with
              | No_extensions => True
              | Ext _ when_block =>
                env_extension_satisfied statics vars when_block
              end).

Definition any_tagged_immediate_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value),
    type_satisfied statics vars v any_tagged_immediate
    <-> is_immediate v.

Definition tag_immediate_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    type_satisfied statics vars v (tag_immediate t)
    <-> ∃ i : Z,
          v = Value_const (Tagged_immediate i)
          ∧ type_satisfied statics vars (Value_const (Naked_immediate i)) t.

Definition these_tagged_immediates_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (imms : list Z),
    type_satisfied statics vars v (these_tagged_immediates imms)
    <-> ∃ i : Z, v = Value_const (Tagged_immediate i) ∧ i ∈ imms.

Definition any_block_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value),
    type_satisfied statics vars v any_block <-> is_block statics v.

Definition blocks_with_these_tags_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (tags : list Z)
    (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v (blocks_with_these_tags tags alloc_mode)
    <-> ∃ (tag : Z) (fields : list value),
          block_of statics v = Some (tag, fields) ∧ tag ∈ tags.

(* The fields of a block, by position. *)
Definition fields_satisfied_nth_statement : Prop :=
  ∀ (P : value -> flambda_type -> Prop) (tys : list flambda_type)
    (fields : list value),
    fields_satisfied P tys fields
    <-> (∀ (i : nat) (ty : flambda_type) (field : value),
          nth_error tys i = Some ty ->
          nth_error fields i = Some field ->
          P field ty).

(* A row-like structure of blocks, by the case of the tag. *)
Definition row_like_for_blocks_find_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t),
    row_like_for_blocks_satisfied statics vars tag fields
      (Row_like_for_blocks known_tags other_tags alloc_mode)
    <-> match Keyed.lookup Z.eqb tag known_tags with
        | Some (Or_unknown.Known case) =>
          row_like_block_case_satisfied statics vars tag fields case
        | Some Or_unknown.Unknown => True
        | None =>
          match other_tags with
          | Or_bottom.Ok case =>
            row_like_block_case_satisfied statics vars tag fields case
          | Or_bottom.Bottom => False
          end
        end.

(* A block of a tag whose fields, as many as there are types, have the
   types. *)
Definition immutable_block_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (is_unique : bool)
    (tag : Z) (shape : block_shape) (alloc_mode : Alloc_mode.t)
    (field_tys : list flambda_type),
    type_satisfied statics vars v
      (immutable_block is_unique tag shape alloc_mode field_tys)
    <-> ∃ fields : list value,
          block_of statics v = Some (tag, fields)
          ∧ block_has_shape shape tag fields
          ∧ Forall2 (type_satisfied statics vars) fields field_tys.

(* The machine has no mutable block. *)
Definition mutable_block_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (alloc_mode : Alloc_mode.t),
    ~ type_satisfied statics vars v (mutable_block alloc_mode).

(* ======================================================================== *)
(* Boxed numbers, strings and closures                                      *)
(* ======================================================================== *)

Definition box_number_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v (box_number k contents alloc_mode)
    <-> ∃ c : const,
          boxed_number_of statics v = Some c
          ∧ kind_of_const c = unboxed_kind k
          ∧ type_satisfied statics vars (Value_const c) contents.

Definition this_immutable_string_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (bytes : list Z),
    type_satisfied statics vars v (this_immutable_string bytes)
    <-> string_of statics v = Some bytes.

(* A row-like structure of closures, by the case of the function slot. *)
Definition row_like_for_closures_lookup_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (known_closures : finite_map row_like_closures_case),
    row_like_for_closures_satisfied statics vars v slot functions
      value_slots (Row_like_for_closures known_closures)
    <-> ∃ case : row_like_closures_case,
          lookup slot known_closures = Some case
          ∧ row_like_closures_case_satisfied statics vars v slot functions
              value_slots case.

(* What is said of the functions, the closures and the value slots of a
   set of closures. *)
Definition closures_entry_lookup_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type),
    closures_entry_satisfied statics vars v slot functions value_slots
      (Closures_entry function_types closure_types value_slot_types)
    <-> (∀ (g : function_slot) (function_type : function_type)
           (code : code_id),
          lookup g function_types = Some (Or_unknown.Known function_type) ->
          lookup g functions = Some code ->
          code = function_type_code_id function_type)
        ∧ (∀ (g : function_slot) (ty : flambda_type) (closure : value),
            lookup g closure_types = Some ty ->
            project_function_slot statics slot g v = Some closure ->
            type_satisfied statics vars closure ty)
        ∧ (∀ (w : value_slot) (ty : flambda_type) (contents : value),
            lookup w value_slot_types = Some ty ->
            lookup w value_slots = Some contents ->
            type_satisfied statics vars contents ty).

(* What a set of closures has for sure, by the index of its case. *)
Definition closures_index_contents
    (index : row_like_index set_of_closures_contents unit)
    : set_of_closures_contents :=
  match index_domain index with
  | Row_like_index_domain.Known contents
  | Row_like_index_domain.At_least contents => contents
  end.

(* A function that the index of a case says the set has, and whose code
   the entry knows, is there with that code. It is what a prover reads:
   [Row_like_for_closures.get_single_tag] asks that the slot be among
   the closures of the index before the function type is looked at. *)
Definition closures_case_function_code_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (index : row_like_index set_of_closures_contents unit)
    (extension : env_extension) (g : function_slot)
    (function_type : function_type),
    row_like_closures_case_satisfied statics vars v slot functions
      value_slots
      (Row_like_case
         (Closures_entry function_types closure_types value_slot_types)
         index extension) ->
    among g (contents_closures (closures_index_contents index)) = true ->
    lookup g function_types = Some (Or_unknown.Known function_type) ->
    lookup g functions = Some (function_type_code_id function_type).

(* The same of the function at the slot of the closure itself, which
   the set has whatever the index says: [Provers.prove_code_ids] reads
   its function type without looking at the index. *)
Definition closures_case_own_code_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (index : row_like_index set_of_closures_contents unit)
    (extension : env_extension) (function_type : function_type),
    closure_in_set statics v = Some (slot, functions, value_slots) ->
    row_like_closures_case_satisfied statics vars v slot functions
      value_slots
      (Row_like_case
         (Closures_entry function_types closure_types value_slot_types)
         index extension) ->
    lookup slot function_types = Some (Or_unknown.Known function_type) ->
    lookup slot functions = Some (function_type_code_id function_type).

(* The closure at a function slot of a set with exactly the functions
   and the value slots given, each as its type says. *)
Definition exactly_this_closure_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v
      (exactly_this_closure slot function_types closure_types
         value_slot_types alloc_mode)
    <-> ∃ (functions : list (function_slot * code_id))
          (value_slots : list (value_slot * value)),
          closure_in_set statics v = Some (slot, functions, value_slots)
          ∧ same_names (domain function_types) (domain functions)
          ∧ same_names (domain value_slot_types) (domain value_slots)
          ∧ closures_entry_satisfied statics vars v slot functions
              value_slots
              (Closures_entry function_types closure_types
                 value_slot_types).

(* The same of a set with at least the functions and the value slots
   given. *)
Definition at_least_this_closure_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v
      (at_least_this_closure slot function_types closure_types
         value_slot_types alloc_mode)
    <-> ∃ (functions : list (function_slot * code_id))
          (value_slots : list (value_slot * value)),
          closure_in_set statics v = Some (slot, functions, value_slots)
          ∧ all_among (domain function_types) (domain functions) = true
          ∧ all_among (domain value_slot_types) (domain value_slots) = true
          ∧ closures_entry_satisfied statics vars v slot functions
              value_slots
              (Closures_entry function_types closure_types
                 value_slot_types).

(* ======================================================================== *)
(* Environment extensions                                                   *)
(* ======================================================================== *)

Definition env_extension_lookup_statement : Prop :=
  ∀ (statics : static_env) (vars : env)
    (equations : Keyed.finite_map name flambda_type),
    env_extension_satisfied statics vars (Env_extension equations)
    <-> (∀ (n : name) (ty : flambda_type),
          Keyed.lookup name_eqb n equations = Some ty ->
          ∃ v : value,
            value_of_name vars n = Some v ∧ type_satisfied statics vars v ty).

(* ======================================================================== *)
(* What a relation says of the value it relates                             *)
(* ======================================================================== *)

(* What follows for a value from the number that a relation relates it
   to. For [Is_null] and [Is_int] it is what [Meet] deduces
   ([reduce_inverse_relations]). For [Get_tag] it is less: [Meet] makes
   the value a block with one of the numbers that are tags, and it may
   be an immediate (see the header). *)
Definition relation_is_null_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (w : value) (n : Z),
    relation_holds statics Relation.Is_null w n ->
    (n = 1%Z ∧ type_satisfied statics vars w null)
    ∨ (n = 0%Z ∧ type_satisfied statics vars w any_non_null_value).

Definition relation_is_int_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (w : value) (n : Z),
    relation_holds statics Relation.Is_int w n ->
    (n = 1%Z ∧ type_satisfied statics vars w any_tagged_immediate)
    ∨ (n = 0%Z ∧ type_satisfied statics vars w any_block).

(* CR mshinwell: this is less than [reduce_inverse_relations] concludes,
   in both implementations of meet: it makes the value a block with one
   of the tags. Here the value may be an immediate, because unboxing
   records the relation, with the tag 0, of a parameter that is one on
   some paths. Either the reduction is wrong where such a relation
   reaches it, or something that is not written down keeps it from
   doing so. No decision has a number; discrepancies T1-1 and R3-2. *)
Definition relation_get_tag_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (w : value) (n : Z)
    (tags : list Z) (alloc_mode : Alloc_mode.t),
    relation_holds statics Relation.Get_tag w n ->
    is_immediate w
    ∨ (is_tag n
       ∧ (n ∈ tags ->
          type_satisfied statics vars w
            (blocks_with_these_tags tags alloc_mode))).

(* ======================================================================== *)
(* What the meaning of a type depends on                                    *)
(* ======================================================================== *)

(* A type keeps its values when the static environment grows, as it does
   in a run (FlambdaWellFormedSpec.run_extends_statics_statement): what is
   true of a value where a handler is defined is true where it runs. *)
Definition type_satisfied_statics_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (vars : env) (v : value)
    (t : flambda_type),
    statics_extend statics statics' ->
    type_satisfied statics vars v t ->
    type_satisfied statics' vars v t.

Definition env_extension_satisfied_statics_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (vars : env) (extension : env_extension),
    statics_extend statics statics' ->
    env_extension_satisfied statics vars extension ->
    env_extension_satisfied statics' vars extension.

(* The meaning of a type reads the environment only to find the value
   of a variable that the type mentions, and never asks that a variable
   have none. So a type keeps its values when the environment gives a
   value to more variables and keeps the values of those that had one:
   what a join adds for a variable that only some of its inputs
   define takes nothing from what was known. *)
Definition type_satisfied_monotone_statement : Prop :=
  ∀ (statics : static_env) (vars vars' : env) (v : value) (t : flambda_type),
    (∀ (x : variable) (w : value),
      x ∈ free_variables t ->
      lookup x vars = Some w -> lookup x vars' = Some w) ->
    type_satisfied statics vars v t -> type_satisfied statics vars' v t.

Definition env_extension_satisfied_monotone_statement : Prop :=
  ∀ (statics : static_env) (vars vars' : env) (extension : env_extension),
    (∀ (x : variable) (w : value),
      x ∈ free_variables_of_env_extension extension ->
      lookup x vars = Some w -> lookup x vars' = Some w) ->
    env_extension_satisfied statics vars extension ->
    env_extension_satisfied statics vars' extension.

(* So the meaning of a type depends on the environment only through
   the values of the variables that the type mentions. *)
Definition type_satisfied_coincide_statement : Prop :=
  ∀ (statics : static_env) (vars vars' : env) (v : value) (t : flambda_type),
    (∀ x : variable, x ∈ free_variables t -> lookup x vars = lookup x vars') ->
    (type_satisfied statics vars v t <-> type_satisfied statics vars' v t).

Definition env_extension_satisfied_coincide_statement : Prop :=
  ∀ (statics : static_env) (vars vars' : env) (extension : env_extension),
    (∀ x : variable,
      x ∈ free_variables_of_env_extension extension ->
      lookup x vars = lookup x vars') ->
    (env_extension_satisfied statics vars extension
     <-> env_extension_satisfied statics vars' extension).

(* So binding a variable that a type does not mention changes nothing:
   what is known stays known under a binder of a fresh name. *)
Definition type_satisfied_bind_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (x : variable) (w v : value)
    (t : flambda_type),
    x ∉ free_variables t ->
    (type_satisfied statics ((x, w) :: vars) v t
     <-> type_satisfied statics vars v t).

(* The unknown type of a kind is the top of the kind. *)
Definition unknown_is_top_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    type_satisfied statics vars v t ->
    type_satisfied statics vars v (unknown (kind t)).

(* ======================================================================== *)
(* Types with no value, and heads with no value in common                   *)
(* ======================================================================== *)

Definition is_obviously_bottom_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    is_obviously_bottom t = true -> ~ type_satisfied statics vars v t.

Definition is_obviously_unknown_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    is_obviously_unknown t = true ->
    (type_satisfied statics vars v t <-> kind_of_value v = kind t).

Definition row_like_for_blocks_is_bottom_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks),
    row_like_for_blocks_is_bottom blocks = true ->
    ~ row_like_for_blocks_satisfied statics vars tag fields blocks.

(* The last case of [Meet.meet_head_of_kind_value_non_null]: two heads
   made with different constructors have no value in common, but for a
   variant and a mutable block; nor have boxed numbers of two kinds. *)
Definition heads_compatible (head head' : head_of_kind_value_non_null)
    : Prop :=
  match head, head' with
  | (Variant _ _ _ _ _ _ | Mutable_block _),
    (Variant _ _ _ _ _ _ | Mutable_block _) => True
  | Boxed_number k _ _, Boxed_number k' _ _ => k = k'
  | Closures _ _, Closures _ _ => True
  | String _, String _ => True
  | _, _ => False
  end.

Definition heads_compatible_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (head head' : head_of_kind_value_non_null),
    head_of_kind_value_non_null_satisfied statics vars v head ->
    head_of_kind_value_non_null_satisfied statics vars v head' ->
    heads_compatible head head'.

(* ======================================================================== *)
(* What the operations on types rely on                                     *)
(* ======================================================================== *)

(* [Meet] makes a product with a component of no value empty, and takes
   two shapes that differ to have no block in common (see the header).
   The statements below say on which conditions that is so, case by
   case. The conditions are not part of the meaning. *)

(* The size that the index of a case of blocks gives, exactly or from
   below. *)
Definition block_index_size (index : row_like_index nat block_shape) : nat :=
  match index_domain index with
  | Row_like_index_domain.Known size
  | Row_like_index_domain.At_least size => size
  end.

(* A case of blocks whose product is no wider than the size of its
   index admits no block if one of the types of the product has no
   value: every type of the product is the type of a field. *)
Definition block_case_empty_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (case : row_like_block_case) (ty : flambda_type),
    length (case_maps_to case) <= block_index_size (case_index case) ->
    ty ∈ case_maps_to case ->
    (∀ v : value, ~ type_satisfied statics vars v ty) ->
    ~ row_like_block_case_satisfied statics vars tag fields case.

(* A value slot that the index of a case of closures says the set has,
   and of which the entry has a type, is there with contents of that
   type. So the case admits no closure if that type has no value. *)
Definition closures_case_value_slot_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (index : row_like_index set_of_closures_contents unit)
    (extension : env_extension) (w : value_slot) (ty : flambda_type),
    row_like_closures_case_satisfied statics vars v slot functions
      value_slots
      (Row_like_case
         (Closures_entry function_types closure_types value_slot_types)
         index extension) ->
    among w (contents_value_slots (closures_index_contents index)) = true ->
    lookup w value_slot_types = Some ty ->
    ∃ contents : value,
      lookup w value_slots = Some contents
      ∧ type_satisfied statics vars contents ty.

(* A shape as the compiler writes it: a mixed record has a flat suffix,
   and is [Value_only] otherwise. *)
Definition block_shape_canonical (shape : block_shape) : Prop :=
  match shape with
  | Scannable (Mixed_record _ flat_suffix) => flat_suffix <> []
  | Scannable Value_only | Float_record => True
  end.

(* Two such shapes that a block both has are one shape. *)
Definition block_shapes_disjoint_statement : Prop :=
  ∀ (shape shape' : block_shape) (tag : Z) (fields : list value),
    block_shape_canonical shape ->
    block_shape_canonical shape' ->
    block_has_shape shape tag fields ->
    block_has_shape shape' tag fields ->
    shape = shape'.

(* ======================================================================== *)
(* What meet and the joins read of a type                                   *)
(* ======================================================================== *)

(* The constant that [must_be_singleton] answers is the only value of
   the type. The relations of a naked immediate are not looked at, and
   need not be: they take values away. *)
Definition must_be_singleton_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type)
    (c : const),
    must_be_singleton t = Some (Some c) ->
    type_satisfied statics vars v t ->
    v = Value_const c.

(* The two heads that meet makes are those of [null] and of the type
   of one tagged immediate. *)
Definition heads_of_meet_statement : Prop :=
  Type_value (Ok (No_alias head_of_kind_value_null)) = null
  ∧ ∀ imm : Z,
      Type_value (Ok (No_alias (head_of_kind_value_tagged_immediate imm)))
      = tag_immediate (this_naked_immediate imm).

(* The field of a block of a shape, at an index, is of the kind that
   the shape gives the index. *)
Definition block_shape_element_kind_field_statement : Prop :=
  ∀ (shape : block_shape) (tag : Z) (fields : list value) (index : nat)
    (field : value) (k : flambda_kind),
    block_has_shape shape tag fields ->
    nth_error fields index = Some field ->
    block_shape_element_kind shape index = Some k ->
    kind_of_value field = k.
