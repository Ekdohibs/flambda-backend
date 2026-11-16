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



(* What the answers of the provers mean.

   This file is to be reviewed: the two forms of statement
   ([prover_sound] and [meet_shortcut_sound]), what each answer is said
   to say of a value (the predicates of the section "What the answers
   say"), the domains, and the statements. The provers are
   FlambdaProvers.v, which is to be read against
   middle_end/flambda2/types/provers.ml; what a type means is
   FlambdaTypeMeaningSpec.v, and what satisfies a typing environment
   FlambdaTypingEnvSpec.v. The plan of the workstream is
   middle_end/flambda2/rocq/requests/T6-provers.md.

   The compiler has no such statements to compare with. What an answer
   means there is what Simplify does with it: these are to be read
   against the callers in simplify/simplify_unary_primitive.ml,
   simplify_binary_primitive.ml and the files that they call. A rewrite
   of Simplify that follows an answer is justified by the statement of
   the prover here, and by nothing else that the types say.

   The two forms.
   - A proof ([Provers.proof_of_property], the functions [prove_*]):
     [Proved x] says [x] of every value of the type
     ([prover_sound]).
   - A meet shortcut ([Provers.meet_shortcut], the functions [meet_*]):
     the answer stands for the meet of the type with the type of the
     values on which the operation that asks is defined, its domain.
     [Known_result x] says [x] of every value of the type that is in
     the domain, and [Invalid] says that no value of the type is in it
     ([meet_shortcut_sound]). The domain is part of the statement of
     each: it is what the caller must know of the value, from the
     operation being defined on it, to use the answer. A domain here is
     never narrower than where the primitive that asks has a value;
     that is for the statement that ties each to its primitive, which
     is to say that the primitive has a value only on the domain, and
     not the converse. There is one exception:
     [meet_project_value_slot_simple], whose domain asks more than
     that the projection has a value. Its statement is conditional on
     a premise that the machine does not give, and its [Invalid] is
     not, by that statement alone, a reason to take a projection to
     have no value (the section "Closures"). Its [Known_result] is
     right without the premise, and has a statement of its own.
   Nothing is said of [Unknown], of [Need_meet], or of [None], which is
   a stop of the compiler (FlambdaProvers.v): a prover may answer any
   of them on any type. That a prover answers when the type decides
   the question is not stated, nor that it does not stop.

   Both forms are about a type, a typing environment and a valuation
   that satisfies the environment strictly
   ([typing_env_satisfied_strictly]): every name that has a value has
   the type that the environment has for it. The provers expand names
   (the alias that the type is, the immediates of a variant, the names
   of the relations of a naked immediate), so a statement about a
   prover needs that of every name that the prover may reach. Both
   forms also take a condition on the environment and the type asked
   about, [input], for the provers that rely on more than the meaning
   of a type gives. There are two here: none ([any_input]), and that
   the type and the types of the environment are well formed
   ([well_formed_input], with [type_well_formed] of
   FlambdaTypeWellFormedSpec.v and [types_well_formed] of
   FlambdaTypingEnvInvariantSpec.v).

   Where well-formedness is needed, and why. The provers that read the
   numbers of a type of naked immediates
   ([prove_naked_immediates_generic]: [meet_naked_immediates] and the
   three provers of tagged immediates) answer that the type has no
   value when its head is of another kind. The model's grammar has a
   type that the compiler's has not, a type of naked numbers at the
   kind of the immediates ([Type_naked_number Kind_naked_immediate]),
   whose head is of naked numbers and whose values are naked
   immediates: of it the answer would be wrong. A well-formed type is
   not that type and has none as the immediates of a variant
   (conditions 2, 3 and 10 of FlambdaTypeWellFormedSpec.v); nothing
   else of well-formedness is used by these. The condition is on the
   types of the environment too, since a prover expands names. That
   expanding the head of a well-formed type in such an environment
   gives a well-formed type is proved here, from the definitions of
   [expand_head] and [Typing_env.find].

   Three other statements take well-formed types, each for a reason
   of its own.
   - [meet_single_closures_entry] answers that a type has no closure
     where the one case of its closures is exact and has no function
     type for its own slot, which the meaning of types does not say
     (FlambdaTypeAccessSpec.v, discrepancy T1-7). A well-formed type
     has no such case (condition 8).
   - That [prove_unique_fully_constructed_immutable_heap_block]
     answers a [Simple] for every field of the block needs a product
     as wide as an exact size (condition 6).
   - [prove_physical_equality] takes two blocks of shapes that are
     not equal to be two blocks, which is so of shapes written in one
     way (condition 7).

   Poison (decision D26).
   - Under strict satisfaction a value that is poison has only types
     that say no more of it than its kind, that it is the value of a
     name, or that it is not null (FlambdaTypeMeaningSpec.v). So on a
     type that has a value that is poison the provers that the
     statements of silence list do not answer [Proved], nor the meet
     shortcuts that they list [Known_result]
     ([prover_silent_on_poison], [meet_shortcut_silent_on_poison]; the
     statements are those whose names end in
     [silent_on_poison_statement]). Only those two answers are ruled
     out. A meet shortcut may answer [Invalid] on such a type, and
     rightly, where the poison is outside its domain:
     [meet_is_int_variant_only] answers [Invalid] on the unknown type
     of naked immediates, which a poison of that kind has. And a
     prover may stop.
   - Two provers are in no such list. [meet_is_null] answers
     [Known_result false] on a type of values that are not null:
     poison is not null. The primitive [Is_null] has no value on
     poison (FlambdaPrimitiveSpec.v), so nothing follows from that
     answer where the value is poison. And
     [prove_equals_to_simple_of_kind] answers the canonical element of
     the alias that a type is, which has the value of the alias,
     poison or not: of the alias of a poison constant it answers
     [Proved] of that constant, and rightly.
   - A name that a prover reaches inside a type has its type, so what
     the prover finds of it is true of its value. In particular the
     three relations of a naked immediate are read soundly by
     [meet_naked_immediates]: [Is_int] and [Get_tag] hold of no poison,
     [Is_null] relates a poison of kind value to 0, and a name that
     holds poison has no type from which a prover answers 1. For
     [Get_tag] the prover answers only when the type of the name has
     no immediate, so it concludes nothing of a name that may be one,
     where the relation says nothing: it does not do what
     [reduce_inverse_relations] of the meet does (discrepancy T1-1).
   - The statements do not hold with [typing_env_satisfied], under
     which a name that holds poison is exempt from its type. There a
     prover that expands such a name answers what the environment says
     of the name, which is not true of its value. One case is known:
     a name that holds poison and that the environment gives the type
     of null, and a naked immediate with the relation [Is_null] to
     that name: the number is 0 by the relation, and
     [meet_naked_immediates] answers that it is 1 (discrepancy T6-1;
     T1-9 is the same). Where no name holds poison the two notions
     are one (FlambdaTypingEnvSpec.v), and a statement here holds
     with [typing_env_satisfied]: the last section says so of the two
     forms. FlambdaProversPoisonSpec.v has what holds there with names
     that hold poison: the statements of the provers that read no
     relation of a naked immediate, of a value that is not poison.

   A [Simple] that a prover answers. Where an answer is a [Simple],
   the statement is that the [Simple] has a value in the valuation
   ([eval_simple valuation s]). That is not yet that a term may use
   it in place of what was asked about. A valuation gives values to
   names that no term can mention (those of name mode [In_types],
   those out of scope), and the environment of the machine agrees
   with it only on the names in scope at the normal name mode
   ([agrees_in_terms] of FlambdaTypingEnvSpec.v). A rewrite that puts
   the [Simple] in a term needs that it is such a name, a symbol or a
   constant, which is for its own statement to get from the name mode
   that it asked the prover for, with the premises on the environment
   that FlambdaTypingEnvSpec.v has for that; nothing here says it.

   What rests on other files. A prover expands the head of a type
   with [expand_head] of FlambdaExpandHead.v, and every statement
   here rests on what FlambdaExpandHeadSpec.v says of it: expanding
   the head of a type keeps its values
   ([expand_head_sound_statement]). Where a prover reads a row-like
   structure it is through the functions of FlambdaTypeAccessSpec.v,
   with the statements there.

   What is not said.
   - Code age, allocation modes and uniqueness: nothing in these
     statements is about them (FlambdaTypeMeaningSpec.v).
   - A tagged immediate is any constant [Tagged_immediate], whatever
     its number, as in the meaning of types; the primitives have a
     value only on one in the range of the machine.
   - A set that a prover answers is a list, and the statements read it
     by membership.

   The primitives. The last section ties the scrutinee questions to
   the primitives of the machine: each of [Is_int], [Is_null],
   [Get_tag] and [Untag_immediate] has a value only on a value of the
   domain used here, and its value is what the answer says
   ([is_int_defined_statement] and the three beside it); and the two
   things that Simplify does with [prove_is_int] are right
   ([prove_is_int_is_int_statement], [prove_is_int_get_tag_statement]).
   The other primitives that ask a prover of this file (arithmetic and
   comparison on tagged and naked immediates, [Boolean_not], the
   conversions) are for the workstream of those rewrites to tie to
   the domains in the same way.

   The proofs are in FlambdaProversProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec FlambdaExpandHead
  FlambdaProvers.
Import ListNotations.

(* ======================================================================== *)
(* The two forms of statement                                               *)
(* ======================================================================== *)

(* No condition on what a prover is asked. *)
Definition any_input (tenv : Typing_env.t) (t : flambda_type) : Prop := True.

(* The type that a prover is asked about is well formed, and so are the
   types of the environment ([types_well_formed] of
   FlambdaTypingEnvInvariantSpec.v: every type that the environment
   has for a name is well formed). *)
(* CR mshinwell: this asks more of a type than the compiler checks:
   [type_well_formed] has conditions that only some of the creation
   functions establish (FlambdaTypeWellFormedSpec.v says which). The
   provers that take it here use conditions 2, 3 and 10, which exclude
   a type that the compiler's grammar has not, and three statements
   one more each (conditions 6, 7 and 8; the header says which); a
   statement that took those alone would hold of more types. Decision
   D32. *)
Definition well_formed_input (tenv : Typing_env.t) (t : flambda_type)
    : Prop :=
  types_well_formed tenv ∧ type_well_formed t = true.

(* A proof is sound: what it answers holds of every value of the
   type. *)
(* CR mshinwell: the provers are stated, here and in
   [meet_shortcut_sound], for a valuation that satisfies the
   environment strictly. After the unboxing of variants only
   [typing_env_satisfied] holds, which exempts the names that hold
   poison, and under it a prover is not sound of every value: it
   answers what the environment says of a name that it expands,
   whatever the name holds (discrepancy T6-1).
   FlambdaProversPoisonSpec.v says of which provers and of which
   values it is sound there. The compiler relies on no answer about
   such a name being used on a path that runs, and nothing checks
   that. Decision D26. *)
Definition prover_sound {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    prover tenv t = Some (Proof_of_property.Proved x) ->
    holds statics valuation x v.

(* A meet shortcut is sound: what it answers holds of every value of
   the type that is in the domain, and if it answers that the meet is
   invalid there is none. *)
Definition meet_shortcut_sound {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    domain statics v ->
    match prover tenv t with
    | Some (Meet_shortcut.Known_result x) => holds statics valuation x v
    | Some Meet_shortcut.Invalid => False
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* On a type that has a value that is poison a proof does not answer
   [Proved], and a meet shortcut does not answer [Known_result]. Nothing
   else is ruled out: [Unknown], [Need_meet], [Invalid] and a stop are
   all possible there. *)
Definition prover_silent_on_poison {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t ->
    is_poison v ->
    prover tenv t <> Some (Proof_of_property.Proved x).

Definition meet_shortcut_silent_on_poison {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
    : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t ->
    is_poison v ->
    prover tenv t <> Some (Meet_shortcut.Known_result x).

(* ======================================================================== *)
(* What the answers say                                                     *)
(* ======================================================================== *)

(* Each is what an answer [x] says of a value [v]. *)

(* Whether the value is a tagged immediate. *)
Definition is_int_is (x : bool) (v : value) : Prop :=
  if x then is_immediate v else ~ is_immediate v.

(* Whether the value is not a pointer: a tagged immediate or null. *)
Definition is_not_a_pointer_is (x : bool) (v : value) : Prop :=
  if x
  then is_immediate v ∨ v = Value_const Null
  else ~ is_immediate v ∧ v <> Value_const Null.

(* Whether the value is null. *)
Definition is_null_is (x : bool) (v : value) : Prop :=
  if x then v = Value_const Null else v <> Value_const Null.

(* The value is a block whose tag is one of the tags. *)
Definition block_with_tag_among (statics : static_env) (tags : list Z)
    (v : value) : Prop :=
  ∃ (tag : Z) (fields : list value),
    block_of statics v = Some (tag, fields) ∧ tag ∈ tags.

(* The value is a naked immediate that is one of the numbers. *)
Definition naked_immediate_among (imms : list Z) (v : value) : Prop :=
  ∃ i : Z, v = Value_const (Naked_immediate i) ∧ i ∈ imms.

(* The value is a tagged immediate whose number is one of the
   numbers. *)
Definition tagged_immediate_among (imms : list Z) (v : value) : Prop :=
  ∃ i : Z, v = Value_const (Tagged_immediate i) ∧ i ∈ imms.

(* ======================================================================== *)
(* The domains                                                              *)
(* ======================================================================== *)

(* The values of a kind. *)
Definition values_of_kind (k : flambda_kind) (statics : static_env)
    (v : value) : Prop :=
  kind_of_value v = k.

(* The values that [Is_int] with [variant_only] is asked of: a tagged
   immediate or a block. *)
Definition variants (statics : static_env) (v : value) : Prop :=
  is_immediate v ∨ is_block statics v.

(* The tagged immediates. *)
Definition tagged_immediates (statics : static_env) (v : value) : Prop :=
  is_immediate v.

(* ======================================================================== *)
(* The scrutinee questions                                                  *)
(* ======================================================================== *)

(* [Provers.prove_is_int]. It is asked for [Is_int] without
   [variant_only] ([Obj.is_int]), so it answers of every value: a boxed
   number, a closure or a string is not a tagged immediate. *)
Definition prove_is_int_sound_statement : Prop :=
  prover_sound Provers.prove_is_int any_input (fun _ _ => is_int_is).

(* [Provers.meet_is_int_variant_only]: of the values of the type that
   are a tagged immediate or a block. On a type of boxed numbers, of
   closures or of strings it answers that there is none. *)
Definition meet_is_int_variant_only_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_is_int_variant_only any_input variants
    (fun _ _ => is_int_is).

(* [Provers.prove_is_not_a_pointer]. *)
Definition prove_is_not_a_pointer_sound_statement : Prop :=
  prover_sound Provers.prove_is_not_a_pointer any_input
    (fun _ _ => is_not_a_pointer_is).

(* [Provers.prove_get_tag]: every value of the type is a block, with
   one of the tags. *)
Definition prove_get_tag_sound_statement : Prop :=
  prover_sound Provers.prove_get_tag any_input
    (fun statics _ => block_with_tag_among statics).

(* [Provers.meet_is_null]: of the values of kind value. *)
Definition meet_is_null_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_is_null any_input (values_of_kind Value)
    (fun _ _ => is_null_is).

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

(* [Provers.meet_naked_immediates]: of the naked immediates. The answer
   takes in what the relations of the type say of its number: each is
   read from the type that the environment has for the name. *)
Definition meet_naked_immediates_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_naked_immediates well_formed_input
    (values_of_kind (Naked_number Kind_naked_immediate))
    (fun _ _ => naked_immediate_among).

(* [Provers.prove_equals_tagged_immediates]: every value of the type is
   a tagged immediate, one of the numbers. *)
Definition prove_equals_tagged_immediates_sound_statement : Prop :=
  prover_sound Provers.prove_equals_tagged_immediates well_formed_input
    (fun _ _ => tagged_immediate_among).

(* [Provers.meet_equals_tagged_immediates]: of the values of the type
   that are tagged immediates. *)
Definition meet_equals_tagged_immediates_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_equals_tagged_immediates
    well_formed_input tagged_immediates
    (fun _ _ => tagged_immediate_among).

(* [Provers.meet_equals_single_tagged_immediate]: the one number. *)
Definition meet_equals_single_tagged_immediate_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_equals_single_tagged_immediate
    well_formed_input tagged_immediates
    (fun _ _ imm v => v = Value_const (Tagged_immediate imm)).

(* [Provers.meet_naked_number], which is [meet_naked_float32s], ...,
   [meet_naked_masks] at the eleven kinds that are not that of the
   immediates: of the values of the kind, which are then numbers of the
   kind, one of those answered ([naked_numbers_satisfied],
   FlambdaTypeMeaningSpec.v). *)
Definition meet_naked_number_sound_statement : Prop :=
  ∀ kind : naked_number_kind,
    kind <> Kind_naked_immediate ->
    meet_shortcut_sound (Provers.meet_naked_number kind) any_input
      (values_of_kind (Naked_number kind))
      (fun _ _ numbers v => naked_numbers_satisfied kind v numbers).

(* ======================================================================== *)
(* Poison                                                                   *)
(* ======================================================================== *)

(* On a type that has a value that is poison none of these answers
   [Proved] or [Known_result]. [meet_is_null] is not among them. *)
Definition provers_silent_on_poison_statement : Prop :=
  prover_silent_on_poison Provers.prove_is_int
  ∧ meet_shortcut_silent_on_poison Provers.meet_is_int_variant_only
  ∧ prover_silent_on_poison Provers.prove_is_not_a_pointer
  ∧ prover_silent_on_poison Provers.prove_get_tag
  ∧ meet_shortcut_silent_on_poison Provers.meet_naked_immediates
  ∧ prover_silent_on_poison Provers.prove_equals_tagged_immediates
  ∧ meet_shortcut_silent_on_poison Provers.meet_equals_tagged_immediates
  ∧ meet_shortcut_silent_on_poison
      Provers.meet_equals_single_tagged_immediate
  ∧ ∀ kind : naked_number_kind,
      meet_shortcut_silent_on_poison (Provers.meet_naked_number kind).

(* [meet_is_null] says of a poison of kind value that it is not null,
   where the type says so, and never that it is. *)
Definition meet_is_null_poison_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t ->
    is_poison v ->
    Provers.meet_is_null tenv t
    <> Some (Meet_shortcut.Known_result true).

(* ======================================================================== *)
(* The answers against the primitives                                       *)
(* ======================================================================== *)

(* The primitives of the scrutinee questions, against the domains and
   the answers above. The first four are of the machine alone
   (FlambdaPrimitiveSpec.v): a primitive that has a value has it on a
   value of the domain, and its value is what the answer says. Not the
   converse: on a value of the domain a primitive need not have a value
   (a tagged immediate out of the range of the machine, for each of
   them; a symbol that is bound to nothing; poison for [Is_null]). *)

(* [Is_int]: with [variant_only], of a tagged immediate or a block
   only. *)
Definition is_int_defined_statement : Prop :=
  ∀ (statics : static_env) (variant_only : bool) (v r : value),
    eval_prim statics (Unary (Is_int variant_only)) [v] = Some r ->
    (variant_only = true -> variants statics v)
    ∧ ∃ b : bool, r = naked_bool b ∧ is_int_is b v.

(* [Is_null]: of a value of kind value. *)
Definition is_null_defined_statement : Prop :=
  ∀ (statics : static_env) (v r : value),
    eval_prim statics (Unary Is_null) [v] = Some r ->
    values_of_kind Value statics v
    ∧ ∃ b : bool, r = naked_bool b ∧ is_null_is b v.

(* [Get_tag]: of a block, and it is its tag. *)
Definition get_tag_defined_statement : Prop :=
  ∀ (statics : static_env) (v r : value),
    eval_prim statics (Unary Get_tag) [v] = Some r ->
    ∃ (tag : Z) (fields : list value),
      block_of statics v = Some (tag, fields)
      ∧ r = Value_const (Naked_immediate tag).

(* [Untag_immediate]: of a tagged immediate, and it is its number. *)
Definition untag_immediate_defined_statement : Prop :=
  ∀ (statics : static_env) (v r : value),
    eval_prim statics (Unary Untag_immediate) [v] = Some r ->
    ∃ i : Z,
      v = Value_const (Tagged_immediate i)
      ∧ r = Value_const (Naked_immediate i).

(* What Simplify does with [Provers.prove_is_int]
   ([Simplify_unary_primitive.simplify_is_int] and [simplify_get_tag]).
   Where it proves [b], [Is_int] without [variant_only] is [b] on every
   value of the type on which it has a value: the primitive may be
   replaced by the constant. *)
Definition prove_is_int_is_int_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (b : bool) (v r : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t ->
    Provers.prove_is_int tenv t = Some (Proof_of_property.Proved b) ->
    eval_prim statics (Unary (Is_int false)) [v] = Some r ->
    r = naked_bool b.

(* And where it proves that every value of the type is a tagged
   immediate, [Get_tag] has no value on any: the primitive may be
   replaced by [Invalid]. *)
Definition prove_is_int_get_tag_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t ->
    Provers.prove_is_int tenv t = Some (Proof_of_property.Proved true) ->
    eval_prim statics (Unary Get_tag) [v] = None.

(* ======================================================================== *)
(* Aliases                                                                  *)
(* ======================================================================== *)

(* [Provers.prove_equals_to_simple_of_kind], at any kind: every value
   of the type is the value of the [Simple] answered. It is not silent
   on poison: the canonical element of an alias has the value of the
   alias, poison or not (the aliases are exact,
   FlambdaEnvAliasesSpec.v). *)
Definition prove_equals_to_simple_of_kind_sound_statement : Prop :=
  ∀ kind : flambda_kind,
    prover_sound
      (fun tenv t => Provers.prove_equals_to_simple_of_kind tenv t kind)
      any_input
      (fun _ valuation s v => eval_simple valuation s = Some v).

(* The value is the tagged immediate of the number that is the value
   of the [Simple]. *)
Definition tagging_of (valuation : env) (s : simple) (v : value) : Prop :=
  ∃ i : Z,
    v = Value_const (Tagged_immediate i)
    ∧ eval_simple valuation s = Some (Value_const (Naked_immediate i)).

(* [Provers.prove_tagging_of_simple], at any name mode. It reads the
   numbers of the immediates of a variant, so the type is to be well
   formed. *)
Definition prove_tagging_of_simple_sound_statement : Prop :=
  ∀ min_name_mode : Name_mode.t,
    prover_sound
      (fun tenv t => Provers.prove_tagging_of_simple tenv min_name_mode t)
      well_formed_input (fun _ => tagging_of).

(* [Provers.meet_tagging_of_simple]: of the values of the type that
   are tagged immediates. *)
Definition meet_tagging_of_simple_sound_statement : Prop :=
  ∀ min_name_mode : Name_mode.t,
    meet_shortcut_sound
      (fun tenv t => Provers.meet_tagging_of_simple tenv min_name_mode t)
      well_formed_input tagged_immediates (fun _ => tagging_of).

(* ======================================================================== *)
(* Boxed numbers                                                            *)
(* ======================================================================== *)

(* The boxed numbers of a kind. [Unbox_number] of the kind has a value
   only on one of them ([unbox_number_defined_statement]), and not on
   every one: a boxed number whose contents are poison is one of
   them, and the primitive has no value on it
   (FlambdaPrimitiveSpec.v). *)
Definition boxed_numbers (kind : boxable_number) (statics : static_env)
    (v : value) : Prop :=
  ∃ c : const,
    boxed_number_of statics v = Some c
    ∧ kind_of_const c = unboxed_kind kind.

(* What [Provers.prove_is_a_boxed_or_tagged_number] answers: the value
   is a tagged immediate; or it is a boxed number of the kind answered
   whose contents have the type answered. The allocation mode answered
   means nothing here. *)
Definition boxed_or_tagged_number_is (statics : static_env)
    (valuation : env) (x : Boxed_or_tagged_number.t) (v : value) : Prop :=
  match x with
  | Boxed_or_tagged_number.Tagged_immediate => is_immediate v
  | Boxed_or_tagged_number.Boxed _ k contents =>
    ∃ c : const,
      boxed_number_of statics v = Some c
      ∧ kind_of_const c = unboxed_kind k
      ∧ type_satisfied statics valuation (Value_const c) contents
  end.

Definition prove_is_a_boxed_or_tagged_number_sound_statement : Prop :=
  prover_sound Provers.prove_is_a_boxed_or_tagged_number any_input
    boxed_or_tagged_number_is.

Definition prove_is_a_tagged_immediate_sound_statement : Prop :=
  prover_sound Provers.prove_is_a_tagged_immediate any_input
    (fun _ _ _ v => is_immediate v).

(* [Provers.prove_is_a_boxed_float32], ..., [prove_is_a_boxed_mask],
   which are [Provers.prove_is_a_boxed_number] at the nine numbers. *)
Definition prove_is_a_boxed_number_sound_statement : Prop :=
  ∀ kind : boxable_number,
    prover_sound (Provers.prove_is_a_boxed_number kind) any_input
      (fun statics _ _ v => boxed_numbers kind statics v).

(* [Provers.prove_is_or_is_not_a_boxed_float]. It answers that a value
   is not a boxed float from the head of its type being of another
   form: a block, a closure, a string and a boxed number of another
   kind are not boxed floats for the machine, whose observations are
   disjoint (FlambdaValueSpec.v). *)
Definition prove_is_or_is_not_a_boxed_float_sound_statement : Prop :=
  prover_sound Provers.prove_is_or_is_not_a_boxed_float any_input
    (fun statics _ x v =>
       if x
       then boxed_numbers Boxable_naked_float statics v
       else ~ boxed_numbers Boxable_naked_float statics v).

(* [Provers.meet_boxed_float32_containing_simple], ...,
   [meet_boxed_mask_containing_simple], which are
   [Provers.meet_boxed_number_containing_simple] at the nine numbers:
   of the values of the type that are boxed numbers of the kind, the
   contents are the value of the [Simple] answered. *)
Definition meet_boxed_number_containing_simple_sound_statement : Prop :=
  ∀ (kind : boxable_number) (min_name_mode : Name_mode.t),
    meet_shortcut_sound
      (fun tenv t =>
         Provers.meet_boxed_number_containing_simple kind tenv min_name_mode
           t)
      any_input (boxed_numbers kind)
      (fun statics valuation s v =>
         ∃ c : const,
           boxed_number_of statics v = Some c
           ∧ eval_simple valuation s = Some (Value_const c)).

(* [Unbox_number] has a value only on a boxed number of its kind, and
   its value is the contents. *)
Definition unbox_number_defined_statement : Prop :=
  ∀ (statics : static_env) (kind : boxable_number) (v r : value),
    eval_prim statics (Unary (Unbox_number kind)) [v] = Some r ->
    boxed_numbers kind statics v
    ∧ ∃ c : const, boxed_number_of statics v = Some c ∧ r = Value_const c.

(* ======================================================================== *)
(* Strings                                                                  *)
(* ======================================================================== *)

(* The strings, and the value is one of the strings. *)
Definition strings (statics : static_env) (v : value) : Prop :=
  ∃ bytes : list Z, string_of statics v = Some bytes.

Definition string_among (statics : static_env) (strs : list (list Z))
    (v : value) : Prop :=
  ∃ bytes : list Z, string_of statics v = Some bytes ∧ bytes ∈ strs.

Definition prove_strings_sound_statement : Prop :=
  prover_sound Provers.prove_strings any_input
    (fun statics _ => string_among statics).

Definition meet_strings_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_strings any_input strings
    (fun statics _ => string_among statics).

(* On a type that has a value that is poison none of these answers
   [Proved] or [Known_result] either. [prove_equals_to_simple_of_kind]
   is not among them. *)
Definition more_provers_silent_on_poison_statement : Prop :=
  (∀ min_name_mode : Name_mode.t,
    prover_silent_on_poison
      (fun tenv t => Provers.prove_tagging_of_simple tenv min_name_mode t))
  ∧ (∀ min_name_mode : Name_mode.t,
      meet_shortcut_silent_on_poison
        (fun tenv t => Provers.meet_tagging_of_simple tenv min_name_mode t))
  ∧ prover_silent_on_poison Provers.prove_is_a_boxed_or_tagged_number
  ∧ prover_silent_on_poison Provers.prove_is_a_tagged_immediate
  ∧ (∀ kind : boxable_number,
      prover_silent_on_poison (Provers.prove_is_a_boxed_number kind))
  ∧ prover_silent_on_poison Provers.prove_is_or_is_not_a_boxed_float
  ∧ (∀ (kind : boxable_number) (min_name_mode : Name_mode.t),
      meet_shortcut_silent_on_poison
        (fun tenv t =>
           Provers.meet_boxed_number_containing_simple kind tenv
             min_name_mode t))
  ∧ prover_silent_on_poison Provers.prove_strings
  ∧ meet_shortcut_silent_on_poison Provers.meet_strings.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

(* What [Provers.prove_variant_like] answers: the value is a tagged
   immediate, one of the constant constructors if they are known; or
   it is a block whose tag is one of those answered, of the size and
   of the shape answered for the tag. *)
Definition variant_like (statics : static_env) (proof : variant_like_proof)
    (v : value) : Prop :=
  (∃ i : Z,
    v = Value_const (Tagged_immediate i)
    ∧ match const_ctors proof with
      | Or_unknown.Known imms => i ∈ imms
      | Or_unknown.Unknown => True
      end)
  ∨ (∃ (tag : Z) (fields : list value) (size : nat) (shape : block_shape),
      block_of statics v = Some (tag, fields)
      ∧ Keyed.lookup Z.eqb tag (non_const_ctors_with_sizes proof)
        = Some (size, shape)
      ∧ length fields = size
      ∧ block_has_shape shape tag fields).

(* [Provers.prove_variant_like]. It reads the numbers of the immediates
   of the variant, so the type is to be well formed. *)
Definition prove_variant_like_sound_statement : Prop :=
  prover_sound Provers.prove_variant_like well_formed_input
    (fun statics _ => variant_like statics).

(* [Provers.meet_variant_like]: of the values of the type that are a
   tagged immediate or a block. *)
Definition meet_variant_like_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_variant_like well_formed_input variants
    (fun statics _ => variant_like statics).

(* The value is a block of a tag, a shape and a size. *)
Definition block_of_tag_shape_and_size (statics : static_env)
    (x : Z * block_shape * nat) (v : value) : Prop :=
  let '(tag, shape, size) := x in
  ∃ fields : list value,
    block_of statics v = Some (tag, fields)
    ∧ length fields = size
    ∧ block_has_shape shape tag fields.

(* [Provers.prove_unique_tag_and_size]. *)
Definition prove_unique_tag_and_size_sound_statement : Prop :=
  prover_sound Provers.prove_unique_tag_and_size any_input
    (fun statics _ => block_of_tag_shape_and_size statics).

(* [Provers.prove_unique_fully_constructed_immutable_heap_block]: the
   value is a block of the tag, the shape and the size answered, and
   each [Simple] answered has the value of the field at its position.
   The [Simple]s are the aliases that the types of the fields are, as
   they stand in the type and not their canonical elements. There are
   as many as the type has types of fields, which may be fewer than
   the size: that there are as many is the statement that follows,
   for a well-formed type. The allocation mode that the prover asks
   for means nothing here. *)
Definition prove_unique_immutable_heap_block_sound_statement : Prop :=
  prover_sound Provers.prove_unique_fully_constructed_immutable_heap_block
    any_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ ∀ (i : nat) (s : simple) (field : value),
             nth_error simples i = Some s ->
             nth_error fields i = Some field ->
             eval_simple valuation s = Some field).

(* The same of a well-formed type, with what the caller needs
   ([Simplify_unary_primitive], which records that a [Make_block] of
   the [Simple]s answered is the block): there are as many [Simple]s
   as the size, and their values are the fields of the block, all of
   them and in their order ([eval_simples] of FlambdaSemanticsSpec.v).
   Without well-formedness there may be fewer [Simple]s than fields. A
   type whose one case of blocks has an exact size and a product of
   fewer types has the blocks of that size, whatever their other
   fields, and the prover answers the size with the [Simple]s of the
   product; condition 6 of FlambdaTypeWellFormedSpec.v excludes such a
   case. Nothing is said of where the block is allocated, nor that a
   block made of the same fields is the same block. *)
Definition prove_unique_immutable_heap_block_fields_statement : Prop :=
  prover_sound Provers.prove_unique_fully_constructed_immutable_heap_block
    well_formed_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ length simples = size
         ∧ eval_simples valuation simples = Some fields).

(* The values on which the load of a field of a kind has a value: a
   block that has the field, of that kind. *)
Definition block_field_loads (field_kind : flambda_kind) (field_index : nat)
    (statics : static_env) (v : value) : Prop :=
  ∃ (tag : Z) (fields : list value) (field : value),
    block_of statics v = Some (tag, fields)
    ∧ nth_error fields field_index = Some field
    ∧ kind_of_value field = field_kind.

(* [Provers.meet_block_field_simple], at any name mode, kind and index:
   of the values of the type on which the load has a value, the field
   is the value of the [Simple] answered. The prover answers that there
   is no such value where the type of the field is of another kind
   than the one asked: a field of another kind is not in the domain. *)
Definition meet_block_field_simple_sound_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (field_kind : flambda_kind)
    (field_index : nat),
    meet_shortcut_sound
      (fun tenv t =>
         Provers.meet_block_field_simple tenv min_name_mode field_kind t
           field_index)
      any_input (block_field_loads field_kind field_index)
      (fun statics valuation s v =>
         ∃ (tag : Z) (fields : list value) (field : value),
           block_of statics v = Some (tag, fields)
           ∧ nth_error fields field_index = Some field
           ∧ eval_simple valuation s = Some field).

(* [Block_load] has a value only on a value of that domain, at the
   kind value, and its value is the field, whatever the load says of
   mutability. *)
Definition block_load_defined_statement : Prop :=
  ∀ (statics : static_env) (kind : block_access_kind) (mut : mutability)
    (field : Z) (v r : value),
    eval_prim statics (Unary (Block_load kind mut field)) [v] = Some r ->
    (0 <= field)%Z
    ∧ block_field_loads Value (Z.to_nat field) statics v
    ∧ ∃ (tag : Z) (fields : list value),
        block_of statics v = Some (tag, fields)
        ∧ nth_error fields (Z.to_nat field) = Some r.

(* On a type that has a value that is poison none of these answers
   [Proved] or [Known_result] either. *)
Definition block_provers_silent_on_poison_statement : Prop :=
  prover_silent_on_poison Provers.prove_variant_like
  ∧ meet_shortcut_silent_on_poison Provers.meet_variant_like
  ∧ prover_silent_on_poison Provers.prove_unique_tag_and_size
  ∧ prover_silent_on_poison
      Provers.prove_unique_fully_constructed_immutable_heap_block
  ∧ ∀ (min_name_mode : Name_mode.t) (field_kind : flambda_kind)
      (field_index : nat),
      meet_shortcut_silent_on_poison
        (fun tenv t =>
           Provers.meet_block_field_simple tenv min_name_mode field_kind t
             field_index).

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

(* The closures: the values that a call may have for its callee, and
   from which a function slot or a value slot may be projected. *)
Definition closures (statics : static_env) (v : value) : Prop :=
  ∃ (slot : function_slot) (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)),
    closure_in_set statics v = Some (slot, functions, value_slots).

(* What [Provers.prove_single_closures_entry] answers: the value is the
   closure at the function slot answered of a set of closures; the
   entry answered holds of it; and the code of the function at that
   slot is the code of the function type answered. The allocation mode
   answered means nothing here. *)
(* CR mshinwell: the code is the very code of the closure, as in the
   meaning of a function type (FlambdaTypeMeaningSpec.v), which leaves
   the age of code out. In the compiler the code id of a function type
   stands for that code or a newer version of it
   ([Code_age_relation]), and Simplify makes a direct call to the code
   id answered. Until the meaning has code age, this says nothing of a
   closure made with a newer version of the code than its type names.
   No decision has a number. *)
Definition single_closures_entry_holds (statics : static_env)
    (valuation : env)
    (x : function_slot * Alloc_mode.t * closures_entry * function_type)
    (v : value) : Prop :=
  let '(slot, _, entry, function_type) := x in
  ∃ (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)),
    closure_in_set statics v = Some (slot, functions, value_slots)
    ∧ closures_entry_satisfied statics valuation v slot functions value_slots
        entry
    ∧ lookup slot functions = Some (function_type_code_id function_type).

(* [Provers.prove_single_closures_entry]. *)
Definition prove_single_closures_entry_sound_statement : Prop :=
  prover_sound Provers.prove_single_closures_entry any_input
    single_closures_entry_holds.

(* [Provers.meet_single_closures_entry]: of the values of the type that
   are closures. The type is to be well formed: where the closures of
   the type are all at one slot of a set whose functions are known
   exactly, and the entry has no function type for that slot, the
   prover answers that the type has no closure, and the meaning of
   types does not say so (FlambdaTypeAccessSpec.v, discrepancy T1-7).
   A well-formed type has no such case (condition 8 of
   FlambdaTypeWellFormedSpec.v). *)
Definition meet_single_closures_entry_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_single_closures_entry well_formed_input
    closures single_closures_entry_holds.

(* What a call reads of its callee, [closure_of], is then the code of
   the function type answered. *)
Definition single_closures_entry_code_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (slot : function_slot)
    (alloc_mode : Alloc_mode.t) (entry : closures_entry)
    (function_type : function_type) (v : value),
    single_closures_entry_holds statics valuation
      (slot, alloc_mode, entry, function_type) v ->
    ∃ value_slots : list (value_slot * value),
      closure_of statics v
      = Some (function_type_code_id function_type, value_slots).

(* The value is a closure whose code is one of the code ids. The code
   is the very code of the closure, as for
   [single_closures_entry_holds]. *)
Definition closure_code_among (statics : static_env)
    (code_ids : list code_id) (v : value) : Prop :=
  ∃ (code : code_id) (value_slots : list (value_slot * value)),
    closure_of statics v = Some (code, value_slots) ∧ code ∈ code_ids.

(* [Provers.prove_code_ids]. *)
Definition prove_code_ids_sound_statement : Prop :=
  prover_sound Provers.prove_code_ids any_input
    (fun statics _ => closure_code_among statics).

(* [Provers.meet_code_ids]: of the values of the type that are
   closures. *)
Definition meet_code_ids_sound_statement : Prop :=
  meet_shortcut_sound Provers.meet_code_ids any_input closures
    (fun statics _ => closure_code_among statics).

(* The values from which a function slot is projected: the closures of
   a set that has the slot, at whichever slot of it the projection
   names ([project_function_slot] of FlambdaValueSpec.v, which is what
   [Project_function_slot] is). *)
Definition function_slot_projects (move_to : function_slot)
    (statics : static_env) (v : value) : Prop :=
  ∃ (move_from : function_slot) (closure : value),
    project_function_slot statics move_from move_to v = Some closure.

(* [Provers.meet_project_function_slot_simple], at any name mode and
   function slot: of the values of the type from which the slot is
   projected, the closure projected is the value of the [Simple]
   answered. The prover is not told the slot that the projection is
   from, and the answer is right for any: the projection has a value
   only from the slot that the closure is at. *)
Definition meet_project_function_slot_simple_sound_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (move_to : function_slot),
    meet_shortcut_sound
      (fun tenv t =>
         Provers.meet_project_function_slot_simple tenv min_name_mode t
           move_to)
      any_input (function_slot_projects move_to)
      (fun statics valuation s v =>
         ∀ (move_from : function_slot) (closure : value),
           project_function_slot statics move_from move_to v = Some closure ->
           eval_simple valuation s = Some closure).

(* The projection of a function slot has a value only on a value of
   that domain. *)
Definition project_function_slot_in_domain_statement : Prop :=
  ∀ (statics : static_env) (move_from move_to : function_slot)
    (v closure : value),
    project_function_slot statics move_from move_to v = Some closure ->
    function_slot_projects move_to statics v.

(* The closures of a set that has a value slot, with contents of a
   kind. This is narrower than where the projection of the slot has a
   value ([project_value_slot] of FlambdaValueSpec.v, which is what
   [Project_value_slot_from] is): the projection does not look at the
   kind of the contents. *)
(* CR mshinwell: a value slot of the model has no kind, and a set of
   closures may have contents of any kind in a value slot. The
   compiler's value slot has a kind ([Value_slot.kind]), and the
   prover answers that the type has no closure where the type of the
   slot is of another kind. So the kind is an argument here, and that
   the contents are of that kind is a premise of the statement, which
   nothing in the model gives: the projection of a slot whose contents
   are a naked number has that number for value, where the prover,
   asked at the kind value, answers [Invalid]
   (FlambdaProversExamplesSpec.v has the example). The statement does
   not justify that [Invalid] for the machine as it is. What would:
   value slots with kinds in the model, and either a projection that
   has no value on contents of another kind, or a condition on
   well-formed terms that the contents of a slot are of its kind;
   whether a condition on kinds may be asked of the compiler's terms
   is for the rulings on kinds (discrepancy T6-4; decisions D9, D10,
   D12 and D16). *)
Definition value_slot_projects (value_slot_kind : flambda_kind)
    (slot : value_slot) (statics : static_env) (v : value) : Prop :=
  ∃ (project_from : function_slot) (contents : value),
    project_value_slot statics project_from slot v = Some contents
    ∧ kind_of_value contents = value_slot_kind.

(* The projection of a value slot has a value in that domain if the
   contents are of the kind: the second premise is the one that the
   machine does not give. *)
Definition project_value_slot_in_domain_statement : Prop :=
  ∀ (statics : static_env) (project_from : function_slot)
    (slot : value_slot) (value_slot_kind : flambda_kind)
    (v contents : value),
    project_value_slot statics project_from slot v = Some contents ->
    kind_of_value contents = value_slot_kind ->
    value_slot_projects value_slot_kind slot statics v.

(* [Provers.meet_project_value_slot_simple], at any name mode, value
   slot and kind, of the values of the type from which the slot is
   projected with contents of that kind: the contents of the slot are
   the value of the [Simple] answered, and if the prover answers
   [Invalid] there is no such value. It says nothing of a closure
   whose slot has contents of another kind. *)
Definition meet_project_value_slot_simple_sound_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind),
    meet_shortcut_sound
      (fun tenv t =>
         Provers.meet_project_value_slot_simple tenv min_name_mode t slot
           value_slot_kind)
      any_input (value_slot_projects value_slot_kind slot)
      (fun statics valuation s v =>
         ∀ (project_from : function_slot) (contents : value),
           project_value_slot statics project_from slot v = Some contents ->
           eval_simple valuation s = Some contents).

(* The answer [Known_result] of [Provers.meet_project_value_slot_simple]
   needs no premise on the kind of the contents: of every value of the
   type from which the slot is projected, whatever the kind of what is
   projected, the contents are the value of the [Simple] answered.
   The prover answers [Known_result] only where the type that the
   closures have for the slot is of the kind asked, and the contents
   of the slot of a closure of the type have that type. So the premise
   on the kind is needed for the answer [Invalid] alone. *)
Definition meet_project_value_slot_simple_known_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (min_name_mode : Name_mode.t) (t : flambda_type) (slot : value_slot)
    (value_slot_kind : flambda_kind) (s : simple) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v t ->
    Provers.meet_project_value_slot_simple tenv min_name_mode t slot
      value_slot_kind
    = Some (Meet_shortcut.Known_result s) ->
    ∀ (project_from : function_slot) (contents : value),
      project_value_slot statics project_from slot v = Some contents ->
      eval_simple valuation s = Some contents.

(* On a type that has a value that is poison none of these answers
   [Proved] or [Known_result] either. *)
Definition closure_provers_silent_on_poison_statement : Prop :=
  prover_silent_on_poison Provers.prove_single_closures_entry
  ∧ meet_shortcut_silent_on_poison Provers.meet_single_closures_entry
  ∧ prover_silent_on_poison Provers.prove_code_ids
  ∧ meet_shortcut_silent_on_poison Provers.meet_code_ids
  ∧ (∀ (min_name_mode : Name_mode.t) (move_to : function_slot),
      meet_shortcut_silent_on_poison
        (fun tenv t =>
           Provers.meet_project_function_slot_simple tenv min_name_mode t
             move_to))
  ∧ ∀ (min_name_mode : Name_mode.t) (slot : value_slot)
      (value_slot_kind : flambda_kind),
      meet_shortcut_silent_on_poison
        (fun tenv t =>
           Provers.meet_project_value_slot_simple tenv min_name_mode t slot
             value_slot_kind).

(* ======================================================================== *)
(* Physical equality                                                        *)
(* ======================================================================== *)

(* [Provers.prove_physical_equality], of a value of each of two types:
   [Proved true] says that they are the same value, and [Proved false]
   that they are not. The types are to be well formed: the prover
   reads the numbers of the immediates of two variants, and takes two
   blocks of shapes that are not equal to be two blocks (conditions 2,
   3, 7 and 10 of FlambdaTypeWellFormedSpec.v). *)
(* CR mshinwell: this is about values of the model, which are not
   addresses: two blocks, two boxed numbers or two closures with the
   same contents are one value here, and may be two allocations when
   the program runs. So [Proved false] says more here than that the
   two are not physically equal, and [Proved true] says less than
   that they are. The prover answers [Proved true] only of two types
   of null, of two variants that are one tagged immediate, and of two
   aliases of one constant, one symbol or one variable. Of two values
   of kind value that are not poison, being the same value is there
   being physically equal, whatever allocation is. The statement has
   no such premise, and of two aliases of a naked constant or of a
   poison constant it says that they are the same value and no more:
   [Phys_equal] is of values of kind value, and what the compiler and
   the backend make of the equality of two poison constants is
   discrepancy P1-1. What [Phys_equal] is of two allocated values is
   not in the model. Decision D19. *)
Definition prove_physical_equality_sound_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) (b : bool) (v1 v2 : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    well_formed_input tenv t1 ->
    well_formed_input tenv t2 ->
    type_satisfied statics valuation v1 t1 ->
    type_satisfied statics valuation v2 t2 ->
    Provers.prove_physical_equality tenv t1 t2
    = Some (Proof_of_property.Proved b) ->
    if b then v1 = v2 else v1 <> v2.

(* The primitive [Phys_equal] against that answer: where it has a
   value on a value of each of the two types, its value is the answer
   of the prover, 1 or 0 for [Phys_eq] as the answer is [true] or
   [false], and the other way round for [Phys_neq]. That is the
   constant that Simplify puts for the primitive
   ([Simplify_binary_primitive.simplify_phys_equal]). The primitive
   has a value on two tagged immediates, and on a tagged immediate
   against an allocated value (FlambdaPrimitiveSpec.v); of two such
   values, being the same value is being physically equal. Where the
   primitive has no value (two allocated values, null, poison) this
   says nothing of the answer of the prover. *)
Definition prove_physical_equality_phys_equal_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) (b : bool) (op : equality_comparison)
    (v1 v2 r : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    well_formed_input tenv t1 ->
    well_formed_input tenv t2 ->
    type_satisfied statics valuation v1 t1 ->
    type_satisfied statics valuation v2 t2 ->
    Provers.prove_physical_equality tenv t1 t2
    = Some (Proof_of_property.Proved b) ->
    eval_prim statics (Binary (Phys_equal op)) [v1; v2] = Some r ->
    r = naked_bool match op with Phys_eq => b | Phys_neq => negb b end.

(* ======================================================================== *)
(* Where no name holds poison                                               *)
(* ======================================================================== *)

(* The two forms of statement for the notion of satisfaction that
   exempts the names that hold poison ([typing_env_satisfied] of
   FlambdaTypingEnvSpec.v), of a valuation in which no name holds
   poison. *)
Definition prover_sound_where_no_poison {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value),
    typing_env_satisfied statics valuation tenv ->
    (∀ n : name, ~ holds_poison valuation n) ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    prover tenv t = Some (Proof_of_property.Proved x) ->
    holds statics valuation x v.

Definition meet_shortcut_sound_where_no_poison {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    typing_env_satisfied statics valuation tenv ->
    (∀ n : name, ~ holds_poison valuation n) ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    domain statics v ->
    match prover tenv t with
    | Some (Meet_shortcut.Known_result x) => holds statics valuation x v
    | Some Meet_shortcut.Invalid => False
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* A prover that is sound is sound in that sense: the two notions of
   satisfaction are one there. So each statement of this file that is
   of one of the two forms gives one for [typing_env_satisfied]; that
   of [prove_physical_equality] goes the same way. The premise is of
   every name of the valuation. It is more than is needed. For the
   provers that read no relation of a naked immediate,
   FlambdaProversPoisonSpec.v has the statements with no premise on
   the names, of a value that is not poison. For the others what is
   needed is that the names that those relations are on hold no
   poison; that statement is not made. *)
Definition sound_where_no_poison_statement : Prop :=
  (∀ (A : Type)
     (prover :
       Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
     (input : Typing_env.t -> flambda_type -> Prop)
     (holds : static_env -> env -> A -> value -> Prop),
    prover_sound prover input holds ->
    prover_sound_where_no_poison prover input holds)
  ∧ ∀ (A : Type)
      (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
      (input : Typing_env.t -> flambda_type -> Prop)
      (domain : static_env -> value -> Prop)
      (holds : static_env -> env -> A -> value -> Prop),
      meet_shortcut_sound prover input domain holds ->
      meet_shortcut_sound_where_no_poison prover input domain holds.
