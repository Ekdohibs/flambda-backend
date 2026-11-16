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


(* What the n-way join of types guarantees: the type traversal of the
   join, against the join of environments.

   To be reviewed: every definition and statement of this file, with
   the section "The join of types, inside a join of environments" of
   FlambdaJoinSpec.v, which is the interface that this file is written
   against and does not change. The functions are FlambdaJoinTypes.v,
   a transcription of the join functions of
   middle_end/flambda2/types/meet_and_n_way_join.ml.

   What is proved is one thing ([n_way_join_sound_statement]): the join
   of types meets [n_way_join_type_sound], the contract that the join
   of environments assumes of the join of types that it is given, as
   that file states it, whatever the number that bounds its depth. With
   it ([n_way_join_bound_is_met_statement]): where the join answers a
   type, a value of a type given, under a valuation that satisfies its
   input, has the type answered under some valuation of the target
   that extends it, so that the upper bound is not said of nothing.
   This file also states the same contract of the functions that join
   heads with no types inside them, each in its own form
   ([flat_heads_sound_statement]); the same of every other function of
   the traversal is proved on the way, and stated with its proof.

   Proved beside it ([n_way_join_equations_kinded_statement]): the
   join of types meets the contract for the names of the equations
   inside the type that it answers
   ([n_way_join_type_equations_kinded], FlambdaJoinSpec.v), if the
   import of a type and the join of extensions meet theirs. A type
   answered has equations inside it only in extensions that the join
   of extensions answered and in types that were imported as they
   stood.

   Proved beside it too ([n_way_join_well_formed_statement]): the
   type that the join of types answers is well formed
   ([type_well_formed], FlambdaTypeWellFormedSpec.v) if the types
   given are, the environments joined have well-formed types and the
   state is kinded ([n_way_join_type_well_formed]). The join of
   extensions is to answer a well-formed extension for well-formed
   extensions. That the state is kinded is asked because the proof
   goes by the kinds of the answer, which the contract gives of a
   state that is kinded; it is not known to be needed.

   How it is stated. The functions of the traversal thread the state of
   the join of environments, of which this file knows what
   FlambdaJoinSpec.v says and no more: three readings, two invariants,
   and the contracts of the functions of the environment side that the
   traversal calls. Each function of the traversal is given a list of
   things (types, heads, cases, [Simple]s), each with the index of the
   environment in which it holds, and answers one thing and a state.
   [join_step_sound] says of such a function what [n_way_join_type_sound]
   says of the join of types, in the same words: the invariant is kept,
   the state grows, the answer mentions only variables of the target,
   the answer is an upper bound of each thing given, under any valuation
   of the target that extends that of its index, and the answer is
   kinded in the target if the state is kinded. What "upper bound"
   means depends on the things: for types, that a value that has the
   one has the other. The contract of the join of types says one thing
   more, which is of types only and which [join_step_sound] does not
   say: the type answered is of the kind of each type given. It is
   proved of the join of types itself, with the rest of its contract.

   The join of types is not a recursion on a type: when it is given an
   alias and something that is not one it expands the alias, to a type
   of an environment, which is no part of the types that it was given.
   So the functions are defined with a number that bounds the depth of
   the calls of the join of types to itself, and what is proved of the
   join of types is proved by induction on that number. An answer of
   [None] is the number run out, or a fatal error of the compiler; it
   is never a refusal of something that the compiler answers.

   What is assumed. Nothing here is an axiom; what a statement needs of
   another file's functions is a premise of it, a named proposition of
   this file:
   - [environment_side_sound]: the four statements of FlambdaJoinSpec.v
     about the functions of the environment side. The owner of the join
     of environments proves them of its functions. The first, that a
     valid state has witnesses, is used by the second statement only.
   - [reduce_head_of_kind_naked_immediate_sound]: a number that has a
     head of naked immediates, under a valuation that satisfies an
     environment strictly, has the head as the function reduces it in
     that environment; in particular the function does not answer
     bottom. The owner of the meet proves it. It is true because a
     relation [Is_int] or [Get_tag] is never true of a scrutinee that
     holds poison (decision D26, as amended; [relation_holds] of
     FlambdaTypeMeaningSpec.v).
   - [expand_head_sound]: a value that has a type has the type as
     [Expand_head] expands it, under a valuation that satisfies the
     environment strictly; the expanded type is in the scope of the
     environment and kinded there, where the type is; and it is of the
     kind of the type, if the type is kinded and the environment has
     [constants_of_kind] (FlambdaTypingEnvKindedSpec.v). The owner of
     the meet proves it.
   - [expand_head_equations_kinded], for the statement about the
     names of the equations only: in an environment that has the
     names of the equations inside its types at their kinds, a type
     that has its own at their kinds is expanded to one that has. (An
     expanded type is the type itself, an unknown type, the type of a
     constant or a type of the environment.)
   - [expand_head_well_formed], for the statement about
     well-formedness only: in an environment whose types are well
     formed, a type that is well formed is expanded to one that is.
   - [n_way_join_env_extension_well_formed], for the same statement
     only: the extension that the join of extensions answers is well
     formed if those given are. It is true of the join of extensions
     of the join of environments, which answers an extension that it
     was given or the empty one
     ([n_way_join_env_extension_answers_given_statement],
     FlambdaJoinEnvSpec.v).
   - [joined_envs_constants_of_kind]: in each environment joined, a
     constant that is the canonical element of a [Simple] of a kind is
     of that kind ([constants_of_kind]). It is a premise about kinds,
     beside [joined_envs_kinded] of the contract and, like it, no part
     of the well-formedness of environments. [Expand_head] answers the
     type of such a constant whatever the kind of the alias
     (discrepancy T4-7), so without it the join of aliases of one kind
     could be a type of another. It follows from [joined_envs_kinded]
     where the well-formedness of environments has their aliases
     consistent: an environment that is kinded has its aliases of one
     kind ([aliases_kinded] of that file). This file is for any
     well-formedness, so it is asked here;
     FlambdaJoinTypesWithEnvSpec.v, which is at the invariant of
     typing environments, does not ask it. (A condition of structure
     would give it too, that the type an environment has for a name
     that is not canonical is the alias type of its canonical element;
     the compiler's environments do not all have that one: a later
     demotion of the canonical element leaves the type as it was, and
     discrepancy T4-10 is another way.)
   - [type_eqb_sound] and [block_shape_eqb_sound]: two types that the
     test of structural equality identifies have the same values, and
     two shapes that the test of shapes identifies are one shape. The
     tests of FlambdaTypeEqualSpec.v have them.

   What is found of the compiler on the way is said at the function
   concerned, in FlambdaJoinTypes.v, and recorded as a discrepancy
   (T5b-1 and after; R1-16 and X3-5 are met here too). *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeWellFormedSpec FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinTypes FlambdaJoinSpec.
Import ListNotations.

(* ======================================================================== *)
(* What an upper bound is                                                   *)
(* ======================================================================== *)

(* Whatever has the thing given, under a valuation of an input, has the
   answer under a valuation of the target. [W] is what has it: a value
   for a type or a head, a tag and fields for blocks. *)
Definition all_kept {W X Y : Type}
    (holds : static_env -> env -> W -> X -> Prop)
    (holds' : static_env -> env -> W -> Y -> Prop)
    : static_env -> env -> X -> env -> Y -> Prop :=
  fun statics valuation x valuation' y =>
    ∀ w : W, holds statics valuation w x -> holds' statics valuation' w y.

(* An answer that may be unknown: unknown says nothing. *)
Definition known_bound {X Y : Type}
    (bound : static_env -> env -> X -> env -> Y -> Prop)
    : static_env -> env -> X -> env -> Or_unknown.t Y -> Prop :=
  fun statics valuation x valuation' y =>
    match y with
    | Or_unknown.Known y => bound statics valuation x valuation' y
    | Or_unknown.Unknown => True
    end.

(* The same where the thing given may be unknown too: then the answer
   is unknown. *)
Definition or_unknown_bound {X Y : Type}
    (bound : static_env -> env -> X -> env -> Y -> Prop)
    : static_env -> env -> Or_unknown.t X -> env -> Or_unknown.t Y -> Prop :=
  fun statics valuation x valuation' y =>
    match x, y with
    | _, Or_unknown.Unknown => True
    | Or_unknown.Known x, Or_unknown.Known y =>
      bound statics valuation x valuation' y
    | Or_unknown.Unknown, Or_unknown.Known _ => False
    end.

(* The variables, or the aliases, of something that may be unknown. *)
Definition of_known {X A : Type} (of_x : X -> list A) (x : Or_unknown.t X)
    : list A :=
  match x with
  | Or_unknown.Known x => of_x x
  | Or_unknown.Unknown => []
  end.

(* What a head of kind value says of whether a value is null:
   [is_null_says vars is_null null] is what it asks of a value that is
   null, if [null], and of one that is not, if not
   ([head_of_kind_value_satisfied]). *)
Definition is_null_says (vars : env) (is_null : is_null) (null : bool)
    : Prop :=
  match is_null with
  | Not_null => null = false
  | Maybe_null is_null => variable_is vars is_null (if null then 1%Z else 0%Z)
  end.

Definition is_null_of_head (head : head_of_kind_value) : is_null :=
  match head with
  | Head_of_kind_value _ is_null => is_null
  end.

Definition variables_of_is_null (is_null : is_null) : list variable :=
  match is_null with
  | Not_null => []
  | Maybe_null is_null => free_variables_of_option is_null
  end.

Definition aliases_of_is_null (is_null : is_null)
    : list (simple * flambda_kind) :=
  match is_null with
  | Not_null => []
  | Maybe_null is_null => aliases_of_relation is_null
  end.

(* ======================================================================== *)
(* A function of the traversal that is sound                                *)
(* ======================================================================== *)

Section Traversal.

(* The parameters of the interface (FlambdaJoinSpec.v). *)
Context (typing_env_well_formed : Typing_env.t -> Prop).

Context {state : Type}
  (joined_env : state -> Index.t -> option Typing_env.t)
  (definitions_of_existentials :
     state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
  (kind_in_target_env : state -> variable -> option flambda_kind)
  (state_valid : state -> Prop)
  (state_kinded : state -> Prop).

(* The functions of the environment side, and those of other files,
   that the traversal is given (FlambdaJoinTypes.v). *)
Context
  (n_way_join_simples :
     state -> flambda_kind -> list (Index.t * simple)
     -> option (Or_bottom.t simple * state))
  (import_type : state -> flambda_type -> option state)
  (join_env_n_way_join_env_extension :
     state -> list (Index.t * env_extension)
     -> option (Or_bottom.t (env_extension * state)))
  (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
  (reduce_head_of_kind_naked_immediate :
     Typing_env.t -> head_of_kind_naked_immediate
     -> Or_bottom.t head_of_kind_naked_immediate)
  (type_eqb : flambda_type -> flambda_type -> bool)
  (block_shape_eqb : block_shape -> block_shape -> bool).

(* What is assumed of the environment side: its invariant has
   witnesses, and its three functions meet their contracts. *)
Definition environment_side_sound : Prop :=
  valid_states_have_witnesses joined_env definitions_of_existentials
    state_valid
  ∧ n_way_join_simples_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples
  ∧ import_type_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      import_type
  ∧ n_way_join_env_extension_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      join_env_n_way_join_env_extension.

(* What is assumed of the reduction of a head of naked immediates: a
   number that has the head has the reduced head, which is then not
   bottom. *)
Definition reduce_head_of_kind_naked_immediate_sound : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (valuation : env)
    (v : value) (head : head_of_kind_naked_immediate),
    typing_env_satisfied_strictly statics valuation tenv ->
    head_of_kind_naked_immediate_satisfied statics valuation v head ->
    ∃ head' : head_of_kind_naked_immediate,
      reduce_head_of_kind_naked_immediate tenv head = Or_bottom.Ok head'
      ∧ head_of_kind_naked_immediate_satisfied statics valuation v head'.

(* What is assumed of [Expand_head.expand_head]: the expanded type has
   the values of the type; it is in scope and kinded where the type is;
   and it is of the kind of the type, in an environment whose canonical
   constants are of the kinds of their aliases, if the type is
   kinded. *)
Definition expand_head_sound : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    expand_head tenv ty = Some expanded ->
    (∀ (statics : static_env) (valuation : env) (v : value),
       typing_env_satisfied_strictly statics valuation tenv ->
       type_satisfied statics valuation v ty ->
       type_satisfied statics valuation v (Expanded_type.to_type expanded))
    ∧ (typing_env_well_formed tenv ->
       type_in_scope tenv ty ->
       type_in_scope tenv (Expanded_type.to_type expanded))
    ∧ (typing_env_kinded tenv ->
       type_kinded tenv ty ->
       type_kinded tenv (Expanded_type.to_type expanded))
    ∧ (constants_of_kind tenv ->
       type_kinded tenv ty ->
       Expanded_type.kind expanded = kind ty
       ∧ kind (Expanded_type.to_type expanded) = kind ty).

(* What is assumed of [Expand_head.expand_head] for the names that
   have equations inside a type: in an environment that has those of
   its types at their kinds, the expanded type has its own at their
   kinds, if the type has. *)
Definition expand_head_equations_kinded : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    expand_head tenv ty = Some expanded ->
    types_equations_kinded tenv ->
    type_equations_kinded tenv ty ->
    type_equations_kinded tenv (Expanded_type.to_type expanded).

(* What is assumed of [Expand_head.expand_head] for the well-formedness
   of types: in an environment whose types are well formed, a type
   that is well formed is expanded to one that is. *)
Definition expand_head_well_formed : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    expand_head tenv ty = Some expanded ->
    types_well_formed tenv ->
    type_well_formed ty = true ->
    type_well_formed (Expanded_type.to_type expanded) = true.

(* What is assumed of the join of extensions for the same: the
   extension that it answers is well formed if those given are. *)
Definition n_way_join_env_extension_well_formed : Prop :=
  ∀ (s s' : state) (extensions : list (Index.t * env_extension))
    (joined : env_extension),
    state_valid s ->
    (∀ (index : Index.t) (extension : env_extension),
       In (index, extension) extensions ->
       env_extension_well_formed extension = true) ->
    join_env_n_way_join_env_extension s extensions
    = Some (Or_bottom.Ok (joined, s')) ->
    env_extension_well_formed joined = true.

(* What is assumed of the kinds of the environments joined, beside
   [joined_envs_kinded]: in each, a constant that is the canonical
   element of a [Simple] of a kind is of that kind. *)
(* CR mshinwell: this is a premise about kinds, and the compiler does
   not check it: [Expand_head] answers the type of a constant that is
   the canonical element of an alias, whatever the kind of the alias
   (discrepancy T4-7), and [Aliases.add] does not look at kinds. It
   follows from the kinds premise of the contract,
   [joined_envs_kinded], which asks that the aliases of each
   environment joined are of one kind, where the well-formedness of
   environments has the aliases consistent
   (FlambdaJoinTypesWithEnvSpec.v); it is a premise here because this
   file is for any well-formedness. Whether ill-kinded code that the
   compiler accepts can make a name of one kind an alias of a constant
   of another is not known (discrepancy T5b-3). Decisions D10 and
   D12. *)
Definition joined_envs_constants_of_kind : Prop :=
  ∀ (s : state) (index : Index.t) (use : Typing_env.t),
    state_valid s ->
    joined_envs_well_formed typing_env_well_formed joined_env s ->
    joined_envs_kinded joined_env s ->
    joined_env s index = Some use -> constants_of_kind use.

(* What is assumed of the two tests of equality. *)
Definition type_eqb_sound : Prop :=
  ∀ t t' : flambda_type,
    type_eqb t t' = true ->
    ∀ (statics : static_env) (vars : env) (v : value),
      type_satisfied statics vars v t <-> type_satisfied statics vars v t'.

Definition block_shape_eqb_sound : Prop :=
  ∀ shape shape' : block_shape,
    block_shape_eqb shape shape' = true -> shape = shape'.

(* A thing given is in the scope of an environment and kinded there:
   the environment knows its variables, and has its aliases at their
   kinds. For a type these are [type_in_scope] and [type_kinded]. *)
Definition arg_in_scope_and_kinded {X : Type}
    (variables_of : X -> list variable)
    (aliases_of : X -> list (simple * flambda_kind)) (tenv : Typing_env.t)
    (x : X) : Prop :=
  (∀ y : variable, y ∈ variables_of x -> knows tenv y)
  ∧ ∀ (simple : simple) (kind : flambda_kind),
      (simple, kind) ∈ aliases_of x -> simple_of_kind tenv simple kind.

(* An answer mentions only variables of the target; and its aliases are
   at their kinds there. For a type these are [type_in_target] and
   [type_kinded_in_target]. *)
Definition answer_in_target {Y : Type} (variables_of : Y -> list variable)
    (s : state) (y : Y) : Prop :=
  ∀ x : variable,
    x ∈ variables_of y -> exists_in_target_env kind_in_target_env s x = true.

Definition answer_kinded_in_target {Y : Type}
    (aliases_of : Y -> list (simple * flambda_kind)) (s : state) (y : Y)
    : Prop :=
  ∀ (simple : simple) (kind : flambda_kind),
    (simple, kind) ∈ aliases_of y ->
    simple_of_kind_in_target joined_env kind_in_target_env s simple kind.

(* A function of the traversal is sound. It is given, at a state that
   is valid, things at distinct indices, each in scope and kinded in
   the environment of its index. If it answers, the state that it
   answers is valid and has grown; its answer is in the target; its
   answer is an upper bound, as [bound] says it, of each thing given,
   under each valuation of the target that extends one that satisfies
   the environment of the thing's index; and if the state given is
   kinded, so is the state answered, and the answer is kinded in the
   target. This is [n_way_join_type_sound], for any things, but for
   one clause of it that is of types only and is not said here: that
   the type answered is of the kind of each type given. *)
Definition join_step_sound {X Y : Type}
    (variables_of_arg : X -> list variable)
    (aliases_of_arg : X -> list (simple * flambda_kind))
    (variables_of_answer : Y -> list variable)
    (aliases_of_answer : Y -> list (simple * flambda_kind))
    (bound : static_env -> env -> X -> env -> Y -> Prop)
    (step : state -> list (Index.t * X) -> option (Y * state)) : Prop :=
  ∀ (s s' : state) (args : list (Index.t * X)) (answer : Y),
    state_valid s ->
    joined_envs_well_formed typing_env_well_formed joined_env s ->
    joined_envs_kinded joined_env s ->
    join_args_distinct args ->
    join_args_in_scope joined_env
      (arg_in_scope_and_kinded variables_of_arg aliases_of_arg) s args ->
    step s args = Some (answer, s') ->
    state_valid s'
    ∧ state_grows joined_env definitions_of_existentials kind_in_target_env s
        s'
    ∧ answer_in_target variables_of_answer s' answer
    ∧ (∀ (index : Index.t) (x : X) (use : Typing_env.t)
         (statics : static_env) (valuation valuation' : env),
         In (index, x) args ->
         joined_env s' index = Some use ->
         typing_env_satisfied_strictly statics valuation use ->
         extends_input joined_env definitions_of_existentials s' index
           valuation valuation' ->
         bound statics valuation x valuation' answer)
    ∧ (state_kinded s ->
       state_kinded s' ∧ answer_kinded_in_target aliases_of_answer s' answer).

(* ======================================================================== *)
(* The functions that join heads with no types inside them                  *)
(* ======================================================================== *)

(* [n_way_join_unknown] of a function that is sound is sound. *)
Definition n_way_join_unknown_sound : Prop :=
  ∀ (A B : Type) (variables_of_arg : A -> list variable)
    (aliases_of_arg : A -> list (simple * flambda_kind))
    (variables_of_answer : B -> list variable)
    (aliases_of_answer : B -> list (simple * flambda_kind))
    (bound : static_env -> env -> A -> env -> B -> Prop)
    (n_way_join_contents :
       state -> list (Index.t * A) -> option (Or_unknown.t B * state)),
    join_step_sound variables_of_arg aliases_of_arg
      (of_known variables_of_answer) (of_known aliases_of_answer)
      (known_bound bound) n_way_join_contents ->
    join_step_sound (of_known variables_of_arg) (of_known aliases_of_arg)
      (of_known variables_of_answer) (of_known aliases_of_answer)
      (or_unknown_bound bound) (n_way_join_unknown n_way_join_contents).

(* [n_way_join_relation_simples]: the variable answered, if there is
   one, holds in the target the number that each [Simple] given holds
   in its input. The [Simple]s are naked immediates. *)
Definition n_way_join_relation_simples_sound : Prop :=
  join_step_sound (fun _ : simple => [])
    (fun simple => [(simple, Naked_number Kind_naked_immediate)])
    free_variables_of_option aliases_of_relation
    (fun _ valuation simple valuation' x =>
       ∀ n : Z,
         eval_simple valuation simple
         = Some (Value_const (Naked_immediate n)) ->
         variable_is valuation' x n)
    (fun s simples =>
       n_way_join_relation_simples n_way_join_simples s (Some simples)).

(* The join of what heads of kind value say of null: what each head
   asks of a value that is null, and of one that is not, the answer
   asks no more of it. *)
Definition n_way_join_is_null_sound : Prop :=
  join_step_sound free_variables_of_head_of_kind_value
    aliases_of_head_of_kind_value variables_of_is_null aliases_of_is_null
    (fun _ valuation head valuation' is_null =>
       ∀ null : bool,
         is_null_says valuation (is_null_of_head head) null ->
         is_null_says valuation' is_null null)
    (n_way_join_is_null n_way_join_simples).

(* [n_way_join_head_of_kind_naked_immediate]. *)
Definition n_way_join_head_of_kind_naked_immediate_sound : Prop :=
  join_step_sound free_variables_of_naked_immediate (fun _ => [])
    (of_known free_variables_of_naked_immediate) (fun _ => [])
    (known_bound
       (all_kept head_of_kind_naked_immediate_satisfied
          head_of_kind_naked_immediate_satisfied))
    (n_way_join_head_of_kind_naked_immediate joined_env
       reduce_head_of_kind_naked_immediate).

(* [n_way_join_head_of_kind_naked_float32], ...: the first head is
   given apart from the others, as [n_way_join_expanded_head] gives
   it. *)
Definition n_way_join_head_of_kind_naked_numbers_sound : Prop :=
  ∀ k : naked_number_kind,
    join_step_sound (fun _ : list Z => []) (fun _ => [])
      (fun _ : Or_unknown.t (list Z) => []) (fun _ => [])
      (known_bound
         (all_kept (fun _ _ v numbers => naked_numbers_satisfied k v numbers)
            (fun _ _ v numbers => naked_numbers_satisfied k v numbers)))
      (fun s heads =>
         match heads with
         | [] => None
         | (id1, head1) :: heads =>
           n_way_join_head_of_kind_naked_numbers s (head1, id1) heads
         end).

(* [n_way_join_env_extension]: where an extension given holds in its
   input, the one answered holds in the target. *)
Definition n_way_join_env_extension_sound_here : Prop :=
  join_step_sound free_variables_of_env_extension aliases_of_env_extension
    free_variables_of_env_extension aliases_of_env_extension
    (fun statics valuation extension valuation' joined =>
       env_extension_satisfied statics valuation extension ->
       env_extension_satisfied statics valuation' joined)
    (n_way_join_env_extension join_env_n_way_join_env_extension).

(* What [deduce_is_int_simple] and [deduce_get_tag_simple] answer for a
   variant is true of its values: the [Simple] holds 1 for an immediate
   and 0 for a block; and the tag of a block. A variant of which
   [deduce_get_tag_simple] answers bottom has no block. *)
Definition deduce_is_int_simple_sound : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) (simple : simple),
    head_of_kind_value_non_null_satisfied statics vars v
      (Variant is_int immediates get_tag blocks extensions is_unique) ->
    deduce_is_int_simple blocks immediates is_int = Some simple ->
    (is_immediate v
     ∧ eval_simple vars simple = Some (Value_const (Naked_immediate 1)))
    ∨ (is_block statics v
       ∧ eval_simple vars simple = Some (Value_const (Naked_immediate 0))).

Definition deduce_get_tag_simple_sound : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) (tag : Z)
    (fields : list value),
    head_of_kind_value_non_null_satisfied statics vars v
      (Variant is_int immediates get_tag blocks extensions is_unique) ->
    block_of statics v = Some (tag, fields) ->
    match deduce_get_tag_simple blocks get_tag with
    | Bottom => False
    | Ok simple =>
      eval_simple vars simple = Some (Value_const (Naked_immediate tag))
    | Unknown => True
    end.

(* ======================================================================== *)
(* The type answered is well formed                                         *)
(* ======================================================================== *)

(* The environments joined have well-formed types. The join of types
   needs it of the types that it expands in them. *)
Definition joined_envs_types_well_formed (s : state) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    joined_env s index = Some use -> types_well_formed use.

(* The join of types, under the premises of [n_way_join_type_sound]: if
   the types given are well formed, the type answered is. The state
   is to be kinded, as for the names of the equations
   ([n_way_join_type_equations_kinded], FlambdaJoinSpec.v). *)
(* CR mshinwell: that the state is kinded is asked because the proof
   goes by the kinds clause of the contract of the join of types,
   which is under that premise. It is not known to be needed for
   well-formedness; a proof without it would go through every
   function of the traversal again. *)
Definition n_way_join_type_well_formed
    (n_way_join :
       state -> list (Index.t * flambda_type)
       -> option (Or_unknown.t flambda_type * state)) : Prop :=
  ∀ (s s' : state) (ts : list (Index.t * flambda_type)) (ty : flambda_type),
    state_valid s ->
    joined_envs_well_formed typing_env_well_formed joined_env s ->
    joined_envs_kinded joined_env s ->
    joined_envs_types_well_formed s ->
    join_args_distinct ts ->
    join_args_in_scope joined_env type_in_scope s ts ->
    join_args_in_scope joined_env type_kinded s ts ->
    (∀ (index : Index.t) (t : flambda_type),
       In (index, t) ts -> type_well_formed t = true) ->
    state_kinded s ->
    n_way_join s ts = Some (Or_unknown.Known ty, s') ->
    type_well_formed ty = true.

End Traversal.

(* The functions that join heads with no types inside them are sound,
   and what is deduced of [Is_int] and [Get_tag] is true. *)
Definition flat_heads_sound_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (state : Type)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (definitions_of_existentials :
       state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
    (kind_in_target_env : state -> variable -> option flambda_kind)
    (state_valid state_kinded : state -> Prop)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate),
    environment_side_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples import_type join_env_n_way_join_env_extension ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    n_way_join_unknown_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
    ∧ n_way_join_relation_simples_sound typing_env_well_formed joined_env
        definitions_of_existentials kind_in_target_env state_valid
        state_kinded n_way_join_simples
    ∧ n_way_join_is_null_sound typing_env_well_formed joined_env
        definitions_of_existentials kind_in_target_env state_valid
        state_kinded n_way_join_simples
    ∧ n_way_join_head_of_kind_naked_immediate_sound typing_env_well_formed
        joined_env definitions_of_existentials kind_in_target_env state_valid
        state_kinded reduce_head_of_kind_naked_immediate
    ∧ n_way_join_head_of_kind_naked_numbers_sound typing_env_well_formed
        joined_env definitions_of_existentials kind_in_target_env state_valid
        state_kinded
    ∧ n_way_join_env_extension_sound_here typing_env_well_formed joined_env
        definitions_of_existentials kind_in_target_env state_valid
        state_kinded join_env_n_way_join_env_extension
    ∧ deduce_is_int_simple_sound
    ∧ deduce_get_tag_simple_sound.

(* The join of types meets the contract that the join of environments
   assumes of it, whatever bounds the depth of its calls to itself. *)
Definition n_way_join_sound_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (state : Type)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (definitions_of_existentials :
       state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
    (kind_in_target_env : state -> variable -> option flambda_kind)
    (state_valid state_kinded : state -> Prop)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool) (fuel : nat),
    environment_side_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples import_type join_env_n_way_join_env_extension ->
    joined_envs_constants_of_kind typing_env_well_formed joined_env
      state_valid ->
    expand_head_sound typing_env_well_formed expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    type_eqb_sound type_eqb ->
    block_shape_eqb_sound block_shape_eqb ->
    n_way_join_type_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      (n_way_join joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* The same of the join of types with a bound on its expansions
   ([n_way_join_bounded], FlambdaJoinTypes.v), whatever the bound: to
   answer the unknown type where the bound is reached keeps the
   contract. *)
Definition n_way_join_bounded_sound_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (state : Type)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (definitions_of_existentials :
       state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
    (kind_in_target_env : state -> variable -> option flambda_kind)
    (state_valid state_kinded : state -> Prop)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool)
    (fuel depth : nat),
    environment_side_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples import_type join_env_n_way_join_env_extension ->
    joined_envs_constants_of_kind typing_env_well_formed joined_env
      state_valid ->
    expand_head_sound typing_env_well_formed expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    type_eqb_sound type_eqb ->
    block_shape_eqb_sound block_shape_eqb ->
    n_way_join_type_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      (n_way_join_bounded joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
         depth).

(* The upper bound is met: where the join answers a type, a value of a
   type given, under a valuation that satisfies its input, has the type
   answered under a valuation of the target that extends that one. *)
Definition n_way_join_bound_is_met_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (state : Type)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (definitions_of_existentials :
       state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
    (kind_in_target_env : state -> variable -> option flambda_kind)
    (state_valid state_kinded : state -> Prop)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool) (fuel : nat),
    environment_side_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples import_type join_env_n_way_join_env_extension ->
    joined_envs_constants_of_kind typing_env_well_formed joined_env
      state_valid ->
    expand_head_sound typing_env_well_formed expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    type_eqb_sound type_eqb ->
    block_shape_eqb_sound block_shape_eqb ->
    ∀ (s s' : state) (ts : list (Index.t * flambda_type)) (ty : flambda_type),
      state_valid s ->
      joined_envs_well_formed typing_env_well_formed joined_env s ->
      joined_envs_kinded joined_env s ->
      join_args_distinct ts ->
      join_args_in_scope joined_env type_in_scope s ts ->
      join_args_in_scope joined_env type_kinded s ts ->
      n_way_join joined_env n_way_join_simples import_type
        join_env_n_way_join_env_extension expand_head
        reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel s
        ts
      = Some (Or_unknown.Known ty, s') ->
      ∀ (index : Index.t) (t_index : flambda_type) (use : Typing_env.t)
        (statics : static_env) (valuation : env) (v : value),
        In (index, t_index) ts ->
        joined_env s' index = Some use ->
        typing_env_satisfied_strictly statics valuation use ->
        type_satisfied statics valuation v t_index ->
        ∃ valuation' : env,
          extends_input joined_env definitions_of_existentials s' index
            valuation valuation'
          ∧ type_satisfied statics valuation' v ty.

(* The join of types meets the contract for the names of the equations
   inside the type that it answers, whatever bounds the depth of its
   calls to itself: if the types given have the names of their
   equations at their kinds, each in the environment of its index, and
   the environments joined have those of their types, the type
   answered has its own at their kinds in the target. The import of a
   type and the join of extensions are to meet their contracts for
   those names too. *)
Definition n_way_join_equations_kinded_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (state : Type)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (definitions_of_existentials :
       state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
    (kind_in_target_env : state -> variable -> option flambda_kind)
    (state_valid state_kinded : state -> Prop)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool) (fuel : nat),
    environment_side_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples import_type join_env_n_way_join_env_extension ->
    import_type_equations_kinded typing_env_well_formed joined_env
      kind_in_target_env state_valid state_kinded import_type ->
    n_way_join_env_extension_equations_kinded typing_env_well_formed
      joined_env kind_in_target_env state_valid state_kinded
      join_env_n_way_join_env_extension ->
    joined_envs_constants_of_kind typing_env_well_formed joined_env
      state_valid ->
    expand_head_sound typing_env_well_formed expand_head ->
    expand_head_equations_kinded expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    type_eqb_sound type_eqb ->
    block_shape_eqb_sound block_shape_eqb ->
    n_way_join_type_equations_kinded typing_env_well_formed joined_env
      kind_in_target_env state_valid state_kinded
      (n_way_join joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* The type that the join of types answers is well formed, whatever
   bounds the depth of its calls to itself: if the types given are
   well formed and the environments joined have well-formed types, so
   is the type answered. The join of extensions is to answer a
   well-formed extension for well-formed extensions, and the expansion
   of a head a well-formed type for one. *)
Definition n_way_join_well_formed_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (state : Type)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (definitions_of_existentials :
       state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
    (kind_in_target_env : state -> variable -> option flambda_kind)
    (state_valid state_kinded : state -> Prop)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool) (fuel : nat),
    environment_side_sound typing_env_well_formed joined_env
      definitions_of_existentials kind_in_target_env state_valid state_kinded
      n_way_join_simples import_type join_env_n_way_join_env_extension ->
    n_way_join_env_extension_well_formed state_valid
      join_env_n_way_join_env_extension ->
    joined_envs_constants_of_kind typing_env_well_formed joined_env
      state_valid ->
    expand_head_sound typing_env_well_formed expand_head ->
    expand_head_well_formed expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    type_eqb_sound type_eqb ->
    block_shape_eqb_sound block_shape_eqb ->
    n_way_join_type_well_formed typing_env_well_formed joined_env state_valid
      state_kinded
      (n_way_join joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).
