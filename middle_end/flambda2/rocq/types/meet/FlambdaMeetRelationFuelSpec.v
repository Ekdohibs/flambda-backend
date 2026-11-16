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

(* The fuel of the meet that a [Switch] makes where it enters an arm:
   two units are enough, and the environment that the meet answers
   has again what is supposed of the types and of the aliases of the
   environment. First for a scrutinee that is the result of
   relations, met with the type of any constant; then for any type,
   met with the type of a number.

   This file is to be reviewed: the definitions and the statements.
   It goes on from FlambdaMeetTerminationSpec.v (what it is for the
   fuel to be enough; the meet of an alias of a name with the type of
   a constant) and from FlambdaMeetDepthSpec.v (the meet with a bare
   shape). The functions are those of FlambdaMeet.v and
   FlambdaMeetEnv.v. The proofs are in FlambdaMeetRelationFuelProofs.v,
   which needs no review.

   The meet. Where Simplify enters an arm of a [Switch] it meets the
   type of the scrutinee, an alias of a variable, with the type of the
   number of the arm, an alias of a constant. If the variable is the
   result of [Is_int], [Get_tag] or [Is_null] of a name, the type that
   the environment has for it is a head of naked immediates with an
   inverse relation to that name. The meet then does three things, one
   inside the other.
   - The variable is demoted to the constant, and the type that it
     had is met with the head of the constant
     (FlambdaMeetTerminationSpec.v, [alias_const_fuel_enough_statement]).
   - That is a meet of two heads of naked immediates. The numbers are
     met, and where they get fewer the relation says what the name it
     is a relation of can be, as a type that is added to that name
     ([reduce_inverse_relations]): [any_block] or
     [any_tagged_immediate] for [Is_int], [any_non_null_value] or
     [null] for [Is_null], and for [Get_tag] the type that the choices
     of the meet give for the tags.
   - Adding that type to the name meets it with the type that the
     environment has for the canonical element of the name.
   The last is the meet of a bare shape with a type of the environment
   (FlambdaMeetDepthSpec.v), for which one unit is enough; the second
   is one step more. A head with several relations has the second
   done for each in turn, each in the environment that the one before
   left.

   What is stated. [relation_const_fuel_enough_statement]: two units
   of fuel are enough for the meet of the two types, where
   - the canonical element that the meet reads for the first type is
     a name that is canonical, in aliases that are consistent, as in
     [alias_const_fuel_enough_statement];
   - the type that the environment has for that name is a head of
     naked immediates, with whatever inverse relations;
   - every type of kind value that the environment has for a name is
     an alias or has no extension ([values_extension_free]);
   - the types that the choices give for [Get_tag] are bare shapes of
     kind value ([get_tag_shapes]); those of the compiler's choice and
     of the repaired choice are ([choices_get_tag_shapes_statement]);
   - the join has [join_of_nothing] (FlambdaMeetDepthSpec.v); the
     join of the model has it where there is fuel for the join of
     environments ([model_join_of_nothing_statement], same file).
   The constant is any constant. Enough is [meet_fuel_enough]: the
   answer is the same whatever is answered where the fuel runs out.
   That the meet answers is not said.
   [relation_const_meet_keeps_statement]: under the same hypotheses,
   where the meet of that fuel or more answers an environment, every
   type of kind value that this environment has for a name is an
   alias or has no extension. So that hypothesis
   ([values_extension_free]) is kept by the meet for an arm.
   [relation_const_meet_keeps_aliases_statement]: and the aliases of
   that environment are consistent: the scrutinee, which was
   canonical, is made an alias of the constant, and a name that an
   equation of a relation is added to keeps its place among the
   aliases or, being canonical, is made an alias of a constant. What
   these three ask of the canonical element and of the type of a
   scrutinee is not shown of the environment answered; the two
   statements that follow do not ask it.

   [naked_immediate_fuel_enough_statement]: two units are enough for
   the meet of any type with the type of a number, as a naked
   immediate, under the last three of the hypotheses above and
   consistent aliases. Nothing is asked of the type, nor of what the
   environment has for it. The cases, of [Meet_env.meet_type], which
   is what [meet_fuel_enough] is about:
   - the type is of another kind: the meet answers nothing whatever
     the fuel (a fatal error of the compiler);
   - the type is not an alias: it is flat, and one unit is enough
     (FlambdaMeetTerminationSpec.v), or it is a head of naked
     immediates, whose relations are followed as above;
   - the type is an alias: the meet reads for it the canonical
     element of the class of what it is an alias of, whatever the
     name mode of that element (the meet reads at the name mode of
     the names of types, which is below the two others,
     [Name_mode.compare_partial_order]); with consistent aliases that
     element is canonical. (Where the environment does not bind what
     the type is an alias of, the meet reads nothing and answers
     nothing, whatever the fuel.) If it is a constant, the meet
     compares the two constants and asks no meet of expanded types.
     If it is a name, the type that the environment has for it at
     the kind is flat, or is a head of naked immediates (the case of
     [relation_const_fuel_enough_statement]), or is an alias, or
     there is none; in the last two the meet answers nothing
     whatever the fuel (fatal errors of the compiler: its
     [Expand_head] stops on a name that it takes for canonical and
     whose type is an alias).
   [naked_immediate_meet_keeps_statement]: where that meet answers an
   environment, with that fuel or more, the environment has again
   the two things that are supposed of the environment. So the meet
   for an arm of a [Switch] may be followed by another in the
   environment that it answers, whatever the scrutinee of each
   (FlambdaSimplifySwitchFuelSpec.v).

   What the hypotheses are for. Each relation adds one equation at
   most, and after it the environment is another, of which the third
   hypothesis is shown again: the type that the name of the relation
   is given is the shape, or the type that it had, or what the meet
   of the two makes, which has no extension
   ([bare_shape_meet_extension_free_statement],
   FlambdaMeetDepthSpec.v); or the name becomes an alias of a
   constant. Types of kind value with no extension: the type that the
   environment has for the name of the relation, or for what that
   name is an alias of, is met with the shape; with an extension in
   it the meet would add the equations of the extension, for which no
   bound is known. The name of the
   relation is not asked to be canonical, nor to be another than the
   scrutinee: whatever its canonical element is, the type of that is
   an alias or has no extension, or the canonical element is a
   constant and the shape is met with its head, or the name is one
   that an equation is being added to and the equation is passed over.

   An example ([example_relation_fuel_statement]): the environment
   after [let tag = Get_tag p] of FlambdaMeetExamplesSpec.v has what
   is supposed, so two units are enough for the arm for 0, with every
   join and all choices that have what is supposed of them; and with
   the meet that is to be proved and the join that forgets, one unit
   is not enough. A second ([example_two_relations_statement]): the
   same environment where [tag] is also the result of [Is_int] of a
   third variable. Two units are enough for the arm for 1, and with
   them the meet answers: the first variable, whose type had the
   variable for its tag, is given a type that the meet makes, and the
   third the type of the tagged immediates.

   What is not here: an environment in which a type of kind value has
   an extension; and what the meet answers, beyond what is said of
   its environment. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetTerminationSpec FlambdaMeetDepthSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is supposed of the environment and of the choices                   *)
(* ======================================================================== *)

(* Every type of kind value that the environment has for a name is an
   alias or has no extension. *)
Definition values_extension_free (tenv : Typing_env.t) : Prop :=
  ∀ (name : name) (ty : flambda_type),
    Typing_env.find tenv name (Some Value) = Some ty ->
    get_alias_opt ty <> None ∨ extension_free ty = true.

(* The types that the choices of the meet give for [Get_tag] are bare
   shapes, of kind value. *)
Definition get_tag_shapes (choices : meet_choices) : Prop :=
  ∀ (tags : list Z) (alloc_mode : Alloc_mode.t),
    bare_shape (type_of_get_tag_scrutinee choices tags alloc_mode) = true
    ∧ kind (type_of_get_tag_scrutinee choices tags alloc_mode) = Value.

Definition choices_get_tag_shapes_statement : Prop :=
  get_tag_shapes compiler_choices ∧ get_tag_shapes repaired_choices.

(* ======================================================================== *)
(* The meet for an arm                                                      *)
(* ======================================================================== *)

(* The meet of a type that is an alias of a canonical name with the
   type of a constant, where the type that the environment has for
   the name is a head of naked immediates. *)
Definition relation_const_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (tenv : Typing_env.t)
    (t : flambda_type) (n : name) (head : head_of_kind_naked_immediate)
    (c : const),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) (simple_of_name n) ->
    Typing_env.get_alias_then_canonical_simple_exn tenv
      (Some Name_mode.in_types) None t
    = Some (Some (simple_of_name n)) ->
    Typing_env.find tenv n (Some (kind t))
    = Some (Type_naked_immediate (Ok (No_alias head))) ->
    meet_fuel_enough join choices 2 tenv t (this_const c).

(* The environment that this meet answers, with that fuel or more, has
   [values_extension_free] again: its types of kind value are aliases
   or have no extension. *)
Definition relation_const_meet_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv tenv' : Typing_env.t) (t ty : flambda_type) (n : name)
    (head : head_of_kind_naked_immediate) (c : const),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) (simple_of_name n) ->
    Typing_env.get_alias_then_canonical_simple_exn tenv
      (Some Name_mode.in_types) None t
    = Some (Some (simple_of_name n)) ->
    Typing_env.find tenv n (Some (kind t))
    = Some (Type_naked_immediate (Ok (No_alias head))) ->
    2 <= fuel ->
    Meet.meet join choices fuel tenv t (this_const c)
    = Some (Or_bottom.Ok (ty, tenv')) ->
    values_extension_free tenv'.

(* And its aliases are consistent. *)
Definition relation_const_meet_keeps_aliases_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv tenv' : Typing_env.t) (t ty : flambda_type) (n : name)
    (head : head_of_kind_naked_immediate) (c : const),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) (simple_of_name n) ->
    Typing_env.get_alias_then_canonical_simple_exn tenv
      (Some Name_mode.in_types) None t
    = Some (Some (simple_of_name n)) ->
    Typing_env.find tenv n (Some (kind t))
    = Some (Type_naked_immediate (Ok (No_alias head))) ->
    2 <= fuel ->
    Meet.meet join choices fuel tenv t (this_const c)
    = Some (Or_bottom.Ok (ty, tenv')) ->
    aliases_consistent (Typing_env.aliases tenv').

(* ======================================================================== *)
(* Any type                                                                 *)
(* ======================================================================== *)

(* The meet of any type with the type of a number, as a naked
   immediate: two units of fuel are enough. Nothing is asked of the
   type, nor of the canonical element that the meet reads for it. *)
Definition naked_immediate_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (tenv : Typing_env.t)
    (t : flambda_type) (i : Z),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    meet_fuel_enough join choices 2 tenv t (this_naked_immediate i).

(* The environment that this meet answers, with that fuel or more, has
   again the two things that are supposed of the environment. *)
Definition naked_immediate_meet_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv tenv' : Typing_env.t) (t ty : flambda_type) (i : Z),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    2 <= fuel ->
    Meet.meet join choices fuel tenv t (this_naked_immediate i)
    = Some (Or_bottom.Ok (ty, tenv')) ->
    values_extension_free tenv'
    ∧ aliases_consistent (Typing_env.aliases tenv').

(* ======================================================================== *)
(* Two examples                                                             *)
(* ======================================================================== *)

(* The environment after [let tag = Get_tag p] of
   FlambdaMeetExamplesSpec.v, and the meet for the arm for 0 of a
   [Switch] on [tag]. The environment has what is supposed, so two
   units of fuel are enough, with every join and all choices that have
   what is supposed of them. One unit is not enough, with the meet
   that is to be proved and the join that forgets. *)
Definition example_relation_fuel_statement : Prop :=
  (∀ (join : join_scoped_envs) (choices : meet_choices),
    join_of_nothing join ->
    get_tag_shapes choices ->
    meet_fuel_enough join choices 2 FlambdaMeetExamplesSpec.example_get_tag_env
      FlambdaMeetExamplesSpec.example_tag_type
      FlambdaMeetExamplesSpec.example_zero_type)
  ∧ ¬ meet_fuel_enough FlambdaMeetExamplesSpec.forgetful_join repaired_choices
        1 FlambdaMeetExamplesSpec.example_get_tag_env
        FlambdaMeetExamplesSpec.example_tag_type
        FlambdaMeetExamplesSpec.example_zero_type.

(* The same with two relations: [tag] is also the result of [Is_int]
   of a third variable, [q], of which nothing else is known. *)
Definition example_q : variable := 3%positive.

Definition example_two_relations_env : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition
      FlambdaMeetExamplesSpec.example_get_tag_env example_q Value
      Name_mode.normal
  in
  Typing_env.replace_equation tenv
    (Name_var FlambdaMeetExamplesSpec.example_tag)
    (Type_naked_immediate
       (Ok
          (No_alias
             (Head_of_kind_naked_immediate Or_unknown.Unknown
                [(Relation.Get_tag,
                  Name_var FlambdaMeetExamplesSpec.example_p);
                 (Relation.Is_int, Name_var example_q)])))).

(* The types that the environment has for [p] and for [q] once the arm
   for 1 is entered, with the meet that is to be proved and the join
   that forgets, at two units of fuel. *)
Definition example_two_relations_types
    : option (option flambda_type * option flambda_type) :=
  match
    Meet.meet FlambdaMeetExamplesSpec.forgetful_join repaired_choices 2
      example_two_relations_env FlambdaMeetExamplesSpec.example_tag_type
      (this_naked_immediate 1)
  with
  | Some (Or_bottom.Ok (_, tenv')) =>
    Some
      (Typing_env.find tenv' (Name_var FlambdaMeetExamplesSpec.example_p)
         (Some Value),
       Typing_env.find tenv' (Name_var example_q) (Some Value))
  | Some Or_bottom.Bottom | None => None
  end.

(* Two units are enough, whatever the join and the choices that have
   what is supposed of them. With them the meet answers: [p], whose
   type had the variable for its tag, is given a type that the meet
   makes, the blocks of tag 1 with that variable, and [q] is given the
   type of the tagged immediates. *)
Definition example_two_relations_statement : Prop :=
  (∀ (join : join_scoped_envs) (choices : meet_choices),
    join_of_nothing join ->
    get_tag_shapes choices ->
    meet_fuel_enough join choices 2 example_two_relations_env
      FlambdaMeetExamplesSpec.example_tag_type (this_naked_immediate 1))
  ∧ example_two_relations_types
    = Some
        (Some
           (Type_value
              (Ok
                 (No_alias
                    (Head_of_kind_value
                       (Ok
                          (Variant None Or_unknown.Unknown
                             (Some FlambdaMeetExamplesSpec.example_tag)
                             (Or_unknown.Known
                                (Row_like_for_blocks
                                   [(1%Z, Or_unknown.Unknown)]
                                   Or_bottom.Bottom
                                   Alloc_mode.Heap_or_local))
                             No_extensions false))
                       Not_null)))),
         Some any_tagged_immediate).
